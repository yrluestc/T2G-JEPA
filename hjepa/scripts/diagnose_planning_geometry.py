"""Compare Euclidean and T2G temporal guidance on identical execution states.

Supports ant/fourroom/cube/pusht via their existing eval configs. Includes
every selected task, successful or failed, rather than selecting paired wins.
Run from hjepa/: python scripts/diagnose_planning_geometry.py --help
Replay cached encoder latents without Torch/MuJoCo using --replay FILE.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HJEPA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HJEPA_DIR))
from cost_diagnostics import write_cost_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="fourroom_flat")
    p.add_argument("--policy", help="Flat LeWM object checkpoint")
    p.add_argument("--metric-ckpt")
    p.add_argument("--eval-trajs")
    p.add_argument("--cache-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plan-task-index", type=int, default=0)
    p.add_argument("--plan-max-tasks", type=int, default=50)
    p.add_argument("--costs", nargs="+", choices=["configured", "mse", "metric", "metric_dense"], default=["configured"])
    p.add_argument("--stride", type=int, help="Environment steps between observations; default: action_block")
    p.add_argument("--max-lag", type=int, help="Common future-goal horizon in sampled steps; default capped by shortest usable trajectory")
    p.add_argument("--gap", type=int, default=1, help="Remaining-step difference for paired ranking")
    p.add_argument("--far-window", type=int, default=5)
    p.add_argument("--geometry-details", action="store_true", help="Also generate legacy covariance diagnostics")
    p.add_argument("--max-pairs", type=int, default=4096, help=argparse.SUPPRESS)
    p.add_argument("--shuffle-repeats", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--pair-mode", choices=["available", "fixed"], default="available",
                   help=argparse.SUPPRESS)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--replay", help="Replay executed_latents.npz; no simulation or checkpoint needed")
    p.add_argument("--override", action="append", default=[], help="Additional Hydra override; repeat as needed")
    return p.parse_args()


def save_cache(path, episodes, metadata, metric_episodes=None, goals=None, metric_goals=None):
    arrays = {f"episode_{i:04d}": z for i, z in enumerate(episodes)}
    for prefix, values in [("metric", metric_episodes), ("goal", goals), ("metric_goal", metric_goals)]:
        if values is not None:
            arrays.update({f"{prefix}_{i:04d}": z for i, z in enumerate(values) if z is not None})
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(str(path), **arrays)


def report(args, episodes, metric_episodes, output, metadata, goals=None, metric_goals=None):
    stride = metadata["stride_environment_steps"]
    horizon = args.max_lag if args.max_lag is not None else metadata["configured_horizon"]
    if args.max_lag is None:
        # Keep short successful executions in the default common cohort.
        lengths = [len(z) - 1 for z in episodes if len(z) >= 3]
        horizon = min(horizon, min(lengths)) if lengths else horizon
    result = write_cost_report(
        episodes, metric_episodes, output, metadata=metadata, horizon=horizon,
        gap=args.gap, far_window=args.far_window,
        scale=metadata.get("metric_scale", 1.), eps=metadata.get("metric_eps", 1e-6),
        goals=goals, metric_goals=metric_goals)
    print(f"{output}: paired far-offset ranking: "
          f"Euclidean={result['metrics']['euclidean']['far_ranking']:.3f} "
          f"T2G={result['metrics']['t2g']['far_ranking']:.3f}; "
          f"horizon={horizon}, trajectories={len(result['eligible_trajectories'])}", flush=True)
    if args.geometry_details:
        from latent_geometry import write_geometry_report
        write_geometry_report(episodes, Path(output) / "geometry_details", metadata=metadata,
                              stride=stride, max_lag=horizon, max_pairs=args.max_pairs,
                              shuffle_repeats=args.shuffle_repeats, seed=args.seed, pair_mode=args.pair_mode)


def main():
    args = parse_args()
    if (args.max_lag is not None and args.max_lag < 2) or args.max_pairs < 2 or args.shuffle_repeats < 0 or args.gap < 1 or args.far_window < 1:
        raise ValueError("invalid lag, pair cap, or shuffle count")
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"]))
            episodes = [cache[k] for k in sorted(cache.files) if k.startswith("episode_")]
            mapped = [cache[f"metric_{i:04d}"] for i in range(len(episodes))] if "metric_0000" in cache else None
            goals = [cache[f"goal_{i:04d}"] if f"goal_{i:04d}" in cache else None for i in range(len(episodes))]
            metric_goals = [cache[f"metric_goal_{i:04d}"] if f"metric_goal_{i:04d}" in cache else None for i in range(len(episodes))]
        if args.stride is not None and args.stride != metadata["stride_environment_steps"]:
            raise ValueError("Replay stride must match the recorded sampling stride")
        if mapped is None or args.metric_ckpt:
            path = args.metric_ckpt or metadata.get("metric_ckpt")
            if not path:
                raise ValueError("Old cache lacks T2G embeddings; supply --metric-ckpt to replay it with Torch on CPU")
            import torch
            from temporal_metric import load_temporal_metric
            metric = load_temporal_metric(path, map_location="cpu")
            with torch.no_grad():
                mapped = [metric.embed(torch.as_tensor(z, dtype=torch.float32)).numpy() for z in episodes]
            metadata["metric_ckpt"] = str(path)
            metadata["metric_scale"] = float(metric.scale().detach().cpu())
            metadata["metric_eps"] = float(metric.eps)
            with torch.no_grad():
                metric_goals = [metric.embed(torch.as_tensor(g, dtype=torch.float32)).numpy()
                                if g is not None else None for g in goals]
        report(args, episodes, mapped, out, metadata, goals, metric_goals)
        return
    if not args.policy:
        raise ValueError("--policy is required unless using --replay")
    if args.plan_task_index < 0 or args.plan_max_tasks < 1 or args.encode_batch_size < 1:
        raise ValueError("invalid task range or encoder batch size")
    cache_dir = args.cache_dir or os.environ.get("HJEPA_HOME")
    if not cache_dir:
        raise ValueError("Set --cache-dir or HJEPA_HOME")
    os.environ["HJEPA_HOME"] = str(cache_dir)
    # Lazy imports keep cached diagnostics usable without the simulator stack.
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from eval_config_utils import _is_hierarchical_solver
    from planning_eval import (_build_eval_process_from_policy_normalizer,
                               _build_policy, _resolve_existing_path, img_transform, load_model)
    from temporal_metric import load_temporal_metric
    from visualize_antmaze import (_encode_frames, _load_payload, _reseed_policy,
                                   _resolve_metric_checkpoint, _run_planning_attempt)

    overrides = [f"seed={args.seed}", f"policy={args.policy}"]
    if args.eval_trajs:
        overrides.append(f"load_eval_trajs_path={args.eval_trajs}")
    with initialize_config_dir(config_dir=str(HJEPA_DIR / "config/eval"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides + args.override)
    if _is_hierarchical_solver(cfg):
        raise ValueError("Use a flat LeWM evaluation config")
    stride = args.stride if args.stride is not None else int(cfg.plan_config.action_block)
    if stride < 1:
        raise ValueError("--stride must be positive")
    configured_cost = str(cfg.get("cost", "mse"))
    OmegaConf.set_struct(cfg, False)
    cfg.cost = "mse"  # Attach a metric only when a selected cost needs it.
    model = load_model(cfg).eval().requires_grad_(False)
    device = next(model.parameters()).device
    path = _resolve_metric_checkpoint(args, cfg, str(cache_dir))
    if path is None:
        raise ValueError("Paired cost diagnosis needs a T2G checkpoint, even for MSE-only planning")
    metric = load_temporal_metric(path, map_location=device)
    model.temporal_metric = metric
    cfg.metric_ckpt = str(path)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    payload_path = _resolve_existing_path(cfg.load_eval_trajs_path)
    payload = _load_payload(Path(payload_path))
    start = args.plan_task_index
    stop = min(len(payload["data"]), start + args.plan_max_tasks)
    if stop <= start:
        raise ValueError("No tasks in selected range")
    costs = dict.fromkeys(configured_cost if c == "configured" else c for c in args.costs)
    for cost in costs:
        if cost not in {"mse", "metric", "metric_dense"}:
            raise ValueError(f"Unsupported planning cost: {cost}")
        cfg.cost = cost
        if cost.startswith("metric") and getattr(model, "temporal_metric", None) is None:
            path = _resolve_metric_checkpoint(args, cfg, str(cache_dir))
            if path is None:
                raise ValueError("A temporal metric checkpoint is needed for this cost")
            model.temporal_metric = load_temporal_metric(path, map_location=device)
            cfg.metric_ckpt = path
        model.cost_type = cost
        model.metric_last_n = cfg.get("metric_last_n", None)
        policy = _build_policy(cfg, process, transform, model=model)
        branch = out / cost
        branch.mkdir(parents=True, exist_ok=True)
        (branch / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        episodes, mapped, goals, metric_goals, tasks = [], [], [], [], []
        metadata = {
            "config_name": args.config_name, "policy": str(args.policy),
            "eval_trajs": str(payload_path), "seed": args.seed, "cost": cost,
            "environment": str(cfg.world.env_name),
            "metric_ckpt": str(path), "schema_version": 2,
            "metric_scale": float(metric.scale().detach().cpu()), "metric_eps": float(metric.eps),
            "stride_environment_steps": stride,
            "configured_horizon": int(cfg.plan_config.horizon),
            "sampling": "actual observations, all selected tasks, truncated at first success",
            "tasks": tasks,
        }
        for task in range(start, stop):
            torch.manual_seed(args.seed + task)
            np.random.seed(args.seed + task)
            random.seed(args.seed + task)
            _reseed_policy(policy, args.seed + task)
            metrics, frames, goal, goal_proprio, _ = _run_planning_attempt(
                cfg, payload, task, policy=policy, process=process,
                image_shape=(int(cfg.eval.img_size),) * 2, goal_source="auto")
            success = bool(metrics["episode_successes"][0])
            steps = metrics["steps_to_success"][0]
            end = int(steps) if success and np.isfinite(steps) else int(cfg.eval.eval_budget)
            frames = [f for f in frames if f[0] <= end and f[0] % stride == 0]
            if not frames:
                raise RuntimeError(f"Task {task}: no execution frames captured")
            zs = []
            for i in range(0, len(frames), args.encode_batch_size):
                chunk = frames[i:i + args.encode_batch_size]
                proprios = [f[2] for f in chunk]
                if any(v is None for v in proprios):
                    if "proprio" in process:
                        raise RuntimeError(f"Task {task}: missing required proprioception")
                    proprios = None
                z = _encode_frames(model, [f[1] for f in chunk], device,
                                   proprio=proprios, process=process)
                zs.append(z.detach().float().cpu().numpy())
            episodes.append(np.concatenate(zs))
            with torch.no_grad():
                mapped.append(metric.embed(torch.as_tensor(episodes[-1], device=device)).float().cpu().numpy())
            if goal is not None:
                if "proprio" in process and goal_proprio is None:
                    raise RuntimeError(f"Task {task}: missing goal proprioception")
                gz = _encode_frames(model, [goal], device,
                                    proprio=[goal_proprio] if goal_proprio is not None else None, process=process)
                goals.append(gz[0].detach().float().cpu().numpy())
                with torch.no_grad():
                    metric_goals.append(metric.embed(gz)[0].float().cpu().numpy())
            else:
                goals.append(None); metric_goals.append(None)
            tasks.append({"task_index": task, "success": success, "end_environment_step": end,
                          "sampled_environment_steps": [f[0] for f in frames]})
            # Cache after every task: statistics can be re-run without planning.
            save_cache(branch / "executed_latents.npz", episodes, metadata, mapped, goals, metric_goals)
            print(f"{cost} task {task}: success={success}, samples={len(episodes[-1])}", flush=True)
        report(args, episodes, mapped, branch, metadata, goals, metric_goals)


if __name__ == "__main__":
    main()
