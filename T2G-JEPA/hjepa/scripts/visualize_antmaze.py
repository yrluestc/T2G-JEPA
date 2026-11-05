"""Save AntMaze offline-trajectory and successful-planning frame sheets.

Examples:
    python scripts/visualize_antmaze.py \
        --config-name ant_flat \
        --policy "$HJEPA_HOME/ckpts/ant/ant_lewm/seed42/ant_lewm_object.ckpt" \
        --eval-trajs assets/eval_trajs/ant/expert_grid_d3_n50.pt \
        --output-dir ../visualizations/antmaze

    # OGBench Cube + T2G planning-cost annotations
    python scripts/visualize_antmaze.py \
        --config-name cube_flat_metric \
        --policy "$HJEPA_HOME/ckpts/cube/cube_lewm/seed42/cube_lewm_object.ckpt" \
        --eval-trajs assets/eval_trajs/ogbench/goal_offset_20_pickup_val.pt \
        --output-dir ../visualizations/cube \
        --skip-offline

The offline part reads raw frames directly from the AntMaze H5 file. The
planning part runs one or more fixed eval tasks and records the environment
frames until success, then saves a contact sheet and optional GIF. When a T2G
checkpoint is available, each saved planning frame is annotated with
``d_T2G(current state, goal)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
HJEPA_DIR = HERE.parent
REPO_DIR = HJEPA_DIR.parent
sys.path.insert(0, str(HJEPA_DIR))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from planning_eval import (
    _build_eval_process_from_policy_normalizer,
    _build_policy,
    _resolve_existing_path,
    img_transform,
    load_model,
)
from temporal_metric import load_temporal_metric, make_temporal_pairs
from train_temporal_metric import normalize_uint8_pixels
from utils import resolve_model_checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="ant_flat")
    parser.add_argument("--policy", required=True, help="Flat LeWM checkpoint path.")
    parser.add_argument("--eval-trajs", default=None, help="Optional eval-trajs .pt override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(REPO_DIR / "visualizations" / "antmaze"))
    parser.add_argument("--cache-dir", default=None, help="Defaults to $HJEPA_HOME.")
    parser.add_argument("--dataset-name", default="visual_antmaze_medium_explore_stitch_train")
    parser.add_argument("--offline-episode", type=int, default=0)
    parser.add_argument("--offline-start", type=int, default=0)
    parser.add_argument("--offline-stride", type=int, default=5)
    parser.add_argument("--offline-num-frames", type=int, default=8)
    parser.add_argument("--metric-ckpt", default=None, help="T2G checkpoint for distance annotations.")
    parser.add_argument("--t2g-gap", type=int, default=2)
    parser.add_argument("--t2g-min-steps", type=int, default=1)
    parser.add_argument("--t2g-max-steps", type=int, default=None)
    parser.add_argument("--t2g-seed", type=int, default=42)
    parser.add_argument("--goal-source", choices=["auto", "env", "traj"], default="auto")
    parser.add_argument("--plan-task-index", type=int, default=0)
    parser.add_argument("--plan-max-tasks", type=int, default=10)
    parser.add_argument("--plan-num-frames", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-planning", action="store_true")
    return parser.parse_args()


def _resolve_metric_checkpoint(args: argparse.Namespace, cfg, cache_dir: str):
    """Resolve the T2G checkpoint, defaulting to one next to the policy ckpt."""
    metric_ckpt = args.metric_ckpt or cfg.get("metric_ckpt", None)
    if metric_ckpt:
        return str(_resolve_existing_path(metric_ckpt))

    try:
        ckpt_path = resolve_model_checkpoint_path(args.policy, cache_dir)
    except Exception:
        return None
    candidate = Path(ckpt_path).parent / "temporal_metric.pt"
    return str(candidate) if candidate.exists() else None


def _as_hwc(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 3:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_vector(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return None
    return array.reshape(-1).astype(np.float32, copy=False)


def _evenly_spaced(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[int(i)] for i in indices]


def _save_contact_sheet(
    frames: list[np.ndarray],
    labels: list[str],
    path: Path,
    *,
    cols: int,
    scale: int,
) -> None:
    if not frames:
        raise ValueError("no frames to save")
    cols = max(1, int(cols))
    scale = max(1, int(scale))
    height, width = frames[0].shape[:2]
    thumb_w, thumb_h = width * scale, height * scale
    label_h = 18
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (frame, label) in enumerate(zip(frames, labels)):
        image = Image.fromarray(frame).resize((thumb_w, thumb_h), Image.NEAREST)
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 3, y + 3), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_frames(
    frames: list[np.ndarray],
    labels: list[str],
    out_dir: Path,
    stem: str,
    *,
    cols: int,
    scale: int,
    gif: bool,
) -> None:
    frame_dir = out_dir / f"{stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{i:03d}.png")
    _save_contact_sheet(frames, labels, out_dir / f"{stem}.png", cols=cols, scale=scale)
    if gif:
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            out_dir / f"{stem}.gif",
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
        )


def _stack_vectors(
    vectors: list[np.ndarray],
    expected_dim: int | None = None,
) -> np.ndarray:
    """Match recorded proprioception to the training normalizer's layout.

    LocomazeEnv emits [qpos, qvel] (29 for Ant), whereas Ant training uses
    [qpos[2:], qvel] (27). Only this known conversion is supported: dropping
    the *first* two coordinates removes global xy and preserves all velocities.
    Arbitrary padding/truncation would silently corrupt diagnostic embeddings.
    """
    if not vectors or any(vector is None for vector in vectors):
        raise ValueError("Expected nonempty, present proprio vectors")
    values = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    dims = [value.size for value in values]
    if expected_dim is None:
        if len(set(dims)) != 1:
            raise ValueError(f"Mixed proprio dimensions {sorted(set(dims))}; training dimension is required")
        target = dims[0]
    else:
        target = int(expected_dim)
    if target < 1:
        raise ValueError("Expected a positive proprio dimension")
    aligned = []
    for value in values:
        if value.size == target:
            aligned.append(value)
        elif target == 27 and value.size == 29:
            aligned.append(value[2:])
        else:
            raise ValueError(
                f"Proprio has {value.size} features; training normalizer expects {target}. "
                "Only the known AntMaze 29->27 conversion ([qpos[2:], qvel]) is supported."
            )
    return np.stack(aligned).astype(np.float32, copy=False)


def _encode_frames(
    model,
    frames: list[np.ndarray],
    device,
    *,
    proprio: list[np.ndarray] | None = None,
    process: dict | None = None,
) -> torch.Tensor:
    array = np.stack(frames).astype(np.uint8, copy=False)
    pixels = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(1)
    batch = {"pixels": pixels.to(device)}
    if proprio is not None:
        expected_dim = None
        if process is not None and "proprio" in process:
            mean = getattr(process["proprio"], "mean_", None)
            if mean is not None:
                expected_dim = int(np.asarray(mean).reshape(-1).size)
            else:
                expected_dim = getattr(process["proprio"], "n_features_in_", None)
        values = _stack_vectors(proprio, expected_dim=expected_dim)
        if process is not None and "proprio" in process:
            values = process["proprio"].transform(values)
        batch["proprio"] = torch.from_numpy(values).float().to(device).unsqueeze(1)
    batch = normalize_uint8_pixels(batch)
    with torch.no_grad():
        return model.encode(batch)["embed_0"][:, 0]


def _t2g_distances(
    model,
    metric,
    frames: list[np.ndarray],
    goal: np.ndarray,
    device,
    *,
    proprio: list[np.ndarray] | None = None,
    goal_proprio: np.ndarray | None = None,
    process: dict | None = None,
) -> np.ndarray:
    embeddings = _encode_frames(
        model, frames, device, proprio=proprio, process=process
    )
    goal_embedding = _encode_frames(
        model,
        [goal],
        device,
        proprio=None if goal_proprio is None else [goal_proprio],
        process=process,
    )
    distances = metric(embeddings, goal_embedding)
    return distances.detach().cpu().numpy()


def _load_offline_frames(
    args: argparse.Namespace,
    cache_dir: str,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
    from stable_worldmodel.data import HDF5Dataset

    dataset = HDF5Dataset(
        args.dataset_name,
        keys_to_load=["pixels", "qpos", "qvel"],
        keys_to_cache=["qpos", "qvel"],
        keys_to_merge={"proprio": [{"key": "qpos", "start": 2}, {"key": "qvel"}]},
        cache_dir=cache_dir,
        level1={"frameskip": 1, "num_steps": 1},
    )
    episode = int(args.offline_episode)
    if episode < 0 or episode >= len(dataset.lengths):
        raise IndexError(f"episode {episode} out of range [0, {len(dataset.lengths)})")
    length = int(dataset.lengths[episode])
    stride = max(1, int(args.offline_stride))
    num_frames = max(1, int(args.offline_num_frames))
    span = (num_frames - 1) * stride
    start = max(0, min(int(args.offline_start), max(0, length - span - 1)))
    indices = start + np.arange(num_frames) * stride
    indices = indices[indices < length]
    if indices.size == 0:
        raise ValueError("offline segment has no valid frames")
    raw = dataset._load_slice(
        episode,
        int(indices[0]),
        int(indices[-1]) + 1,
        frameskip=1,
        apply_transform=False,
    )
    pixels = raw["pixels"]
    proprio = raw["proprio"]
    frames = []
    proprios = []
    valid_indices = []
    for index in indices:
        offset = int(index) - int(indices[0])
        frame = _as_hwc(pixels[offset])
        vector = _as_vector(proprio[offset])
        if frame is not None and vector is not None:
            frames.append(frame)
            proprios.append(vector)
            valid_indices.append(int(index))
    return frames, valid_indices, proprios


def _annotate_offline_t2g(
    args: argparse.Namespace,
    frames: list[np.ndarray],
    indices: list[int],
    proprios: list[np.ndarray],
    *,
    model,
    metric,
    process: dict,
    output_dir: Path,
) -> None:
    device = next(model.parameters()).device
    embeddings = _encode_frames(
        model,
        frames,
        device,
        proprio=proprios,
        process=process,
    ).unsqueeze(0)
    max_steps = args.t2g_max_steps
    if max_steps is None:
        max_steps = len(frames) - 1
    max_steps = max(
        int(args.t2g_min_steps) + int(args.t2g_gap),
        min(int(max_steps), len(frames) - 1),
    )
    generator = torch.Generator(device=device).manual_seed(int(args.t2g_seed))
    pairs = make_temporal_pairs(
        embeddings,
        max_steps=max_steps,
        min_steps=int(args.t2g_min_steps),
        gap=int(args.t2g_gap),
        generator=generator,
    )
    anchor = int(pairs["t"][0])
    near = anchor + int(pairs["k_near"][0])
    far = anchor + int(pairs["k_far"][0])
    anchor_embedding = embeddings[0, anchor : anchor + 1]
    d_anchor_all = metric(embeddings[0], anchor_embedding).detach().cpu().numpy()
    d_anchor_near = float(d_anchor_all[near])
    d_anchor_far = float(d_anchor_all[far])

    marks = {anchor: "A", near: "N", far: "F"}
    labels = []
    for position, raw_index in enumerate(indices):
        mark = marks.get(position, "")
        prefix = f"{mark} " if mark else ""
        distance = 0.0 if position == anchor else float(d_anchor_all[position])
        labels.append(f"{prefix}t={raw_index} | d(A,t)={distance:.3f}")
    _save_frames(
        frames,
        labels,
        output_dir / "offline",
        "offline_t2g_window",
        cols=args.cols,
        scale=args.scale,
        gif=args.gif,
    )

    triplet_positions = [anchor, near, far]
    triplet_frames = [frames[position] for position in triplet_positions]
    triplet_labels = [
        f"anchor | t={indices[anchor]} | d(A,A)=0.000",
        f"near | t={indices[near]} | dt={near - anchor} | d(A,N)={d_anchor_near:.3f}",
        f"far | t={indices[far]} | dt={far - anchor} | d(A,F)={d_anchor_far:.3f}",
    ]
    _save_frames(
        triplet_frames,
        triplet_labels,
        output_dir / "offline",
        "offline_t2g_triplet",
        cols=3,
        scale=args.scale,
        gif=args.gif,
    )
    metadata = {
        "episode": int(args.offline_episode),
        "start": int(indices[0]),
        "stride": int(args.offline_stride),
        "loss": "d(anchor, near) < d(anchor, far)",
        "triplet": {
            "anchor": int(indices[anchor]),
            "near": int(indices[near]),
            "far": int(indices[far]),
            "k_near": int(pairs["k_near"][0]),
            "k_far": int(pairs["k_far"][0]),
        },
        "d_anchor_near": d_anchor_near,
        "d_anchor_far": d_anchor_far,
        "d_anchor_all": [float(value) for value in d_anchor_all],
    }
    (output_dir / "offline" / "offline_t2g_triplet.json").write_text(
        json.dumps(metadata, indent=2)
    )


def _record_world(
    world,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray | None]], np.ndarray | None, np.ndarray | None]:
    frames: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    original_step = world.step

    def capture() -> None:
        infos = getattr(world, "infos", {})
        frame = _as_hwc(infos.get("pixels"))
        if frame is not None:
            frames.append((len(frames), frame, _as_vector(infos.get("proprio"))))

    def step(*call_args, **call_kwargs):
        if not frames:
            capture()
        result = original_step(*call_args, **call_kwargs)
        capture()
        return result

    world.step = step
    infos = getattr(world, "infos", {})
    goal = _as_hwc(infos.get("goal"))
    goal_proprio = _as_vector(infos.get("goal_proprio"))
    return frames, goal, goal_proprio


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "data" not in payload or "columns" not in payload:
        raise ValueError(f"{path} is not a H-JEPA eval-trajectory payload")
    return payload


def _last_image(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim < 3:
        return None
    return _as_hwc(array[-1])


def _goal_from_payload(
    payload: dict,
    task_index: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    data = payload["data"][task_index]
    goal = _last_image(data.get("pixels"))
    if goal is None:
        goal = _last_image(data.get("goal"))
    proprio = None
    value = data.get("proprio")
    if value is not None:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim >= 2:
            value = array[-1]
        proprio = _as_vector(value)
    if proprio is None:
        value = data.get("goal_proprio")
        if value is not None:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim >= 2:
                value = array[-1]
            proprio = _as_vector(value)
    return goal, proprio


def _reseed_policy(policy, seed: int) -> None:
    solver = getattr(policy, "solver", None)
    solvers = getattr(solver, "level_solvers", {1: solver})
    for level_solver in solvers.values():
        generator = getattr(level_solver, "torch_gen", None)
        if generator is not None:
            generator.manual_seed(int(seed))


def _close_recorded_world(world, previous_step_override, had_step_override):
    """Break the recording closure's reference cycle before environment close."""
    if had_step_override:
        world.__dict__["step"] = previous_step_override
    else:
        world.__dict__.pop("step", None)
    world.close()


def _run_planning_attempt(
    cfg,
    payload: dict,
    task_index: int,
    *,
    policy,
    process: dict,
    image_shape: tuple[int, int],
    goal_source: str,
) -> tuple[dict, list[tuple[int, np.ndarray, np.ndarray | None]], np.ndarray | None, np.ndarray | None, int]:
    import stable_worldmodel as swm

    data = payload["data"]
    if task_index < 0 or task_index >= len(data):
        raise IndexError(f"task {task_index} out of range [0, {len(data)})")
    episodes_idx = payload.get("episodes_idx")
    if episodes_idx is None or len(episodes_idx) == 0:
        episodes_idx = list(range(len(data)))
    chunk = {
        **payload,
        "data": [dict(data[task_index])],
        "episodes_idx": [episodes_idx[task_index]],
    }

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = 1
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    world = swm.World(**world_cfg, image_shape=image_shape)
    frames: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    goal = None
    goal_proprio = None
    had_step_override = "step" in world.__dict__
    previous_step_override = world.__dict__.get("step")
    try:
        world.set_policy(policy)
        frames, goal, goal_proprio = _record_world(world)
        callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            metrics = world.evaluate_from_dataset(
                None,
                episodes_idx=None,
                start_steps=None,
                goal_offset_steps=int(cfg.eval.goal_offset_steps),
                eval_budget=int(cfg.eval.eval_budget),
                callables=callables,
                load_eval_trajs_path=chunk,
                process=process,
                start_state_mode=cfg.eval.get("start_state_mode", "dataset_full"),
                goal_state_mode=cfg.eval.get("goal_state_mode", "dataset_full"),
            )
        env_goal = _as_hwc(getattr(world, "infos", {}).get("goal"))
        env_goal_proprio = _as_vector(getattr(world, "infos", {}).get("goal_proprio"))
        goal = None
        goal_proprio = None
        if goal_source in {"auto", "env"}:
            goal = env_goal
            goal_proprio = env_goal_proprio
        if goal is None and goal_source in {"auto", "traj"}:
            goal, goal_proprio = _goal_from_payload(payload, task_index)
    finally:
        _close_recorded_world(world, previous_step_override, had_step_override)
    return metrics, frames, goal, goal_proprio, task_index


def _run_planning(
    cfg,
    args: argparse.Namespace,
    *,
    model,
    metric,
    process: dict,
    transform: dict,
    output_dir: Path,
) -> None:
    eval_trajs = args.eval_trajs or cfg.get("load_eval_trajs_path", None)
    if not eval_trajs:
        raise ValueError("planning needs --eval-trajs or load_eval_trajs_path in the config")
    payload = _load_payload(Path(_resolve_existing_path(eval_trajs)))
    image_shape = (int(cfg.eval.img_size),) * 2
    policy = _build_policy(cfg, process, transform, model=model)

    start = int(args.plan_task_index)
    stop = min(len(payload["data"]), start + max(1, int(args.plan_max_tasks)))
    saved = False
    for task_index in range(start, stop):
        print(f"planning task {task_index} ...", flush=True)
        _reseed_policy(policy, int(args.seed) + task_index)
        metrics, frames, goal, goal_proprio, task_index = _run_planning_attempt(
            cfg,
            payload,
            task_index,
            policy=policy,
            process=process,
            image_shape=image_shape,
            goal_source=args.goal_source,
        )
        success = bool(metrics["episode_successes"][0])
        steps = metrics["steps_to_success"][0]
        print(f"  success={success} steps_to_success={steps}", flush=True)
        if not success:
            continue
        step_count = int(steps) if np.isfinite(steps) else len(frames) - 1
        selected = frames[: min(len(frames), step_count + 1)]
        selected = _evenly_spaced(selected, max(2, int(args.plan_num_frames)))
        selected_frames = [frame for _, frame, _ in selected]
        selected_proprios = [proprio for _, _, proprio in selected]
        if any(proprio is None for proprio in selected_proprios):
            selected_proprios = None
        labels = [f"t={step}" for step, _, _ in selected]
        t2g_values = None
        if metric is not None and goal is not None:
            try:
                device = next(model.parameters()).device
                distances = _t2g_distances(
                    model,
                    metric,
                    selected_frames,
                    goal,
                    device,
                    proprio=selected_proprios,
                    goal_proprio=goal_proprio,
                    process=process,
                )
                t2g_values = [float(distance) for distance in distances]
                labels = [
                    f"t={step} | dt={step_count - step} | d_T2G={float(distance):.3f}"
                    for (step, _, _), distance in zip(selected, distances)
                ]
            except Exception as exc:
                print(f"[warn] T2G distance annotation failed: {exc}", flush=True)
        _save_frames(
            selected_frames,
            labels,
            output_dir / "planning",
            f"planning_task{task_index:03d}_success",
            cols=args.cols,
            scale=args.scale,
            gif=args.gif,
        )
        if t2g_values is not None:
            metadata = {
                "task_index": int(task_index),
                "success": success,
                "steps_to_success": int(step_count),
                "goal_source": args.goal_source,
                "frames": [
                    {
                        "frame_index": int(frame_index),
                        "step": int(step),
                        "remaining_steps": int(step_count - step),
                        "d_T2G": float(distance),
                    }
                    for frame_index, ((step, _, _), distance) in enumerate(
                        zip(selected, t2g_values)
                    )
                ],
            }
            (output_dir / "planning" / f"planning_task{task_index:03d}_success.json").write_text(
                json.dumps(metadata, indent=2)
            )
        if goal is not None:
            Image.fromarray(goal).save(output_dir / "planning" / f"task{task_index:03d}_goal.png")
        print(f"saved planning visualization for task {task_index}", flush=True)
        saved = True
        break
    if not saved:
        raise RuntimeError(
            f"no successful task found in [{start}, {stop}); "
            "increase --plan-max-tasks or choose another --plan-task-index"
        )


def main() -> None:
    args = _parse_args()
    cache_dir = args.cache_dir or os.environ.get("HJEPA_HOME")
    if not cache_dir:
        raise RuntimeError("Set --cache-dir or $HJEPA_HOME")
    os.environ.setdefault("HJEPA_HOME", str(cache_dir))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(
        config_dir=str(HJEPA_DIR / "config" / "eval"),
        version_base=None,
    ):
        overrides = [
            f"seed={int(args.seed)}",
            f"policy={args.policy}",
        ]
        if args.eval_trajs:
            overrides.append(f"load_eval_trajs_path={args.eval_trajs}")
        cfg = compose(config_name=args.config_name, overrides=overrides)

    metric_ckpt = _resolve_metric_checkpoint(args, cfg, str(cache_dir))
    need_model = (not args.skip_planning) or metric_ckpt is not None
    model = load_model(cfg) if need_model else None
    metric = None
    process = {}
    if model is not None:
        device = next(model.parameters()).device
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)
        if metric_ckpt is not None and Path(metric_ckpt).exists():
            metric = load_temporal_metric(metric_ckpt, map_location=device)
            print(f"loaded T2G metric: {metric_ckpt}", flush=True)
        elif metric_ckpt is not None:
            print(f"[warn] T2G checkpoint not found: {metric_ckpt}", flush=True)
        else:
            print(
                "[warn] no T2G checkpoint resolved; T2G annotations disabled",
                flush=True,
            )

    if not args.skip_offline:
        frames, indices, proprios = _load_offline_frames(args, str(cache_dir))
        labels = [f"ep {args.offline_episode} | t={index}" for index in indices]
        _save_frames(
            frames,
            labels,
            output_dir / "offline",
            "offline_trajectory",
            cols=args.cols,
            scale=args.scale,
            gif=args.gif,
        )
        if metric is not None:
            _annotate_offline_t2g(
                args,
                frames,
                indices,
                proprios,
                model=model,
                metric=metric,
                process=process,
                output_dir=output_dir,
            )
        print(f"saved offline trajectory to {output_dir / 'offline'}", flush=True)

    if args.skip_planning:
        return

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    _run_planning(
        cfg,
        args,
        model=model,
        metric=metric,
        process=process,
        transform=transform,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
