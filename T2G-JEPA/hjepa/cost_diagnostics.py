"""Paired, scale-aware diagnostics of Euclidean and T2G goal costs.

Both scores always evaluate identical states and identical goals. Statistics
are averaged within each trajectory, then equally across trajectories. No
covariance/isotropy or calibrated-time assumptions are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


NAMES = ("euclidean", "t2g")


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Spearman inputs must be aligned vectors")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    a, b = _rank(x), _rank(y)
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else None


def _mean(values):
    valid = [x for x in values if x is not None]
    return float(np.mean(valid)) if valid else None


def _scores(raw, mapped, goal, metric_goal, scale, eps):
    return {
        "euclidean": np.sum((raw - goal) ** 2, axis=-1),
        "t2g": scale * np.maximum(np.linalg.norm(mapped - metric_goal, axis=-1), eps),
    }


def analyze_costs(episodes, metric_episodes, horizon, gap=1, far_window=5,
                  scale=1., eps=1e-6, goals=None, metric_goals=None, metadata=None):
    if horizon < 2 or not 1 <= gap < horizon or far_window < 1:
        raise ValueError("Need horizon >= 2, 1 <= gap < horizon, far_window >= 1")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(eps) or eps < 0:
        raise ValueError("Invalid metric scale or epsilon")
    if not episodes or len(episodes) != len(metric_episodes):
        raise ValueError("Need matching raw and T2G-embedded trajectories")
    raw_dim, metric_dim = None, None
    for x, y in zip(episodes, metric_episodes):
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("Each trajectory must contain aligned nonempty (T,D) arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite embeddings")
        raw_dim = x.shape[1] if raw_dim is None else raw_dim
        metric_dim = y.shape[1] if metric_dim is None else metric_dim
        if raw_dim != x.shape[1] or metric_dim != y.shape[1]:
            raise ValueError("Inconsistent embedding dimensions")
    metadata = metadata or {}
    eligible = [i for i, x in enumerate(episodes) if len(x) > horizon]
    if not eligible:
        raise ValueError(f"No trajectory has {horizon + 1} states; reduce --max-lag/horizon")
    ks = np.arange(1, horizon + 1)
    near = np.arange(1, horizon - gap + 1)
    far = near >= max(1, horizon - far_window)
    curves = {n: [] for n in NAMES}
    ranks = {n: [] for n in NAMES}
    ties = {n: [] for n in NAMES}
    correlations = {n: [] for n in NAMES}
    counts = []
    for i in eligible:
        x, y = np.asarray(episodes[i], dtype=float), np.asarray(metric_episodes[i], dtype=float)
        # Identical future goals at every offset: support does not change with k.
        g = np.arange(horizon, len(x))
        indices = g[:, None] - ks[None, :]
        scores = _scores(x[indices], y[indices], x[g, None], y[g, None], scale, eps)
        counts.append(len(g))
        for name, costs in scores.items():
            curves[name].append(costs.mean(axis=0))
            delta = costs[:, gap:] - costs[:, :-gap]
            ranks[name].append(((delta > 0) + .5 * (delta == 0)).mean(axis=0))
            ties[name].append((delta == 0).mean(axis=0))
            correlations[name].append(_mean([spearman(ks, row) for row in costs]))
    result = {
        "metadata": metadata, "num_trajectories": len(episodes),
        "eligible_trajectories": eligible, "excluded_short_trajectories": len(episodes) - len(eligible),
        "horizon_sampled_steps": horizon, "gap_sampled_steps": gap,
        "stride_environment_steps": int(metadata.get("stride_environment_steps", 1)),
        "goals_per_eligible_trajectory": counts, "remaining_steps": ks.tolist(),
        "near_offsets": near.tolist(), "far_near_offsets": near[far].tolist(),
        "metrics": {}, "execution": [],
        "notes": [
            "Both costs score exactly the same states and goals; future goals are fixed across all offsets.",
            "Each trajectory has equal weight, irrespective of its length or outcome.",
            "Ranking gives ties half credit. Flat costs score 0.5, not zero or perfect.",
            "Curves/contrasts divide by each score's mean over the tested offsets; raw means are saved.",
            "Temporal offsets on a recorded path are not shortest-path arrival times.",
            "Future-goal ordering on a stalled/looping execution is a temporal surrogate, not measured navigation progress.",
            "Execution progress uses actual task goals; success correlations use realized remaining time, not optimal arrival time.",
            "Trajectory clips and overlapping goals are dependent; no pair-level confidence intervals are claimed.",
            "Results may favor either score; this diagnostic does not assume T2G improves every task.",
        ],
    }
    for name in NAMES:
        curve = np.mean(curves[name], axis=0)
        rank_rows = np.asarray(ranks[name])
        normalization = float(curve.mean())
        normalized = curve / normalization if normalization > 0 else np.zeros_like(curve)
        result["metrics"][name] = {
            "raw_mean_distance": curve.tolist(), "normalization": normalization,
            "normalized_mean_distance": normalized.tolist(),
            "normalized_temporal_contrast": (normalized[gap:] - normalized[:-gap]).tolist(),
            "ranking_by_offset": rank_rows.mean(axis=0).tolist(),
            "overall_ranking": float(rank_rows.mean()),
            "far_ranking": float(rank_rows[:, far].mean()),
            "tie_fraction": float(np.mean(ties[name])),
            "mean_per_goal_spearman": _mean(correlations[name]),
        }
    result["far_ranking_improvement"] = result["metrics"]["t2g"]["far_ranking"] - result["metrics"]["euclidean"]["far_ranking"]
    tasks = metadata.get("tasks", [])
    if goals is not None and metric_goals is not None:
        if len(goals) != len(episodes) or len(metric_goals) != len(episodes):
            raise ValueError("Goals must align with every trajectory")
        for i, (x, y, g, mg) in enumerate(zip(episodes, metric_episodes, goals, metric_goals)):
            if g is None or mg is None:
                continue
            if np.asarray(g).shape != (raw_dim,) or np.asarray(mg).shape != (metric_dim,):
                raise ValueError("Actual-goal encodings must match their embedding dimensions")
            if not np.isfinite(g).all() or not np.isfinite(mg).all():
                raise ValueError("Non-finite actual-goal encoding")
            task = tasks[i] if i < len(tasks) else {}
            scores = _scores(x, y, g, mg, scale, eps)
            success = task.get("success")
            row = {"trajectory_index": i, "task_index": task.get("task_index", i),
                   "success": success, "end_environment_step": task.get("end_environment_step"),
                   "scores": {}}
            for name, cost in scores.items():
                row["scores"][name] = {
                    "cost": cost.tolist(),
                    "relative_to_initial": (cost / cost[0]).tolist() if cost[0] > 0 else None,
                    "spearman_vs_realized_remaining_time": spearman(-np.arange(len(cost)), cost) if success is True else None,
                }
            result["execution"].append(row)
    if not result["execution"]:
        result["notes"].append("No actual-goal encodings: execution-progress plot omitted; future-goal comparison remains valid.")
    result["planning_outcomes"] = {
        "num_tasks": len(tasks),
        "num_successes": sum(t.get("success") is True for t in tasks),
        "tasks": [{k: t.get(k) for k in ("task_index", "success", "end_environment_step")} for t in tasks],
    }
    return result


def analyze_contrast_bound(episodes, horizon, gap=1, far_window=5):
    """Reconstruct B and trace(C) on exactly the core report's pair measure.

    Pick an eligible trajectory uniformly, then a future goal uniformly from
    its indices horizon..T-1. At each k pair X=z[g-k] with Y=z[g]. Global
    weighted means are used for centering, so between-trajectory variation is
    retained. This is not a frame-weighted or per-episode-demeaned covariance.
    """
    if horizon < 2 or not 1 <= gap < horizon or far_window < 1:
        raise ValueError("Need horizon >= 2, 1 <= gap < horizon, far_window >= 1")
    raw = [np.asarray(z, dtype=np.float64) for z in episodes]
    if not raw or any(z.ndim != 2 or len(z) == 0 for z in raw):
        raise ValueError("Expected nonempty (T,D) trajectories")
    d = raw[0].shape[1]
    if d < 1 or any(z.shape[1] != d or not np.isfinite(z).all() for z in raw):
        raise ValueError("Inconsistent dimensions or non-finite latents")
    eligible = [i for i, z in enumerate(raw) if len(z) > horizon]
    if not eligible:
        raise ValueError("No trajectory supports the requested horizon")
    distances, bases, traces, mean_gaps, variance_traces = [], [], [], [], []
    for k in range(1, horizon + 1):
        pairs = []
        for i in eligible:
            goals = np.arange(horizon, len(raw[i]))
            pairs.append((raw[i][goals - k], raw[i][goals]))
        mx = np.mean([x.mean(axis=0) for x, _ in pairs], axis=0)
        my = np.mean([y.mean(axis=0) for _, y in pairs], axis=0)
        vx = float(np.mean([np.sum((x - mx) ** 2, axis=1).mean() for x, _ in pairs]))
        vy = float(np.mean([np.sum((y - my) ** 2, axis=1).mean() for _, y in pairs]))
        q = float(np.mean([np.sum((x - mx) * (y - my), axis=1).mean() for x, y in pairs]))
        mean_gap = float(np.sum((mx - my) ** 2))
        distances.append(float(np.mean([np.sum((x - y) ** 2, axis=1).mean() for x, y in pairs])))
        bases.append(mean_gap + vx + vy)
        traces.append(q); mean_gaps.append(mean_gap); variance_traces.append([vx, vy])
    distance, base, q = [np.asarray(x) for x in (distances, bases, traces)]
    contrast = distance[gap:] - distance[:-gap]
    db = base[gap:] - base[:-gap]
    dq = q[gap:] - q[:-gap]
    rhs = np.abs(db) + 2 * np.abs(dq)
    scale = float(distance.mean())
    normalizer = scale if scale > 0 else 1.
    offsets = np.arange(1, horizon - gap + 1)
    far = offsets >= max(1, horizon - far_window)
    if not np.any(far):
        raise ValueError("No far-offset comparisons: increase far_window to at least gap")
    result = {
        "horizon_sampled_steps": horizon, "gap_sampled_steps": gap,
        "far_window": far_window, "eligible_trajectories": eligible,
        "goals_per_eligible_trajectory": [len(raw[i]) - horizon for i in eligible],
        "excluded_short_trajectories": len(raw) - len(eligible),
        "remaining_steps": list(range(1, horizon + 1)),
        "near_offsets": offsets.tolist(), "far_near_offsets": offsets[far].tolist(),
        "D_E": distance.tolist(), "B": base.tolist(), "q": q.tolist(),
        "squared_mean_gap": mean_gaps, "marginal_variance_traces": variance_traces,
        "S_E": contrast.tolist(), "delta_B": db.tolist(), "delta_q": dq.tolist(),
        "abs_S_E": np.abs(contrast).tolist(), "abs_delta_B": np.abs(db).tolist(),
        "twice_abs_delta_q": (2 * np.abs(dq)).tolist(), "upper_bound": rhs.tolist(),
        "normalization": normalizer, "mean_squared_distance": scale,
        "max_distance_identity_error": float(np.max(np.abs(distance - (base - 2 * q)))),
        "max_contrast_identity_error": float(np.max(np.abs(contrast - (db - 2 * dq)))),
        "max_bound_violation": float(max(0., np.max(np.abs(contrast) - rhs))),
        "notes": [
            "Identical future goals at every k; equal trajectory weights, uniform goals within each trajectory.",
            "Global weighted means define covariance traces; no episode demeaning or stationarity assumption.",
            "The algebraic identity/bound must hold by construction; empirical evidence is the size of the two RHS terms.",
            "Absolute-value averages differ from the signed mean contrasts in temporal_signal.",
            "Small |S| alone can also result from cancellation; inspect both RHS terms and signed values in JSON.",
            "Finite execution data do not establish asymptotic convergence or identify causal planning failures.",
        ],
    }
    for label, selection in [("early", slice(0, min(far_window, len(offsets)))), ("far", far)]:
        result[label] = {
            "mean_abs_S_normalized": float(np.mean(np.abs(contrast[selection])) / normalizer),
            "mean_abs_delta_B_normalized": float(np.mean(np.abs(db[selection])) / normalizer),
            "mean_twice_abs_delta_q_normalized": float(2 * np.mean(np.abs(dq[selection])) / normalizer),
            "mean_upper_bound_normalized": float(rhs[selection].mean() / normalizer),
            "mean_signed_S_normalized": float(contrast[selection].mean() / normalizer),
        }
    tolerance = 1e-9 * max(1., float(np.max(np.abs(base))), float(np.max(np.abs(q))))
    if max(result["max_distance_identity_error"], result["max_contrast_identity_error"], result["max_bound_violation"]) > tolerance:
        raise ArithmeticError("Distance decomposition/bound failed numeric consistency checks")
    return result


def write_cost_report(episodes, metric_episodes, output_dir, **kwargs):
    result = analyze_costs(episodes, metric_episodes, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(str(out / (name + "." + ext)), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ks = np.asarray(result["remaining_steps"]) * result["stride_environment_steps"]
    near = np.asarray(result["near_offsets"]) * result["stride_environment_steps"]
    for name, label in [("euclidean", "Euclidean squared"), ("t2g", "T2G")]:
        stats = result["metrics"][name]
        axes[0].plot(ks, stats["normalized_mean_distance"], label=label)
        axes[1].plot(near, stats["normalized_temporal_contrast"], label=label)
        axes[2].plot(near, stats["ranking_by_offset"], label=label)
    axes[0].set_title("Distance to the same future goals")
    axes[0].set_ylabel("Distance / mean distance")
    axes[1].set_title("Temporal contrast (fixed gap)")
    axes[1].set_ylabel("Contrast / mean distance")
    axes[1].axhline(0, color="gray", ls="--")
    axes[2].set_title("Does the nearer state get lower cost?")
    axes[2].set_ylabel("Ranking accuracy (ties: half credit)")
    axes[2].axhline(.5, color="gray", ls="--", label="Chance / all ties")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Remaining steps (environment steps)")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "temporal_signal")
    if result["execution"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        time = np.linspace(0, 1, 101)
        for ax, success in zip(axes, [True, False]):
            cohort = [r for r in result["execution"] if r["success"] is success]
            ax.set_title(("Successful" if success else "Failed") + f" executions (n={len(cohort)})")
            for name, label in [("euclidean", "Euclidean squared"), ("t2g", "T2G")]:
                rows = [r["scores"][name]["relative_to_initial"] for r in cohort]
                rows = [np.asarray(r) for r in rows if r is not None and len(r) > 1]
                if rows:
                    aligned = [np.interp(time, np.linspace(0, 1, len(r)), r) for r in rows]
                    ax.plot(time, np.mean(aligned, axis=0), label=label)
            if not cohort:
                ax.text(.5, .5, "No executions in this group", ha="center", transform=ax.transAxes)
            else:
                ax.legend()
            ax.set_xlabel("Fraction of each recorded execution")
            ax.set_ylabel("Actual-goal cost / initial cost"); ax.grid(alpha=.2)
        save(fig, "planning_progress")
    return result
