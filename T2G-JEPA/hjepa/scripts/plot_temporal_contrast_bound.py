"""Plot changes in B(k) and q(k) from one or more cached latent datasets.

No Torch, simulator or metric checkpoint is needed. By default, use the
horizon/gap of diagnostics.json next to the cache and check the reconstructed
Euclidean distance against that report. Within each dataset both curves share
the mean Euclidean squared distance as their normalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost_diagnostics import analyze_contrast_bound


def load_result(cache_path, args):
    path = Path(cache_path).expanduser().resolve()
    reference_path = path.with_name("diagnostics.json")
    if args.diagnostics:
        reference_path = Path(args.diagnostics).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
    with np.load(str(path), allow_pickle=False) as cache:
        episodes = [cache[k] for k in sorted(cache.files) if k.startswith("episode_")]
        metadata = json.loads(str(cache["metadata_json"])) if "metadata_json" in cache else {}
    horizon = args.horizon if args.horizon is not None else reference.get("horizon_sampled_steps", metadata.get("configured_horizon"))
    if horizon is None:
        raise ValueError("Specify --horizon when neither cache nor reference records it")
    gap = args.gap if args.gap is not None else reference.get("gap_sampled_steps", 1)
    result = analyze_contrast_bound(episodes, horizon, gap, args.far_window)
    result["metadata"] = metadata
    result["cache"] = str(path)
    result["reference_diagnostics"] = str(reference_path) if reference else None
    if reference.get("horizon_sampled_steps") == horizon and "metrics" in reference:
        expected = np.asarray(reference["metrics"]["euclidean"]["raw_mean_distance"])
        error = float(np.max(np.abs(expected - np.asarray(result["D_E"]))))
        result["max_reference_distance_error"] = error
        if not np.allclose(expected, result["D_E"], rtol=1e-9, atol=1e-9):
            raise ValueError("Cache and reference disagree: pair support/data may differ")
    result["stride_environment_steps"] = int(metadata.get(
        "stride_environment_steps", reference.get("stride_environment_steps", 1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, nargs="+", help="One executed_latents.npz per panel")
    parser.add_argument("--labels", nargs="+", help="Panel titles, in cache order")
    parser.add_argument("--diagnostics", help="Reference JSON (single cache only); otherwise use neighboring JSONs")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--gap", type=int)
    parser.add_argument("--far-window", type=int, default=5)
    parser.add_argument("--model-step-environment-steps", required=True, type=int,
                        help="Environment steps per model step (5 for flat AntMaze/FourRoom)")
    parser.add_argument("--output-dir", help="Required for multiple caches; default: next to single cache")
    args = parser.parse_args()
    if args.model_step_environment_steps < 1:
        parser.error("--model-step-environment-steps must be positive")
    if args.labels and len(args.labels) != len(args.cache):
        parser.error("Provide one --labels entry per cache")
    if len(args.cache) > 1 and (args.diagnostics or not args.output_dir):
        parser.error("Multiple caches require --output-dir and their own neighboring diagnostics.json")
    results = [load_result(path, args) for path in args.cache]
    for result in results:
        ratio = result["stride_environment_steps"] / args.model_step_environment_steps
        result["model_step_environment_steps"] = args.model_step_environment_steps
        result["near_offsets_model_steps"] = (np.asarray(result["near_offsets"]) * ratio).tolist()
        result["gap_model_steps"] = result["gap_sampled_steps"] * ratio
    labels = args.labels or [r["metadata"].get("config_name", "Dataset") for r in results]
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(results[0]["cache"]).parent
    out.mkdir(parents=True, exist_ok=True)
    report = {"panels": [{"label": label, "result": result} for label, result in zip(labels, results)],
              "horizontal_unit": "model steps",
              "plotted_quantities": ["abs(delta_B) / normalization", "abs(delta_q) / normalization"]}
    (out / "temporal_contrast_components.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.8),
                             squeeze=False, sharey=True)
    series = [
        ("abs_delta_B", r"$|\delta_\Delta B(k)|$", "#377eb8", "-"),
        ("delta_q", r"$|\delta_\Delta q(k)|$", "#e68a2e", "-"),
    ]
    for ax, result, title in zip(axes[0], results, labels):
        ks = np.asarray(result["near_offsets_model_steps"])
        for key, label, color, style in series:
            ax.plot(ks, np.abs(np.asarray(result[key])) / result["normalization"],
                    color=color, ls=style, label=label, marker="o", markersize=3, lw=1.6)
        ax.set_title(title + rf" ($\Delta={result['gap_model_steps']:g}$ model step)")
        ax.set_xlabel(r"Steps to the future goal, $k$ (model steps)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    axes[0, 0].set_ylabel(r"Change magnitude / $\overline{D}_E$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(out / ("temporal_contrast_components." + ext)), dpi=180)
    plt.close(fig)
    print("Saved:", out / "temporal_contrast_components.pdf")
    for label, result in zip(labels, results):
        print(label, "trajectories:", len(result["eligible_trajectories"]),
              "identity error:", result["max_distance_identity_error"],
              "reference distance error:", result.get("max_reference_distance_error"))


if __name__ == "__main__":
    main()
