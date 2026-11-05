"""Redraw temporal-signal curves from diagnostics.json in model steps.

Example (from hjepa/):
python scripts/plot_temporal_signal.py --diagnostics PATH/diagnostics.json \
    --model-step-environment-steps 5 --output-prefix PATH/temporal_signal_ant
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--model-step-environment-steps", required=True, type=int)
    parser.add_argument("--output-prefix", help="Output path without extension; default: temporal_signal next to JSON")
    args = parser.parse_args()
    if args.model_step_environment_steps < 1:
        parser.error("Model-step stride must be positive")
    source = Path(args.diagnostics).expanduser().resolve()
    result = json.loads(source.read_text(encoding="utf-8"))
    ratio = result["stride_environment_steps"] / args.model_step_environment_steps
    ks = np.asarray(result["remaining_steps"]) * ratio
    near = np.asarray(result["near_offsets"]) * ratio
    gap = result["gap_sampled_steps"] * ratio
    prefix = Path(args.output_prefix).expanduser().resolve() if args.output_prefix else source.parent / "temporal_signal"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for name, label in [("euclidean", "Euclidean squared"), ("t2g", "T2G")]:
        stats = result["metrics"][name]
        axes[0].plot(ks, stats["normalized_mean_distance"], label=label)
        axes[1].plot(near, stats["normalized_temporal_contrast"], label=label)
        axes[2].plot(near, stats["ranking_by_offset"], label=label)
    axes[0].set_title("Distance to the same future goals")
    axes[0].set_ylabel("Distance / mean distance")
    axes[1].set_title(rf"Temporal contrast ($\Delta={gap:g}$ model step)")
    axes[1].set_ylabel("Contrast / mean distance")
    axes[1].axhline(0, color="gray", ls="--")
    axes[2].set_title("Does the nearer state get lower cost?")
    axes[2].set_ylabel("Ranking accuracy (ties: half credit)")
    axes[2].axhline(.5, color="gray", ls="--", label="Chance / all ties")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel(r"Temporal separation, $k$ (model steps)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        path = Path(str(prefix) + "." + extension)
        fig.savefig(str(path), dpi=180)
        print("Saved:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
