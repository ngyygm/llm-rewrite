#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Curve Analysis for EMNLP 2026 Chinese Rewriting Evaluation.

Plots evaluator performance (Spearman correlation) as a function of
training data size for different evaluator approaches.

Uses real experiment results from all_results.json.

Usage:
    python learning_curves.py [--output-dir DIR] [--dpi 300]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALL_RESULTS = PROJECT_ROOT / "data" / "baselines" / "all_results.json"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "baselines" / "method_metadata.json"
DEFAULT_EVAL_DATA = PROJECT_ROOT / "data" / "human_eval" / "eval.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Method display names and styles
METHOD_STYLES = {
    "lora_balanced_simple": {"label": "LoRA Evaluator (Ours)", "color": "#2563EB", "marker": "o", "ls": "-", "lw": 2.5, "ms": 10},
    "prometheus2": {"label": "Prometheus 2 (7B)", "color": "#059669", "marker": "D", "ls": ":", "lw": 1.5, "ms": 7},
    "zeroshot_qwen14b": {"label": "Zero-shot Qwen2.5-14B", "color": "#D97706", "marker": "^", "ls": "-.", "lw": 1.5, "ms": 7},
    "zeroshot_qwen7b": {"label": "Zero-shot Qwen2.5-7B", "color": "#DC2626", "marker": "s", "ls": "--", "lw": 1.5, "ms": 7},
    "geval_qwen7b": {"label": "G-Eval (Qwen2.5-7B)", "color": "#9333EA", "marker": "v", "ls": "--", "lw": 1.5, "ms": 7},
    "lora_score_only_full": {"label": "LoRA (600 unbalanced)", "color": "#6B7280", "marker": "P", "ls": "--", "lw": 1.5, "ms": 7},
}


def compute_spearman(predictions, references):
    """Compute Spearman correlation, handling NaN."""
    p = np.array(predictions, dtype=float)
    r = np.array(references, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(r))
    if valid.sum() < 2:
        return 0.0
    rho, _ = stats.spearmanr(p[valid], r[valid])
    return round(float(rho), 4)


def load_real_data(all_results_path, metadata_path, eval_data_path):
    """Load real experiment results and organize into learning curves."""
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(all_results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    human_avg = [item["avg_score"] for item in eval_data]

    curves = {}

    # LoRA learning curve (subsets + full)
    subset_sizes = [50, 100, 200, 400]
    lora_curve = {}
    for size in subset_sizes:
        key = f"lora_balanced_simple_{size}"
        if key in all_results:
            lora_curve[size] = compute_spearman(all_results[key], human_avg)
    # Add full model (1008 balanced)
    if "lora_balanced_simple" in all_results:
        lora_curve[1008] = compute_spearman(all_results["lora_balanced_simple"], human_avg)
    if lora_curve:
        curves["lora_balanced_simple"] = {str(k): v for k, v in lora_curve.items()}

    # Constant baselines (no learning curve, just flat lines)
    constant_methods = ["zeroshot_qwen14b", "zeroshot_qwen7b", "geval_qwen7b",
                        "lora_score_only_full"]
    all_sizes = sorted([50, 100, 200, 400, 1008])
    for method in constant_methods:
        if method in all_results:
            rho = compute_spearman(all_results[method], human_avg)
            curves[method] = {str(sz): rho for sz in all_sizes}

    # Prometheus 2 (aggregate only, no per-sample predictions)
    if "prometheus2" in metadata:
        prom_rho = metadata["prometheus2"].get("spearman_vs_avg", 0)
        curves["prometheus2"] = {str(sz): prom_rho for sz in all_sizes}

    return curves, metadata


def plot_learning_curves(curves, metadata, output_dir, dpi=300):
    """Generate learning curve figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_sizes = set()
    for data in curves.values():
        all_sizes.update(int(k) for k in data.keys())
    train_sizes = sorted(all_sizes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    # Plot constant baselines first (dashed)
    for method_name, method_data in curves.items():
        if method_name == "lora_balanced_simple":
            continue  # Plot last (on top)
        style = METHOD_STYLES.get(method_name, {})
        sizes = sorted(int(k) for k in method_data.keys())
        corrs = [method_data[str(s)] for s in sizes]
        ax.plot(sizes, corrs,
                color=style.get("color", "#666"),
                marker=style.get("marker", "o"),
                linestyle=style.get("ls", "--"),
                linewidth=style.get("lw", 1.5),
                markersize=style.get("ms", 7),
                alpha=0.7,
                label=style.get("label", method_name))

    # Plot LoRA curve on top with fill
    if "lora_balanced_simple" in curves:
        lora_data = curves["lora_balanced_simple"]
        sizes = sorted(int(k) for k in lora_data.keys())
        corrs = [lora_data[str(s)] for s in sizes]

        style = METHOD_STYLES.get("lora_balanced_simple", {})
        ax.plot(sizes, corrs,
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=style["lw"],
                markersize=style["ms"], zorder=10,
                label=style["label"])
        ax.fill_between(sizes,
                        [c - 0.015 for c in corrs],
                        [c + 0.015 for c in corrs],
                        alpha=0.1, color=style["color"])

        # Annotate final point
        ax.annotate(f"{corrs[-1]:.3f}",
                     xy=(sizes[-1], corrs[-1]),
                     xytext=(10, 5), textcoords="offset points",
                     fontsize=9, color=style["color"], fontweight="bold")

    ax.set_xlabel("Training Samples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=13, fontweight="bold")
    ax.set_title("Learning Curves: Evaluator Performance vs Training Data Size",
                 fontsize=14, fontweight="bold", pad=12)

    ax.set_xscale("log")
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(s) for s in train_sizes])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(-0.2, 0.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.grid(True, which="minor", alpha=0.15, linestyle="--")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#CCC", fancybox=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=10)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/learning_curve.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_learning_curves_main_only(output_dir, dpi=300):
    """Generate a cleaner learning curve with only the LoRA subsets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_ALL_RESULTS, "r") as f:
        all_results = json.load(f)
    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    human_avg = [item["avg_score"] for item in eval_data]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # LoRA learning curve
    sizes = []
    corrs = []
    for subset_size in [50, 100, 200, 400]:
        key = f"lora_balanced_simple_{subset_size}"
        if key in all_results:
            rho = compute_spearman(all_results[key], human_avg)
            sizes.append(subset_size)
            corrs.append(rho)
    # Full model
    if "lora_balanced_simple" in all_results:
        rho = compute_spearman(all_results["lora_balanced_simple"], human_avg)
        sizes.append(1008)
        corrs.append(rho)

    ax.plot(sizes, corrs, "o-", color="#2563EB", linewidth=2.5, markersize=10, zorder=10,
            label="LoRA-balanced-simple (Ours)")

    # Baseline constant lines
    baseline_methods = [
        ("prometheus2", "Prometheus 2 (7B)", "#059669"),
        ("zeroshot_qwen14b", "Zero-shot Qwen2.5-14B", "#D97706"),
        ("zeroshot_qwen7b", "Zero-shot Qwen2.5-7B", "#DC2626"),
    ]
    with open(DEFAULT_METADATA, "r") as f:
        meta = json.load(f)
    for method, label, color in baseline_methods:
        if method in meta:
            rho = meta[method].get("spearman_vs_avg", 0)
            ax.axhline(y=rho, color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=label)

    # Annotate points
    for sz, rho in zip(sizes, corrs):
        ax.annotate(f"{rho:.3f}", xy=(sz, rho), xytext=(8, 5),
                     textcoords="offset points", fontsize=8, color="#2563EB", fontweight="bold")

    ax.set_xlabel("Training Samples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=13, fontweight="bold")
    ax.set_title("LoRA Evaluator Learning Curve", fontsize=14, fontweight="bold", pad=12)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(-0.2, 0.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=10)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/learning_curve_lora_only.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Learning curve analysis")
    parser.add_argument("--all-results", type=str, default=str(DEFAULT_ALL_RESULTS))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--eval-data", type=str, default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    curves, metadata = load_real_data(args.all_results, args.metadata, args.eval_data)

    print("Learning curve data (real):")
    for method, data in curves.items():
        display = metadata.get(method, {}).get("display_name", method)
        print(f"  {display}:")
        for sz in sorted(data.keys(), key=int):
            print(f"    n={sz}: rho={data[sz]:+.4f}")

    print("\nGenerating figures...")
    plot_learning_curves(curves, metadata, args.output_dir, args.dpi)
    plot_learning_curves_main_only(args.output_dir, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
