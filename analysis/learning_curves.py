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
import re
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
DEFAULT_TABLES_DIR = PROJECT_ROOT / "analysis" / "results"

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Method display names and styles
METHOD_STYLES = {
    "lora_balanced_simple": {"label": "RewriteJudge (Ours, Qwen3-8B)", "color": "#2563EB", "marker": "o", "ls": "-", "lw": 2.5, "ms": 10},
    "prometheus2": {"label": "Prometheus 2 (7B)", "color": "#059669", "marker": "D", "ls": ":", "lw": 1.5, "ms": 7},
    "zeroshot_qwen3_14b": {"label": "Zero-shot Qwen3-14B", "color": "#D97706", "marker": "^", "ls": "-.", "lw": 1.5, "ms": 7},
    "zeroshot_qwen3_8b": {"label": "Zero-shot Qwen3-8B", "color": "#DC2626", "marker": "s", "ls": "--", "lw": 1.5, "ms": 7},
    "geval_qwen3_8b": {"label": "G-Eval (Qwen3-8B)", "color": "#9333EA", "marker": "v", "ls": "--", "lw": 1.5, "ms": 7},
    "lora_score_only_unbalanced": {"label": "Unbalanced training (600)", "color": "#6B7280", "marker": "P", "ls": "--", "lw": 1.5, "ms": 7},
    # Keep old names as fallback
    "zeroshot_qwen14b": {"label": "Zero-shot Qwen2.5-14B", "color": "#D97706", "marker": "^", "ls": "-.", "lw": 1.5, "ms": 7},
    "zeroshot_qwen7b": {"label": "Zero-shot Qwen2.5-7B", "color": "#DC2626", "marker": "s", "ls": "--", "lw": 1.5, "ms": 7},
    "geval_qwen7b": {"label": "G-Eval (Qwen2.5-7B)", "color": "#9333EA", "marker": "v", "ls": "--", "lw": 1.5, "ms": 7},
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


def load_curve_from_results_dir(results_dir):
    """
    Load LoRA learning-curve Spearman values directly from a results folder.
    Expected files include:
      - results_lora_score_only_balanced_full.json / results_balanced_simple.json
      - results_lora_score_only_balanced_<N>.json / results_balanced_simple_<N>.json / results_lora_score_only_<N>.json
    """
    results_dir = Path(results_dir)
    curve = {}

    full_candidates = [
        "results_lora_score_only_balanced_full.json",
        "results_lora_score_only_detail_balanced_full.json",
        "results_balanced_simple.json",
    ]
    for name in full_candidates:
        p = results_dir / name
        if p.exists():
            d = json.load(open(p))
            rho = d.get("metrics_vs_avg_score", {}).get("spearman_rho")
            if rho is not None:
                curve[1008] = float(rho)
            break

    subset_candidates = {}
    for p in results_dir.glob("*.json"):
        m = re.match(r"results_lora_score_only_balanced_(\d+)\.json$", p.name)
        if not m:
            m = re.match(r"results_lora_score_only_detail_balanced_(\d+)\.json$", p.name)
        if not m:
            m = re.match(r"results_balanced_simple_(\d+)\.json$", p.name)
        if not m:
            m = re.match(r"results_lora_score_only_(\d+)\.json$", p.name)
        if m:
            subset_candidates[int(m.group(1))] = p

    for size, p in sorted(subset_candidates.items()):
        try:
            d = json.load(open(p))
            rho = d.get("metrics_vs_avg_score", {}).get("spearman_rho")
            if rho is not None:
                curve[int(size)] = float(rho)
        except Exception:
            continue

    return curve


def collect_baseline_lines(model_family, metadata, results_dir):
    """Collect constant baseline lines for model-specific learning curve plots."""
    lines = []
    results_dir = Path(results_dir)

    # Unbalanced LoRA from the same run directory (if available).
    for name in ["results_lora_score_only_unbalanced_full.json", "results_unbalanced_simple.json"]:
        p = results_dir / name
        if p.exists():
            try:
                d = json.load(open(p))
                rho = d.get("metrics_vs_avg_score", {}).get("spearman_rho")
                if rho is not None:
                    lines.append(("Unbalanced training", float(rho), "#6B7280", ":"))
            except Exception:
                pass
            break

    # Shared baselines from consolidated metadata.
    if "prometheus2" in metadata:
        lines.append(("Prometheus 2 (7B)", float(metadata["prometheus2"].get("spearman_vs_avg", 0)), "#059669", "--"))

    if model_family == "qwen3":
        preferred = [
            ("zeroshot_qwen3_14b", "Zero-shot Qwen3-14B", "#D97706"),
            ("zeroshot_qwen3_8b", "Zero-shot Qwen3-8B", "#DC2626"),
            ("geval_qwen3_8b", "G-Eval (Qwen3-8B)", "#9333EA"),
        ]
        fallback = [
            ("zeroshot_qwen14b", "Zero-shot Qwen2.5-14B", "#D97706"),
            ("zeroshot_qwen7b", "Zero-shot Qwen2.5-7B", "#DC2626"),
            ("geval_qwen7b", "G-Eval (Qwen2.5-7B)", "#9333EA"),
        ]
    else:
        preferred = [
            ("zeroshot_qwen14b", "Zero-shot Qwen2.5-14B", "#D97706"),
            ("zeroshot_qwen7b", "Zero-shot Qwen2.5-7B", "#DC2626"),
            ("geval_qwen7b", "G-Eval (Qwen2.5-7B)", "#9333EA"),
        ]
        fallback = [
            ("zeroshot_qwen3_14b", "Zero-shot Qwen3-14B", "#D97706"),
            ("zeroshot_qwen3_8b", "Zero-shot Qwen3-8B", "#DC2626"),
            ("geval_qwen3_8b", "G-Eval (Qwen3-8B)", "#9333EA"),
        ]

    seen_roles = set()
    role_map = {"14b": "zs14b", "8b": "zs8b", "geval": "geval"}
    for key, label, color in preferred + fallback:
        if key not in metadata:
            continue
        role = "geval" if "geval" in key else ("14b" if "14b" in key else "8b")
        role = role_map[role]
        if role in seen_roles:
            continue
        seen_roles.add(role)
        lines.append((label, float(metadata[key].get("spearman_vs_avg", 0)), color, "--"))
    return lines


def plot_single_model_curve(curve, model_label, output_dir, file_stem, dpi=300, baseline_lines=None):
    """Plot one model family's learning curve only (with constant baselines)."""
    if not curve:
        print(f"  [skip] No curve data for {model_label}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sizes = sorted(curve.keys())
    corrs = [curve[s] for s in sizes]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # Constant baselines (same style spirit as learning_curve / lora_only).
    for label, rho, color, ls in (baseline_lines or []):
        ax.plot(
            sizes, [rho] * len(sizes),
            color=color, linestyle=ls, linewidth=1.4, alpha=0.75, label=label,
        )

    # Main LoRA curve on top with uncertainty band.
    ax.plot(
        sizes, corrs, "o-",
        color="#2563EB", linewidth=2.5, markersize=9, zorder=10, label=model_label,
    )
    ax.fill_between(
        sizes,
        [c - 0.015 for c in corrs],
        [c + 0.015 for c in corrs],
        alpha=0.1, color="#2563EB",
    )
    for sz, rho in zip(sizes, corrs):
        ax.annotate(
            f"{rho:.3f}", xy=(sz, rho), xytext=(8, 5),
            textcoords="offset points", fontsize=8, color="#2563EB", fontweight="bold"
        )

    ax.set_xlabel("Training Samples", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=12, fontweight="bold")
    ax.set_title(f"Learning Curve ({model_label})", fontsize=13, fontweight="bold", pad=10)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(-0.2, 0.7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/{file_stem}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def write_two_model_curve_table(curve25, curve3, tables_dir):
    """Write one LaTeX table containing Qwen2.5-7B and Qwen3-8B curves."""
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    all_sizes = sorted(set(curve25.keys()) | set(curve3.keys()))
    if not all_sizes:
        print("  [skip] No curve data for table.")
        return

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Learning curve comparison between Qwen2.5-7B and Qwen3-8B evaluators.}")
    lines.append(r"\label{tab:learning_curve_two_models}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Training Samples & Qwen2.5-7B Spearman $\rho$ & Qwen3-8B Spearman $\rho$ \\")
    lines.append(r"\midrule")
    for s in all_sizes:
        v25 = curve25.get(s)
        v3 = curve3.get(s)
        v25_str = f"{v25:+.4f}" if v25 is not None else "-"
        v3_str = f"{v3:+.4f}" if v3 is not None else "-"
        label = f"{s} (full)" if s == 1008 else str(s)
        lines.append(f"{label} & {v25_str} & {v3_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = Path(tables_dir) / "learning_curve_two_models_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def plot_two_model_curves_combined(curve25, curve3, output_dir, dpi=300):
    """Plot Qwen2.5-7B and Qwen3-8B learning curves in one figure."""
    if not curve25 and not curve3:
        print("  [skip] No curve data for combined plot.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.8))

    def _plot_curve(curve, label, color, marker):
        if not curve:
            return
        sizes = sorted(curve.keys())
        corrs = [curve[s] for s in sizes]
        ax.plot(
            sizes, corrs,
            color=color, marker=marker, linestyle="-", linewidth=2.3, markersize=8,
            label=label, zorder=10
        )
        ax.fill_between(
            sizes,
            [c - 0.015 for c in corrs],
            [c + 0.015 for c in corrs],
            alpha=0.08, color=color,
        )
        for sz, rho in zip(sizes, corrs):
            ax.annotate(
                f"{rho:.3f}", xy=(sz, rho), xytext=(6, 4),
                textcoords="offset points", fontsize=7.5, color=color, fontweight="bold"
            )

    _plot_curve(curve25, "RewriteJudge (Qwen2.5-7B)", "#DC2626", "s")
    _plot_curve(curve3, "RewriteJudge (Qwen3-8B)", "#2563EB", "o")

    all_sizes = sorted(set(curve25.keys()) | set(curve3.keys()))
    if all_sizes:
        ax.set_xscale("log")
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([str(s) for s in all_sizes])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("Training Samples", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=12, fontweight="bold")
    ax.set_title("Learning Curves: Qwen2.5-7B vs Qwen3-8B", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.2, 0.7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/learning_curve_qwen2.5-7b_vs_qwen3-8b.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


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
    # Prefer Qwen3 methods, fall back to Qwen2.5 if not available
    constant_methods = [
        "zeroshot_qwen3_14b", "zeroshot_qwen3_8b", "geval_qwen3_8b",
        "lora_score_only_unbalanced",
        # fallbacks
        "zeroshot_qwen14b", "zeroshot_qwen7b", "geval_qwen7b",
    ]
    # Deduplicate: skip fallback if Qwen3 version already added
    seen_roles = set()
    role_map = {
        "zeroshot_qwen3_14b": "zs14b", "zeroshot_qwen14b": "zs14b",
        "zeroshot_qwen3_8b": "zs8b", "zeroshot_qwen7b": "zs8b",
        "geval_qwen3_8b": "geval", "geval_qwen7b": "geval",
        "lora_score_only_unbalanced": "unbalanced", "lora_score_only_full": "unbalanced",
    }
    filtered_methods = []
    for m in constant_methods:
        role = role_map.get(m, m)
        if role not in seen_roles:
            filtered_methods.append(m)
            if m in all_results:
                seen_roles.add(role)
    constant_methods = filtered_methods
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


def plot_learning_curves(curves, metadata, output_dir, dpi=300, compare_curves=None):
    """Generate learning curve figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_sizes = set()
    for data in curves.values():
        all_sizes.update(int(k) for k in data.keys())
    for _, data, _, _ in (compare_curves or []):
        all_sizes.update(int(k) for k in data.keys())
    train_sizes = sorted(all_sizes)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

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

    # Plot default LoRA curve on top with fill.
    # If compare_curves is provided, use those model-specific lines instead.
    if "lora_balanced_simple" in curves and not compare_curves:
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

    # Optional extra LoRA curves (e.g., Qwen2.5-7B / Qwen3-8B).
    for label, data, color, marker in (compare_curves or []):
        if not data:
            continue
        sizes = sorted(int(k) for k in data.keys())
        corrs = [data[str(s)] for s in sizes]
        ax.plot(
            sizes, corrs, color=color, marker=marker, linestyle="-",
            linewidth=2.0, markersize=8, alpha=0.95, zorder=11, label=label
        )
        for sz, rho in zip(sizes, corrs):
            ax.annotate(
                f"{rho:.3f}", xy=(sz, rho), xytext=(8, 5),
                textcoords="offset points", fontsize=8, color=color, fontweight="bold"
            )

    ax.set_xlabel("Training Samples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=13, fontweight="bold")
    ax.set_title("Learning Curves: Evaluator Performance vs Training Data Size",
                 fontsize=14, fontweight="bold", pad=12)

    ax.set_xscale("log")
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(s) for s in train_sizes])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(-0.2, 0.7)
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


def plot_learning_curves_main_only(output_dir, dpi=300, compare_curves=None):
    """Generate a cleaner learning curve with only the LoRA subsets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_ALL_RESULTS, "r") as f:
        all_results = json.load(f)
    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    human_avg = [item["avg_score"] for item in eval_data]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

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

    if not compare_curves:
        ax.plot(sizes, corrs, "o-", color="#2563EB", linewidth=2.5, markersize=10, zorder=10,
                label="LoRA-balanced-simple (Ours)")

    # Optional extra model curves overlaid on the old figure.
    for label, data, color, marker in (compare_curves or []):
        if not data:
            continue
        s2 = sorted(int(k) for k in data.keys())
        c2 = [data[str(s)] for s in s2]
        ax.plot(
            s2, c2, color=color, marker=marker, linestyle="-",
            linewidth=2.0, markersize=8, zorder=11, label=label
        )
        for sz, rho in zip(s2, c2):
            ax.annotate(
                f"{rho:.3f}", xy=(sz, rho), xytext=(8, 5),
                textcoords="offset points", fontsize=8, color=color, fontweight="bold"
            )

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
    if not compare_curves:
        for sz, rho in zip(sizes, corrs):
            ax.annotate(f"{rho:.3f}", xy=(sz, rho), xytext=(8, 5),
                         textcoords="offset points", fontsize=8, color="#2563EB", fontweight="bold")

    ax.set_xlabel("Training Samples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spearman $\\rho$ with Human Annotations", fontsize=13, fontweight="bold")
    ax.set_title("LoRA Evaluator Learning Curve", fontsize=14, fontweight="bold", pad=12)

    ax.set_xscale("log")
    tick_sizes = set(sizes)
    for _, data, _, _ in (compare_curves or []):
        tick_sizes.update(int(k) for k in data.keys())
    tick_sizes = sorted(tick_sizes)
    ax.set_xticks(tick_sizes)
    ax.set_xticklabels([str(s) for s in tick_sizes])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(-0.2, 0.7)
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
    parser.add_argument("--tables-dir", type=str, default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--qwen25-results-dir", type=str, default=None)
    parser.add_argument("--qwen3-results-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    compare_curves = []
    if args.qwen25_results_dir:
        curve25 = load_curve_from_results_dir(args.qwen25_results_dir)
        if curve25:
            compare_curves.append(("RewriteJudge (Ours, Qwen2.5-7B)", {str(k): v for k, v in curve25.items()}, "#DC2626", "s"))
            print(f"Loaded compare curve qwen2.5-7B: {sorted(curve25.items())}")
    if args.qwen3_results_dir:
        curve3 = load_curve_from_results_dir(args.qwen3_results_dir)
        if curve3:
            compare_curves.append(("RewriteJudge (Ours, Qwen3-8B)", {str(k): v for k, v in curve3.items()}, "#2563EB", "o"))
            print(f"Loaded compare curve qwen3-8B: {sorted(curve3.items())}")

    curves, metadata = load_real_data(args.all_results, args.metadata, args.eval_data)

    print("Learning curve data (real):")
    for method, data in curves.items():
        display = metadata.get(method, {}).get("display_name", method)
        print(f"  {display}:")
        for sz in sorted(data.keys(), key=int):
            print(f"    n={sz}: rho={data[sz]:+.4f}")

    print("\nGenerating figures...")
    plot_learning_curves(curves, metadata, args.output_dir, args.dpi, compare_curves=compare_curves)
    plot_learning_curves_main_only(args.output_dir, args.dpi, compare_curves=compare_curves)
    print("Done.")


if __name__ == "__main__":
    main()
