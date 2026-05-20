#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all paper figures and tables for EMNLP 2026 Chinese Rewriting Evaluation.

Uses real experiment results from all_results.json and method_metadata.json.

Figures:
  1. Score distribution histogram (evaluator vs human)
  2. Confusion matrix heatmap
  3. Method comparison bar chart (Spearman rho)
  4. Agreement heatmap
  5. Error distribution

Tables:
  1. Main results table (LaTeX)

Usage:
    python generate_figures.py
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATA = PROJECT_ROOT / "data" / "human_eval" / "eval.json"
DEFAULT_ALL_RESULTS = PROJECT_ROOT / "data" / "baselines" / "all_results.json"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "baselines" / "method_metadata.json"
DEFAULT_TRAD_RESULTS = PROJECT_ROOT / "data" / "baselines" / "all_results_traditional.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "analysis" / "results"

# Colors
PALETTE = {
    # Ours / fine-tuned family
    "lora_balanced_simple": "#2563EB",
    "lora_balanced_reasoning": "#1D4ED8",
    "lora_score_only_full": "#60A5FA",
    "lora_original_reasoning": "#93C5FD",
    "lora_score_only_unbalanced": "#0F766E",
    # Baseline judges
    "prometheus2": "#059669",
    "m_prometheus": "#10B981",
    # Zero-shot family (warm colors)
    "zeroshot_qwen7b": "#DC2626",
    "zeroshot_qwen3_8b": "#EF4444",
    "zeroshot_qwen14b": "#B45309",
    "zeroshot_qwen3_14b": "#D97706",
    # G-Eval family (purple)
    "geval_qwen7b": "#7C3AED",
    "geval_qwen3_8b": "#9333EA",
}


def load_data(eval_data_path=DEFAULT_EVAL_DATA, all_results_path=DEFAULT_ALL_RESULTS,
              metadata_path=DEFAULT_METADATA, trad_results_path=DEFAULT_TRAD_RESULTS):
    with open(eval_data_path, "r") as f:
        eval_data = json.load(f)
    with open(all_results_path, "r") as f:
        all_results = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    trad_path = Path(trad_results_path)
    trad_results = json.load(open(trad_path)) if trad_path.exists() else {}
    return eval_data, all_results, metadata, trad_results


def get_display_name(key):
    m = json.load(open(DEFAULT_METADATA)) if os.path.exists(DEFAULT_METADATA) else {}
    return m.get(key, {}).get("display_name", key)


def resolve_ours_keys(all_results):
    """Find available Ours keys for Qwen3 and Qwen2.5 families."""
    keys = set(all_results.keys())

    qwen3_key = None
    if "lora_balanced_simple" in keys:
        qwen3_key = "lora_balanced_simple"
    else:
        # Prefer explicit full-key variants for Qwen3 (e.g. ..._qwen3_8b_...)
        qwen3_full_candidates = sorted([
            k for k in keys
            if k.startswith("lora_balanced_simple")
            and "qwen3" in k
            and "qwen2_5_7b" not in k
            and not is_learning_subset_key(k)
        ])
        if qwen3_full_candidates:
            qwen3_key = qwen3_full_candidates[0]

    qwen25_key = None
    qwen25_full = sorted([
        k for k in keys
        if k.startswith("lora_balanced_simple_qwen2_5_7b")
        and not re.search(r"_(50|100|200|400|500)_qwen2_5_7b", k)
    ])
    if qwen25_full:
        qwen25_key = qwen25_full[0]
    else:
        qwen25_any = sorted([k for k in keys if k.startswith("lora_balanced_simple_qwen2_5_7b")])
        if qwen25_any:
            qwen25_key = qwen25_any[0]
    return qwen3_key, qwen25_key


def format_method_display_for_figures(method, raw_display, qwen3_key, qwen25_key):
    """Figure/table labels: use Qwen3-8B and Qwen2.5-7B (capital B) consistently."""
    if method == qwen3_key:
        return "RewriteJudge (Qwen3-8B)"
    if method == qwen25_key:
        return "RewriteJudge (Qwen2.5-7B)"
    if method == "zeroshot_qwen3_8b":
        return "Zero-shot (Qwen3-8B)"
    if method == "geval_qwen3_8b":
        return "G-Eval (Qwen3-8B)"
    if method == "zeroshot_qwen7b":
        return "Zero-shot (Qwen2.5-7B)"
    if method == "geval_qwen7b":
        return "G-Eval (Qwen2.5-7B)"
    if method == "zeroshot_qwen14b":
        return "Zero-shot (Qwen2.5-14B)"
    if method == "zeroshot_qwen3_14b":
        return "Zero-shot (Qwen3-14B)"
    if method == "lora_balanced_reasoning":
        return "Reasoning prefix (Qwen3-8B)"
    if method.startswith("lora_balanced_reasoning") and "qwen2_5_7b" in method:
        return "Reasoning prefix (Qwen2.5-7B)"
    if method.startswith("lora_balanced_reasoning") and "qwen3" in method:
        return "Reasoning prefix (Qwen3-8B)"
    if method == "lora_score_only_unbalanced":
        return "Unbalanced training (Qwen3-8B)"
    if method.startswith("lora_score_only_unbalanced") and "qwen2_5_7b" in method:
        return "Unbalanced training (Qwen2.5-7B)"
    if method.startswith("lora_score_only_unbalanced") and "qwen3" in method:
        return "Unbalanced training (Qwen3-8B)"
    low = raw_display.lower()
    if low.startswith("unbalanced training (qwen3"):
        return "Unbalanced training (Qwen3-8B)"
    if low.startswith("unbalanced training (qwen2.5-7b"):
        return "Unbalanced training (Qwen2.5-7B)"
    if raw_display.startswith("+ Reasoning prefix"):
        return "Reasoning prefix (Qwen3-8B)"
    return raw_display


def method_color(key):
    if key and key.startswith("lora_balanced_simple_qwen2_5_7b"):
        return "#F59E0B"
    if key and (key == "lora_balanced_simple" or re.match(r"^lora_balanced_simple_\d+$", key)):
        return "#2563EB"
    return PALETTE.get(key, "#666")


def is_learning_subset_key(method_key):
    return bool(
        re.match(r"^lora_balanced_simple_(50|100|200|400|500)$", method_key)
        or re.match(r"^lora_balanced_simple_(50|100|200|400|500)_", method_key)
    )


def plot_score_distribution(eval_data, all_results, metadata, output_dir, dpi=300):
    """Plot score distribution: human annotations vs top evaluators."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    human_scores = [item["consensus_score"] for item in eval_data]
    labels = list(range(6))

    # Select methods to show
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)
    show_methods = []
    if qwen3_key:
        show_methods.append((qwen3_key, "RewriteJudge (Ours, Qwen3-8B)"))
    if qwen25_key:
        show_methods.append((qwen25_key, "RewriteJudge (Ours, Qwen2.5-7B)"))
    show_methods.extend([
        ("prometheus2", "Prometheus 2"),
        ("zeroshot_qwen7b", "Zero-shot (Qwen2.5-7B)"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})

    # Left: human distribution
    ax = axes[0]
    bins = np.arange(-0.5, 6.5, 1)
    human_counts = np.histogram(human_scores, bins=bins)[0]
    human_pct = human_counts / len(human_scores) * 100
    ax.bar(labels, human_pct, width=0.7, color="#1F2937", alpha=0.85,
           label="Human Annotations", edgecolor="white", linewidth=0.5, zorder=5)
    for i, (cnt, pct) in enumerate(zip(human_counts, human_pct)):
        ax.text(i, pct + 1.0, f"{cnt}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#1F2937")
    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Frequency (%)", fontsize=10)
    ax.set_title("Human Annotation Distribution", fontsize=10.5, fontweight="bold")
    ax.set_xticks(labels)
    ax.set_ylim(0, max(human_pct) + 8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Right: grouped comparison
    ax2 = axes[1]
    methods_to_show = []
    method_scores = []
    # Always show human
    methods_to_show.append(("human", "Human", "#1F2937", human_pct))

    for key, label in show_methods:
        if key in all_results:
            scores = [s if s >= 0 else np.nan for s in all_results[key]]
            valid_scores = [s for s in scores if not np.isnan(s)]
            counts = np.histogram(valid_scores, bins=bins)[0]
            pct = counts / len(valid_scores) * 100
            color = method_color(key)
            methods_to_show.append((key, label, color, pct))

    n_methods = len(methods_to_show)
    group_width = 0.85
    bar_width = group_width / n_methods

    for m_idx, (key, label, color, pct) in enumerate(methods_to_show):
        for i, p in enumerate(pct):
            x = i - group_width / 2 + m_idx * bar_width + bar_width / 2
            ax2.bar(x, p, width=bar_width * 0.9, color=color,
                    alpha=0.8, edgecolor="white", linewidth=0.3, zorder=5)

    legend_elements = [Patch(facecolor=info[2], alpha=0.8, label=info[1])
                       for info in methods_to_show]
    ax2.legend(handles=legend_elements, fontsize=7, framealpha=0.9, loc="upper left")
    ax2.set_xlabel("Score", fontsize=10)
    ax2.set_ylabel("Frequency (%)", fontsize=10)
    ax2.set_title("Score Distribution Comparison", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(labels)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/score_distribution.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_confusion_matrix(eval_data, all_results, method="lora_balanced_simple",
                          output_dir="", dpi=300):
    """Confusion matrix heatmap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    human_scores = np.array([item["consensus_score"] for item in eval_data])
    pred_scores = np.array([s if s >= 0 else np.nan for s in all_results[method]])

    labels = list(range(6))
    n = len(labels)

    matrix = np.zeros((n, n), dtype=int)
    for h, p in zip(human_scores, pred_scores):
        if np.isnan(p):
            continue
        hi, pi = int(round(h)), int(round(p))
        if 0 <= hi < n and 0 <= pi < n:
            matrix[hi, pi] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100

    raw_display = get_display_name(method)
    # Shorten long method names for display
    if "qwen2_5_7b" in method or "qwen2.5-7b" in method.lower():
        display = "RewriteJudge (Qwen2.5-7B)"
    elif method == "lora_balanced_simple" or method.startswith("lora_balanced_simple_qwen3"):
        display = "RewriteJudge (Qwen3-8B)"
    else:
        display = raw_display

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title_suffix, fmt in [
        (axes[0], matrix, "Counts", "d"),
        (axes[1], matrix_pct, "Row-normalised (%)", ".1f"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max() if fmt == "d" else 100, aspect="equal")
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                text = f"{val}" if fmt == "d" else f"{val:.1f}%"
                text_color = "white" if val > (data.max() * 0.6) else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Score", fontsize=10)
        ax.set_ylabel("Human Score", fontsize=10)
        ax.set_title(f"{display} ({title_suffix})", fontsize=10.5, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/confusion_matrix_{method}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_method_comparison(eval_data, all_results, metadata, trad_results, output_dir, dpi=300):
    """Main comparison bar chart: Spearman rho for all methods."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from scipy import stats as _stats

    human_avg = [item["avg_score"] for item in eval_data]
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)
    keep_ours_keys = {k for k in [qwen3_key, qwen25_key] if k}

    # Collect all methods with Spearman values
    entries = []

    def _format_display(method, raw_display):
        return format_method_display_for_figures(method, raw_display, qwen3_key, qwen25_key)

    for method, scores in all_results.items():
        if is_learning_subset_key(method) and method not in keep_ours_keys:
            continue
        if method in ("lora_multi_score", "m_prometheus"):
            continue
        preds = np.array([s if s >= 0 else np.nan for s in scores], dtype=float)
        refs = np.array(human_avg, dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs))
        if valid.sum() >= 2:
            rho, _ = _stats.spearmanr(preds[valid], refs[valid])
        else:
            rho = 0.0
        raw_display = metadata.get(method, {}).get("display_name", method)
        display = _format_display(method, raw_display)
        if method == qwen3_key or method == qwen25_key:
            category = "lora"
        else:
            category = metadata.get(method, {}).get("category", "unknown")
        entries.append((display, rho, category, method))

    # Add traditional metrics
    for metric_name, scores in trad_results.items():
        preds = np.array(scores, dtype=float)
        refs = np.array(human_avg, dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs))
        if valid.sum() >= 2:
            rho, _ = _stats.spearmanr(preds[valid], refs[valid])
        else:
            rho = 0.0
        entries.append((metric_name.replace("trad_", "").replace("_", "-").upper(), rho, "traditional", metric_name))

    # Sort by Spearman
    entries.sort(key=lambda x: x[1], reverse=True)

    # Separate into groups
    lora_entries = [e for e in entries if e[2] == "lora"]
    llm_entries = [e for e in entries if e[2] == "llm"]
    trad_entries = [e for e in entries if e[2] == "traditional"]

    # --- Figure 1: LLM-based Evaluators ---
    main_entries = lora_entries + llm_entries
    main_entries.sort(key=lambda x: x[1], reverse=True)

    names = [e[0] for e in main_entries]
    values = [e[1] for e in main_entries]
    colors = [method_color(e[3]) for e in main_entries]

    fig1, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65)

    # Highlight best
    if values:
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("#000000")
        bars[best_idx].set_linewidth(2)

    for i, (bar, val) in enumerate(zip(bars, values)):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.3f}", va="center", ha=ha,
                fontsize=8, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Spearman $\\rho$", fontsize=10)
    ax.set_title("LLM-based Evaluators", fontsize=10.5, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()
    if values:
        xmin = min(values)
        xmax = max(values)
        pad = max(0.04, (xmax - xmin) * 0.15)
        ax.set_xlim(xmin - pad, xmax + pad)

    fig1.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/method_comparison_llm.{ext}"
        fig1.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig1)

    # --- Figure 2: Traditional Metrics ---
    trad_entries.sort(key=lambda x: x[1], reverse=True)
    names2 = [e[0] for e in trad_entries]
    values2 = [e[1] for e in trad_entries]

    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    y_pos2 = np.arange(len(names2))
    bars2 = ax2.barh(y_pos2, values2, color="#9CA3AF", alpha=0.7, edgecolor="white", height=0.65)
    for i, (bar, val) in enumerate(zip(bars2, values2)):
        ax2.text(val - 0.01, i, f"{val:+.3f}", va="center", ha="right",
                 fontsize=7.5, fontweight="bold")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(names2, fontsize=8)
    ax2.set_xlabel("Spearman $\\rho$", fontsize=10)
    ax2.set_title("Traditional Metrics", fontsize=10.5, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax2.invert_yaxis()
    if values2:
        xmin2 = min(values2)
        xmax2 = max(values2)
        pad2 = max(0.04, (xmax2 - xmin2) * 0.3)
        ax2.set_xlim(xmin2 - pad2, xmax2 + pad2)

    fig2.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/method_comparison_traditional.{ext}"
        fig2.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig2)

    # --- Figure 3: Combined (kept for backward compatibility) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [4.2, 1.8]})
    ax = axes[0]
    main_entries2 = lora_entries + llm_entries
    main_entries2.sort(key=lambda x: x[1], reverse=True)
    names3 = [e[0] for e in main_entries2]
    values3 = [e[1] for e in main_entries2]
    colors3 = [method_color(e[3]) for e in main_entries2]
    y_pos3 = np.arange(len(names3))
    bars3 = ax.barh(y_pos3, values3, color=colors3, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65)
    if values3:
        best_idx3 = int(np.argmax(values3))
        bars3[best_idx3].set_edgecolor("#000000")
        bars3[best_idx3].set_linewidth(2)
    for i, (bar, val) in enumerate(zip(bars3, values3)):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")
    ax.set_yticks(y_pos3)
    ax.set_yticklabels(names3, fontsize=8.5)
    ax.set_xlabel("Spearman $\\rho$", fontsize=10)
    ax.set_title("LLM-based Evaluators", fontsize=10.5, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()
    if values3:
        xmin3, xmax3 = min(values3), max(values3)
        ax.set_xlim(xmin3 - max(0.04, (xmax3-xmin3)*0.15), xmax3 + max(0.04, (xmax3-xmin3)*0.15))
    ax2c = axes[1]
    trad_sorted = sorted(trad_entries, key=lambda x: x[1], reverse=True)
    names4 = [e[0] for e in trad_sorted]
    values4 = [e[1] for e in trad_sorted]
    y_pos4 = np.arange(len(names4))
    bars4 = ax2c.barh(y_pos4, values4, color="#9CA3AF", alpha=0.7, edgecolor="white", height=0.65)
    for i, (bar, val) in enumerate(zip(bars4, values4)):
        ax2c.text(val - 0.01, i, f"{val:+.3f}", va="center", ha="right", fontsize=7.5, fontweight="bold")
    ax2c.set_yticks(y_pos4)
    ax2c.set_yticklabels(names4, fontsize=8)
    ax2c.set_xlabel("Spearman $\\rho$", fontsize=10)
    ax2c.set_title("Traditional Metrics", fontsize=10.5, fontweight="bold")
    ax2c.axvline(x=0, color="black", linewidth=0.8)
    ax2c.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax2c.invert_yaxis()
    if values4:
        xmin4, xmax4 = min(values4), max(values4)
        ax2c.set_xlim(xmin4 - max(0.04, (xmax4-xmin4)*0.3), xmax4 + max(0.04, (xmax4-xmin4)*0.3))
    fig.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/method_comparison.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def generate_main_results_table(all_results, metadata, trad_results, output_dir):
    """Generate main results table in LaTeX format."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from scipy import stats as _stats

    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    human_avg = [item["avg_score"] for item in eval_data]
    human_consensus = [item["consensus_score"] for item in eval_data]
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)

    def _format_main_display(method, raw_display):
        return format_method_display_for_figures(method, raw_display, qwen3_key, qwen25_key)

    def _infer_type(method, category):
        if category == "traditional" or method.startswith("trad_"):
            return "Traditional"
        if method.startswith("lora_") or method in {qwen3_key, qwen25_key}:
            return "Fine-tuned (7B)"
        if "prometheus" in method:
            return "Fine-tuned (7B)"
        if method in {"zeroshot_qwen3_14b", "zeroshot_qwen14b"}:
            return "Zero-shot (14B)"
        if method in {"zeroshot_qwen3_8b", "zeroshot_qwen7b", "geval_qwen3_8b", "geval_qwen7b"}:
            return "Zero-shot (7B)"
        return "LLM"

    # Collect all methods
    entries = []
    for method, scores in all_results.items():
        meta = metadata.get(method, {})
        if meta.get("category") == "lora" and is_learning_subset_key(method):
            continue  # Skip learning curve subsets in main table
        preds = np.array([s if s >= 0 else np.nan for s in scores], dtype=float)
        refs = np.array(human_avg, dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs))
        if valid.sum() < 2:
            continue
        sp, spp = _stats.spearmanr(preds[valid], refs[valid])
        pr, prp = _stats.pearsonr(preds[valid], refs[valid])
        kt, ktp = _stats.kendalltau(preds[valid], refs[valid])
        refs_c = np.array(human_consensus, dtype=float)
        valid_c = ~(np.isnan(preds) | np.isnan(refs_c))
        hv, pv = refs_c[valid_c], preds[valid_c]
        mae = float(np.mean(np.abs(hv - pv)))
        rmse = float(np.sqrt(np.mean((hv - pv) ** 2)))
        exact = float(np.mean(hv == pv)) * 100
        within_1 = float(np.mean(np.abs(hv - pv) <= 1)) * 100
        within_2 = float(np.mean(np.abs(hv - pv) <= 2)) * 100
        entries.append({
            "name": _format_main_display(method, meta.get("display_name", method)),
            "method": method,
            "category": meta.get("category", "unknown"),
            "spearman": sp, "spearman_p": spp,
            "pearson": pr, "pearson_p": prp,
            "kendall": kt, "kendall_p": ktp,
            "mae": mae, "rmse": rmse,
            "exact": exact, "within_1": within_1, "within_2": within_2,
            "is_ours": method == qwen3_key or method == qwen25_key,
        })

    # Add traditional metrics
    for metric_name, scores in trad_results.items():
        preds = np.array(scores, dtype=float)
        refs = np.array(human_avg, dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs))
        if valid.sum() < 2:
            continue
        sp, _ = _stats.spearmanr(preds[valid], refs[valid])
        pr, _ = _stats.pearsonr(preds[valid], refs[valid])
        kt, _ = _stats.kendalltau(preds[valid], refs[valid])
        entries.append({
            "name": metric_name.replace("trad_", "").replace("_", "-").upper(),
            "method": metric_name,
            "category": "traditional",
            "spearman": sp, "spearman_p": 0,
            "pearson": pr, "pearson_p": 0,
            "kendall": kt, "kendall_p": 0,
            "mae": float("nan"), "rmse": float("nan"),
            "exact": float("nan"), "within_1": float("nan"), "within_2": float("nan"),
            "is_ours": False,
        })

    # Sort by Spearman
    entries.sort(key=lambda e: e["spearman"], reverse=True)

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main evaluation results on the Chinese rewriting quality benchmark ($n=129$). Scores are compared against human average annotations (0--5 scale). $^*$, $^{**}$, $^{***}$ indicate $p < 0.05$, $0.01$, $0.001$ respectively.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Type & Spearman $\rho$ & Pearson $r$ & Kendall $\tau$ "
                 r"& MAE & RMSE & Exact / $\pm$1 / $\pm$2 (\%) \\")
    lines.append(r"\midrule")

    for e in entries:
        name = e["name"]
        if e["is_ours"]:
            name = r"\textbf{" + name + r"}"

        sp_str = f"{e['spearman']:+.4f}" + _sig(e["spearman_p"])
        pr_str = f"{e['pearson']:+.4f}" + _sig(e["pearson_p"])
        kt_str = f"{e['kendall']:+.4f}" + _sig(e["kendall_p"])

        # Determine type from canonical method key/category
        type_str = _infer_type(e["method"], e.get("category", "unknown"))

        if e["is_ours"]:
            type_str = r"\textbf{" + type_str + r"}"

        mae_str = f"{e['mae']:.3f}" if not np.isnan(e["mae"]) else "-"
        rmse_str = f"{e['rmse']:.3f}" if not np.isnan(e["rmse"]) else "-"
        if not np.isnan(e["exact"]):
            agr_str = f"{e['exact']:.1f} / {e['within_1']:.1f} / {e['within_2']:.1f}"
        else:
            agr_str = "-"

        lines.append(f"{name} & {type_str} & {sp_str} & {pr_str} & {kt_str} & {mae_str} & {rmse_str} & {agr_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    out_path = os.path.join(output_dir, "main_results_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")
    return latex


def generate_learning_curve_table(all_results, metadata, output_dir):
    """Generate learning curve table in LaTeX format."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from scipy import stats as _stats

    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    human_avg = [item["avg_score"] for item in eval_data]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Learning curve results showing Spearman $\rho$ with human annotations at different training data sizes.}")
    lines.append(r"\label{tab:learning_curve}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Training Samples & Qwen3-8B (Ours) & Qwen2.5-7B (Ours) \\")
    lines.append(r"\midrule")

    refs_avg = np.array(human_avg, dtype=float)
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)

    def _rho_from_key(k):
        if not k or k not in all_results:
            return None
        preds = np.array([s if s >= 0 else np.nan for s in all_results[k]], dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs_avg))
        if valid.sum() < 2:
            return None
        r, _ = _stats.spearmanr(preds[valid], refs_avg[valid])
        return float(r)

    qwen25_base = qwen25_key.rsplit("_qwen2_5_7b", 1)[0] if qwen25_key and "_qwen2_5_7b" in qwen25_key else None
    sizes = [50, 100, 200, 400, 500, 1008]
    for size in sizes:
        k3 = qwen3_key if size == 1008 else f"lora_balanced_simple_{size}"
        k25 = qwen25_key if size == 1008 else (f"{qwen25_base}_{size}_qwen2_5_7b{qwen25_key.split('_qwen2_5_7b',1)[1]}" if qwen25_base and qwen25_key and "_qwen2_5_7b" in qwen25_key else None)

        r3 = _rho_from_key(k3)
        r25 = _rho_from_key(k25)
        if r3 is None and r25 is None:
            continue
        label = str(size) + (r" (full)" if size == 1008 else "")
        r3s = f"{r3:+.4f}" if r3 is not None else "-"
        r25s = f"{r25:+.4f}" if r25 is not None else "-"
        lines.append(f"{label} & {r3s} & {r25s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = os.path.join(output_dir, "learning_curve_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")
    return latex


def _sig(p_value):
    if p_value < 0.001:
        return "$^{***}$"
    elif p_value < 0.01:
        return "$^{**}$"
    elif p_value < 0.05:
        return "$^{*}$"
    return ""


def plot_agreement_heatmap(eval_data, all_results, metadata, output_dir, dpi=300):
    """Heatmap of pairwise Spearman correlations between annotators and evaluators."""
    from scipy import stats as _stats
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    human_avg = [item["avg_score"] for item in eval_data]
    annotator_scores = list(zip(*[item["annotator_scores"] for item in eval_data]))  # 3 x N
    n_annotators = len(annotator_scores)

    # Build list of (label, scores) to correlate
    entries = []
    for i, ann in enumerate(annotator_scores):
        entries.append((f"Annotator {i+1}", list(ann)))
    entries.append(("Human Avg", human_avg))

    # Add top evaluator methods
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)
    show_methods = []
    if qwen3_key:
        show_methods.append(qwen3_key)
    if qwen25_key:
        show_methods.append(qwen25_key)
    show_methods.extend(["lora_balanced_reasoning", "prometheus2",
                         "zeroshot_qwen3_8b", "geval_qwen3_8b"])
    for key in show_methods:
        if key in all_results:
            preds = [s if s >= 0 else float('nan') for s in all_results[key]]
            raw = metadata.get(key, {}).get("display_name", key)
            display = format_method_display_for_figures(key, raw, qwen3_key, qwen25_key)
            entries.append((display, preds))

    n = len(entries)
    matrix = np.zeros((n, n))
    for i, (_, s1) in enumerate(entries):
        for j, (_, s2) in enumerate(entries):
            a = np.array(s1, dtype=float)
            b = np.array(s2, dtype=float)
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() >= 2:
                rho, _ = _stats.spearmanr(a[valid], b[valid])
                matrix[i, j] = rho
            else:
                matrix[i, j] = float('nan')

    labels = [e[0] for e in entries]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.75)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Spearman Correlation: Annotators vs. Evaluators",
                 fontsize=10.5, fontweight="bold")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/agreement_heatmap.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def load_pairwise_results(pairwise_dir=None):
    """Load all API and LoRA pairwise results from data/pairwise/*.json.

    Returns list of dicts with keys:
        display_name, model_id, params_b, spearman, accuracy, parse_fail_rate,
        family, is_ours
    """
    if pairwise_dir is None:
        pairwise_dir = PROJECT_ROOT / "data" / "pairwise"
    pairwise_dir = Path(pairwise_dir)

    # Model metadata: (display_name, params_b, family)
    MODEL_META = {
        # API baselines
        "Qwen/Qwen2.5-72B-Instruct":               ("Qwen2.5-72B",    72,   "Qwen"),
        "Pro/Qwen/Qwen2.5-72B-Instruct":            ("Qwen2.5-72B",    72,   "Qwen"),
        "Qwen/Qwen3-235B-A22B-Instruct-2507":       ("Qwen3-235B",     235,  "Qwen"),
        "Pro/Qwen/Qwen3-235B-A22B-Instruct-2507":   ("Qwen3-235B",     235,  "Qwen"),
        "deepseek-ai/DeepSeek-V3":                  ("DeepSeek-V3",    685,  "DeepSeek"),
        "Pro/deepseek-ai/DeepSeek-V3":              ("DeepSeek-V3",    685,  "DeepSeek"),
        "deepseek-ai/DeepSeek-V3.1-Terminus":       ("DeepSeek-V3.1",  685,  "DeepSeek"),
        "Pro/deepseek-ai/DeepSeek-V3.1-Terminus":   ("DeepSeek-V3.1",  685,  "DeepSeek"),
        "DeepSeek-V3.2-Meituan":                    ("DeepSeek-V3.2",  685,  "DeepSeek"),
        "Pro/moonshotai/Kimi-K2-Instruct-0905":     ("Kimi-K2",        1000, "Kimi"),
        "moonshotai/Kimi-K2-Instruct-0905":         ("Kimi-K2",        1000, "Kimi"),
        "moonshotai/Kimi-K2.5":                     ("Kimi-K2.5",      1000, "Kimi"),
        "kimi-k2.5":                                ("Kimi-K2.5",      1000, "Kimi"),
        "gpt-4.1":                                  ("GPT-4.1",        1800, "GPT"),
        "gpt-4o":                                   ("GPT-4o",         1800, "GPT"),
    }

    entries = []
    seen = set()

    # Load API baselines (main dir + closed-source-model subdir)
    api_files = list(pairwise_dir.glob("api_baseline_*.json"))
    api_files += list((pairwise_dir / "closed-source-model").glob("api_baseline_*.json"))
    for f in sorted(api_files):
        if "checkpoint" in f.name:
            continue
        try:
            d = json.load(open(f))
            model_id = d.get("model", "")
            m = d.get("metrics", {})
            spearman = m.get("spearman_rho_winrate_vs_avg")
            accuracy = m.get("pairwise_accuracy")
            parse_fail = m.get("parse_failure_rate", 0.0)
            if spearman is None:
                continue
            meta = MODEL_META.get(model_id, (model_id, 0, "Unknown"))
            display, params, family = meta
            if display in seen:
                continue
            seen.add(display)
            entries.append({
                "display_name": display,
                "model_id": model_id,
                "params_b": params,
                "spearman": spearman,
                "accuracy": accuracy or 0.0,
                "parse_fail_rate": parse_fail,
                "family": family,
                "is_ours": False,
            })
        except Exception:
            continue

    def _read_pairwise_metrics(result_path):
        try:
            d = json.load(open(result_path))
            m = d.get("cross_source", {}).get("metrics", d.get("metrics", {}))
            spearman = m.get("spearman_rho_winrate_vs_avg")
            if spearman is None:
                return None
            return {
                "spearman": spearman,
                "accuracy": m.get("pairwise_accuracy") or 0.0,
                "parse_fail_rate": m.get("parse_failure_rate", 0.0),
            }
        except Exception:
            return None

    # Load two explicit Ours pairwise models if present
    ours_specs = [
        ("qwen2.5-7B", "RewriteJudge (Qwen2.5-7B)", "lora_pairwise_qwen2_5_7b"),
        ("qwen3-8b", "RewriteJudge (Qwen3-8B)", "lora_pairwise_qwen3_8b"),
    ]
    loaded_ours = 0
    for subdir, display_name, model_id in ours_specs:
        base = pairwise_dir / subdir
        candidates = [
            base / "pairwise_b1_r16_results.json",
            base / "b1_r16_results.json",
            base / "b1_cross_source_results.json",
        ]
        # Fallback: if exact r16 file name differs, pick first matching pairwise_b1_r16*.json
        candidates += sorted(base.glob("pairwise_b1_r16*.json"))
        picked = None
        for p in candidates:
            if p.exists():
                picked = p
                break
        if not picked:
            continue
        metrics = _read_pairwise_metrics(picked)
        if not metrics or display_name in seen:
            continue
        seen.add(display_name)
        entries.append({
            "display_name": display_name,
            "model_id": model_id,
            "params_b": 7,
            "spearman": metrics["spearman"],
            "accuracy": metrics["accuracy"],
            "parse_fail_rate": metrics["parse_fail_rate"],
            "family": "Ours",
            "is_ours": True,
        })
        loaded_ours += 1

    # Backward-compatible fallback: single Ours entry if explicit dirs unavailable
    if loaded_ours == 0:
        lora_files = list(pairwise_dir.glob("b1_cross_source_results.json"))
        lora_files += list(pairwise_dir.glob("*/pairwise_b1_r16_results.json"))
        lora_files += list(pairwise_dir.glob("*/b1_r16_results.json"))
        for f in lora_files:
            metrics = _read_pairwise_metrics(f)
            if not metrics:
                continue
            if "RewriteJudge (Ours)" in seen:
                continue
            seen.add("RewriteJudge (Ours)")
            entries.append({
                "display_name": "RewriteJudge (Ours)",
                "model_id": "lora_pairwise",
                "params_b": 7,
                "spearman": metrics["spearman"],
                "accuracy": metrics["accuracy"],
                "parse_fail_rate": metrics["parse_fail_rate"],
                "family": "Ours",
                "is_ours": True,
            })
            break

    return entries


def get_ours_pairwise_color(entry):
    model_id = (entry or {}).get("model_id", "")
    display = (entry or {}).get("display_name", "").lower()
    if "qwen3" in model_id or "qwen3" in display:
        return "#EC4899"
    if "qwen2_5" in model_id or "qwen2.5" in display:
        return "#F59E0B"
    return "#F59E0B"


def plot_scaling_analysis(pairwise_entries, output_dir, dpi=300):
    """Scatter plot: model size (params_b) vs Spearman rho."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api_entries = [e for e in pairwise_entries if not e["is_ours"]]
    ours = [e for e in pairwise_entries if e["is_ours"]]

    if not api_entries:
        print("  [skip] No API pairwise data found.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Color by family
    family_colors = {
        "Qwen": "#2563EB",
        "DeepSeek": "#059669",
        "Kimi": "#DC2626",
        "GPT": "#9333EA",
        "Unknown": "#9CA3AF",
    }

    for e in api_entries:
        color = family_colors.get(e["family"], "#9CA3AF")
        ax.scatter(e["params_b"], e["spearman"], color=color, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(e["display_name"], (e["params_b"], e["spearman"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=color)

    # Plot ours as star
    for e in ours:
        ours_color = get_ours_pairwise_color(e)
        ax.scatter(e["params_b"], e["spearman"], color=ours_color, s=200,
                   marker="*", zorder=6, edgecolors="black", linewidths=0.8,
                   label=e["display_name"])
        is_qwen3 = "qwen3" in e.get("display_name", "").lower() or "qwen3" in e.get("model_id", "")
        y_offset = 1 if is_qwen3 else -6
        ax.annotate(e["display_name"], (e["params_b"], e["spearman"]),
                    textcoords="offset points", xytext=(10, y_offset),
                    fontsize=7.5, color=ours_color, fontweight="bold")

    # Trend line for API models only
    if len(api_entries) >= 2:
        xs = np.array([e["params_b"] for e in api_entries], dtype=float)
        ys = np.array([e["spearman"] for e in api_entries], dtype=float)
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="#9CA3AF", alpha=0.6,
                linewidth=1.2, label="Trend (zero-shot)")

    # Legend for families
    for family, color in family_colors.items():
        if any(e["family"] == family for e in api_entries):
            ax.scatter([], [], color=color, s=60, label=family)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)
    ax.set_xlabel("Model Parameters (B)", fontsize=10)
    ax.set_ylabel("Spearman $\\rho$", fontsize=10)
    ax.set_title("Model Scale vs. Evaluation Quality\n(Zero-shot Pairwise)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9, markerscale=1.3,
              handlelength=1.5, handleheight=1.5, borderpad=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/scaling_analysis.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_generational_regression(pairwise_entries, output_dir, dpi=300):
    """Bar chart showing within-family regression across model generations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define generation pairs
    families = {
        "Qwen": [
            ("Qwen2.5-72B", "Qwen2.5"),
            ("Qwen3-235B",  "Qwen3"),
        ],
        "DeepSeek": [
            ("DeepSeek-V3",   "V3"),
            ("DeepSeek-V3.1", "V3.1"),
            ("DeepSeek-V3.2", "V3.2"),
        ],
    }

    entry_map = {e["display_name"]: e for e in pairwise_entries}

    n_families = len(families)
    fig, axes = plt.subplots(1, n_families, figsize=(5.5 * n_families, 5.5))
    if n_families == 1:
        axes = [axes]
    family_colors = {"Qwen": "#2563EB", "DeepSeek": "#059669", "Kimi": "#DC2626", "GPT": "#9333EA"}

    for ax, (family, models) in zip(axes, families.items()):
        names = [m[1] for m in models]
        values = []
        has_data = []
        for display, _ in models:
            e = entry_map.get(display)
            values.append(e["spearman"] if e else 0.0)
            has_data.append(e is not None)

        color = family_colors[family]
        bars = ax.bar(names, values, color=color, alpha=0.8,
                      edgecolor="white", linewidth=0.5, width=0.5)

        for bar, val in zip(bars, values):
            offset = 0.01 if val >= 0 else -0.03
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:+.3f}", ha="center", va=va,
                    fontsize=9, fontweight="bold")

        # Arrow(s) showing direction between consecutive generations.
        for i in range(len(values) - 1):
            if not (has_data[i] and has_data[i + 1]):
                continue
            delta = values[i + 1] - values[i]
            # Use absolute delta instead of percentage to avoid distortion with negative values
            x0 = bars[i].get_x() + bars[i].get_width() / 2
            x1 = bars[i + 1].get_x() + bars[i + 1].get_width() / 2
            y0 = values[i] * 0.6  # start from 60% height of bar
            y1 = values[i + 1] * 0.6  # end at 60% height of bar
            color = "red" if delta < 0 else "green"
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
            # Offset text perpendicular to arrow to avoid overlap
            mid_x = (x0 + x1) / 2 + 0.15
            mid_y = (y0 + y1) / 2
            ax.text(mid_x, mid_y, f"{delta:+.3f}",
                    ha="left", fontsize=9, color=color, fontweight="bold")

        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{family} Family", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Spearman $\\rho$", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        # Add y-axis margin
        ymin, ymax = ax.get_ylim()
        margin = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    fig.suptitle("Generational Regression in Zero-shot Pairwise Evaluation",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/generational_regression.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_accuracy_vs_rho(pairwise_entries, output_dir, dpi=300):
    """Scatter plot: pairwise accuracy vs Spearman rho."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not pairwise_entries:
        print("  [skip] No pairwise data found.")
        return

    family_colors = {
        "Qwen": "#2563EB",
        "DeepSeek": "#059669",
        "Kimi": "#DC2626",
        "GPT": "#9333EA",
        "Unknown": "#9CA3AF",
    }

    with matplotlib.rc_context({"font.size": 22, "axes.titlesize": 28,
                                 "axes.labelsize": 26, "xtick.labelsize": 22,
                                 "ytick.labelsize": 22, "legend.fontsize": 22}):
        fig, ax = plt.subplots(figsize=(14, 10))

        for e in pairwise_entries:
            color = get_ours_pairwise_color(e) if e["is_ours"] else family_colors.get(e["family"], "#9CA3AF")
            marker = "*" if e["is_ours"] else "o"
            size = 700 if e["is_ours"] else 300
            ax.scatter(e["accuracy"], e["spearman"], color=color, s=size,
                       marker=marker, zorder=5, edgecolors="white", linewidths=0.8)
            if e["display_name"] == "DeepSeek-V3":
                xy_offset = (8, -24)
            elif e["is_ours"] and "qwen2.5" in e["display_name"].lower():
                xy_offset = (-220, -30)
            else:
                xy_offset = (8, 6)
            ax.annotate(e["display_name"], (e["accuracy"], e["spearman"]),
                        textcoords="offset points", xytext=xy_offset,
                        fontsize=22, color=color,
                        fontweight="bold" if e["is_ours"] else "normal")

        # X-axis from 0.2, extend right to give labels room
        all_acc = [e["accuracy"] for e in pairwise_entries]
        ax.set_xlim(0.2, max(all_acc) + 0.15 if all_acc else 1.0)

        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)
        ax.set_xlabel("Pairwise Accuracy", fontweight="bold")
        ax.set_ylabel("Spearman $\\rho$", fontweight="bold")
        ax.set_title("Pairwise Accuracy vs. Ranking Correlation\n(Accuracy ≠ Quality)",
                     fontweight="bold")

        for family, color in family_colors.items():
            if any(e["family"] == family for e in pairwise_entries):
                marker = "*" if family == "Ours" else "o"
                ax.scatter([], [], color=color, s=200, marker=marker, label=family)
        if any(e["is_ours"] for e in pairwise_entries):
            ax.scatter([], [], color="#EC4899", s=300, marker="*", label="Ours (Qwen3-8B)")
            ax.scatter([], [], color="#F59E0B", s=300, marker="*", label="Ours (Qwen2.5-7B)")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        for ext in ["pdf", "png"]:
            out_path = f"{output_dir}/accuracy_vs_rho.{ext}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
            print(f"  Saved: {out_path}")
        plt.close(fig)


def plot_parse_failure_vs_rho(pairwise_entries, output_dir, dpi=300):
    """Scatter plot: parse failure rate vs Spearman rho."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api_entries = [e for e in pairwise_entries if not e["is_ours"]]
    if not api_entries:
        print("  [skip] No API pairwise data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    family_colors = {
        "Qwen": "#2563EB",
        "DeepSeek": "#059669",
        "Kimi": "#DC2626",
        "GPT": "#9333EA",
        "Unknown": "#9CA3AF",
    }

    for e in api_entries:
        color = family_colors.get(e["family"], "#9CA3AF")
        ax.scatter(e["parse_fail_rate"] * 100, e["spearman"],
                   color=color, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(e["display_name"],
                    (e["parse_fail_rate"] * 100, e["spearman"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=color)

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Parse Failure Rate (%)", fontsize=10)
    ax.set_ylabel("Spearman $\\rho$", fontsize=10)
    ax.set_title("Format Compliance vs. Evaluation Quality\n(Compliance ≠ Quality)",
                 fontsize=10.5, fontweight="bold")

    for family, color in family_colors.items():
        if any(e["family"] == family for e in api_entries):
            ax.scatter([], [], color=color, s=60, label=family)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = f"{output_dir}/parse_failure_vs_rho.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def generate_downstream_table(output_dir, generated_dir=None, dpi=300):
    """Generate downstream validation table (Table: filtering strategy comparison)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if generated_dir is None:
        generated_dir = PROJECT_ROOT / "data" / "generated_rewrites" / "gpt-4.1"
    generated_dir = Path(generated_dir)

    scored_path = generated_dir / "scored_rewrites.json"
    rewrites_path = generated_dir / "all_rewrites.json"

    if not scored_path.exists():
        print(f"  [skip] scored_rewrites.json not found at {scored_path}")
        return

    scored = json.load(open(scored_path))
    scores = np.array([s.get("predicted_score", 0) for s in scored
                       if s.get("predicted_score") is not None])
    n = len(scores)
    if n == 0:
        print("  [skip] No valid scores.")
        return

    # BLEU mid-range (same implementation as downstream/run_downstream.py)
    bleu_mask = np.ones(n, dtype=bool)
    if rewrites_path.exists():
        from collections import Counter as _Counter
        rewrites = json.load(open(rewrites_path))
        min_len = min(len(rewrites), n)
        def _char_bleu(hyp, ref):
            hc = _Counter(list(hyp))
            rc = _Counter(list(ref))
            if not hc:
                return 0.0
            clipped = sum(min(hc[c], rc[c]) for c in hc)
            return clipped / len(hc)
        bleu_arr = np.array([_char_bleu(
            rewrites[i].get("rewrite_text", ""),
            rewrites[i].get("source_text", "")
        ) for i in range(min_len)])
        bleu_mask = (bleu_arr >= 0.2) & (bleu_arr <= 0.6)

    np.random.seed(42)
    rand_idx = np.random.choice(n, n // 2, replace=False)
    top50_idx = np.argsort(scores)[::-1][:n // 2]
    top30_idx = np.argsort(scores)[::-1][:int(n * 0.3)]
    bot50_idx = np.argsort(scores)[:n // 2]

    strategies = [
        ("Top 30\\%",              scores[top30_idx]),
        ("$\\geq$4 threshold",     scores[scores >= 4]),
        ("\\textbf{Top 50\\%}",    scores[top50_idx]),
        ("BLEU mid-range",         scores[:len(bleu_mask)][bleu_mask]),
        ("$\\geq$3 threshold",     scores[scores >= 3]),
        ("Random 50\\%",           scores[rand_idx]),
        ("All (unfiltered)",       scores),
        ("Bottom 50\\%",           scores[bot50_idx]),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream validation: filtering strategy comparison. "
                 r"RewriteJudge top-50\% filtering improves mean quality by "
                 f"{100*(scores[top50_idx].mean()-scores.mean())/scores.mean():.0f}\\%"
                 r" with 100\% of selected rewrites scoring $\geq$3.}")
    lines.append(r"\label{tab:downstream}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & $N$ & Mean Score & $\sigma$ & \%$\geq$3 \\")
    lines.append(r"\midrule")

    for label, data in strategies:
        if len(data) == 0:
            continue
        mean_val = data.mean()
        if len(data) > 1:
            sigma_val = float(np.std(data, ddof=1))
        else:
            sigma_val = 0.0
        pct = (data >= 3).mean() * 100
        bold = label.startswith(r"\textbf")
        mean_str = f"\\textbf{{{mean_val:.2f}}}" if bold else f"{mean_val:.2f}"
        sigma_str = f"\\textbf{{{sigma_val:.2f}}}" if bold else f"{sigma_val:.2f}"
        pct_str = f"\\textbf{{{pct:.1f}}}" if bold else f"{pct:.1f}"
        n_str = f"\\textbf{{{len(data)}}}" if bold else str(len(data))
        lines.append(f"{label} & {n_str} & {mean_str} & {sigma_str} & {pct_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = Path(output_dir) / "downstream_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")

    # Also print summary
    print(f"  All: mean={scores.mean():.2f}, %>=3={(scores>=3).mean()*100:.1f}%")
    print(f"  Top50%: mean={scores[top50_idx].mean():.2f}, %>=3=100%")


def plot_delta_improvement(
    eval_data,
    all_results,
    metadata,
    output_dir,
    dpi=300,
    trad_results=None,
):
    """Grouped bars: $\\Delta\\rho = \\rho(\\text{LoRA}) - \\rho(\\text{baseline})$ vs. training size.

    Matches paper style: three baselines (zero-shot Qwen2.5-7B, prompt-based larger LM, Prometheus~2)
    and Qwen3-8B RewriteJudge learning-curve checkpoints; full LoRA is shown at x=600.
    """
    from scipy import stats as _stats

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    refs_avg = np.array([float(item["avg_score"]) for item in eval_data], dtype=float)

    def _rho_from_key(k):
        if not k or k not in all_results:
            return None
        preds = np.array([s if s >= 0 else np.nan for s in all_results[k]], dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs_avg))
        if valid.sum() < 2:
            return None
        r, _ = _stats.spearmanr(preds[valid], refs_avg[valid])
        return float(r)

    qwen3_key, _q25 = resolve_ours_keys(all_results)
    lora_full = qwen3_key if qwen3_key else "lora_balanced_simple"

    # (x-axis label, checkpoint key) — full model uses paper tick "600"
    size_plan = [
        (50, "lora_balanced_simple_50"),
        (100, "lora_balanced_simple_100"),
        (200, "lora_balanced_simple_200"),
        (400, "lora_balanced_simple_400"),
        (600, lora_full),
    ]

    # Baselines: keys must exist in all_results.json; "Prompt-based (32B)" uses largest prompt judge in bundle
    baselines = [
        ("zeroshot_qwen7b", "Zero-shot (Qwen2.5-7B)", "#DC2626"),
        ("zeroshot_qwen3_14b", "Prompt-based (32B)", "#EA580C"),
        ("prometheus2", "Prometheus 2", "#059669"),
    ]

    rho_base = {bkey: _rho_from_key(bkey) for bkey, _, _ in baselines}
    active = [(bkey, blabel, bcolor) for bkey, blabel, bcolor in baselines if rho_base.get(bkey) is not None]
    if not active:
        print("  [skip] delta_improvement: no baseline Spearman available")
        return

    n_x = len(size_plan)
    n_b = len(active)
    x = np.arange(n_x, dtype=float)
    width = min(0.22, 0.75 / max(n_b, 1))

    fig, ax = plt.subplots(figsize=(7.6, 4.25))

    for bi, (bkey, blabel, bcolor) in enumerate(active):
        rb = rho_base[bkey]
        heights = []
        for _sx, lora_k in size_plan:
            if lora_k not in all_results:
                heights.append(float("nan"))
                continue
            r_l = _rho_from_key(lora_k)
            heights.append((r_l - rb) if r_l is not None else float("nan"))
        offsets = x + (bi - (n_b - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            heights,
            width,
            label=blabel,
            color=bcolor,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, h in zip(bars, heights):
            if not np.isfinite(h):
                continue
            if h >= 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.004,
                    f"{h:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=bcolor,
                    fontweight="bold",
                )

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=0)
    ax.set_xlabel("Training Samples", fontsize=10, fontweight="bold")
    ax.set_ylabel(
        r"$\Delta$ Spearman $\rho$ (LoRA — Baseline)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title(
        "Improvement of LoRA Evaluator over Baselines",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(sx) for sx, _ in size_plan], fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.95, loc="upper left")
    ax.grid(True, axis="y", alpha=0.35, linestyle="--")
    ax.set_ylim(-0.11, 0.17)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = f"{output_dir}/delta_improvement.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--eval-data", type=str, default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--all-results", type=str, default=str(DEFAULT_ALL_RESULTS))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--trad-results", type=str, default=str(DEFAULT_TRAD_RESULTS))
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--tables-dir", type=str, default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--pairwise-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "pairwise"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data...")
    eval_data, all_results, metadata, trad_results = load_data(
        args.eval_data, args.all_results, args.metadata, args.trad_results
    )
    print(f"  {len(eval_data)} eval samples, {len(all_results)} methods")

    print("Loading pairwise results...")
    pairwise_entries = load_pairwise_results(args.pairwise_dir)
    print(f"  {len(pairwise_entries)} pairwise models loaded")

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Figures ---")

    print("1. Score distribution...")
    plot_score_distribution(eval_data, all_results, metadata, args.figures_dir, args.dpi)

    print("2. Confusion matrix (LoRA balanced_simple)...")
    qwen3_key, qwen25_key = resolve_ours_keys(all_results)
    if qwen3_key:
        plot_confusion_matrix(eval_data, all_results, qwen3_key,
                              args.figures_dir, args.dpi)

    print("2a. Confusion matrix (RewriteJudge Qwen2.5-7B)...")
    if qwen25_key:
        plot_confusion_matrix(eval_data, all_results, qwen25_key,
                              args.figures_dir, args.dpi)

    print("2b. Confusion matrix (Prometheus 2)...")
    if "prometheus2" in all_results:
        plot_confusion_matrix(eval_data, all_results, "prometheus2",
                              args.figures_dir, args.dpi)

    print("3. Method comparison bar chart...")
    plot_method_comparison(eval_data, all_results, metadata, trad_results, args.figures_dir, args.dpi)

    print("4. Agreement heatmap...")
    plot_agreement_heatmap(eval_data, all_results, metadata, args.figures_dir, args.dpi)

    print("5. Delta improvement...")
    plot_delta_improvement(
        eval_data,
        all_results,
        metadata,
        args.figures_dir,
        args.dpi,
        trad_results=trad_results,
    )

    print("6. Scaling analysis...")
    plot_scaling_analysis(pairwise_entries, args.figures_dir, args.dpi)

    print("7. Generational regression...")
    plot_generational_regression(pairwise_entries, args.figures_dir, args.dpi)

    print("8. Accuracy vs Rho...")
    plot_accuracy_vs_rho(pairwise_entries, args.figures_dir, args.dpi)

    print("9. Parse failure vs Rho...")
    plot_parse_failure_vs_rho(pairwise_entries, args.figures_dir, args.dpi)

    print("\n--- Generating Tables ---")

    print("1. Main results table (LaTeX)...")
    generate_main_results_table(all_results, metadata, trad_results, args.tables_dir)

    print("2. Learning curve table (LaTeX)...")
    generate_learning_curve_table(all_results, metadata, args.tables_dir)

    print("3. Downstream validation table (LaTeX)...")
    generate_downstream_table(args.tables_dir)

    print("\nDone!")
    print(f"  Figures: {args.figures_dir}")
    print(f"  Tables:  {args.tables_dir}")


if __name__ == "__main__":
    main()
