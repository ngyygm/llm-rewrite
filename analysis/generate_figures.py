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
    "lora_balanced_simple": "#2563EB",
    "lora_balanced_reasoning": "#3B82F6",
    "lora_score_only_full": "#93C5FD",
    "lora_original_reasoning": "#60A5FA",
    "prometheus2": "#059669",
    "zeroshot_qwen14b": "#D97706",
    "zeroshot_qwen7b": "#DC2626",
    "geval_qwen7b": "#9333EA",
}


def load_data():
    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    with open(DEFAULT_ALL_RESULTS, "r") as f:
        all_results = json.load(f)
    with open(DEFAULT_METADATA, "r") as f:
        metadata = json.load(f)
    trad_path = DEFAULT_TRAD_RESULTS
    trad_results = json.load(open(trad_path)) if trad_path.exists() else {}
    return eval_data, all_results, metadata, trad_results


def get_display_name(key):
    m = json.load(open(DEFAULT_METADATA)) if os.path.exists(DEFAULT_METADATA) else {}
    return m.get(key, {}).get("display_name", key)


def plot_score_distribution(eval_data, all_results, metadata, output_dir, dpi=300):
    """Plot score distribution: human annotations vs top evaluators."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    human_scores = [item["consensus_score"] for item in eval_data]
    labels = list(range(6))

    # Select methods to show
    show_methods = [
        ("lora_balanced_simple", "LoRA (Ours)"),
        ("prometheus2", "Prometheus 2"),
        ("zeroshot_qwen7b", "Zero-shot 7B"),
    ]

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
            color = PALETTE.get(key, "#666")
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
    ax2.legend(handles=legend_elements, fontsize=7, framealpha=0.9, loc="upper right")
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

    display = get_display_name(method)

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


def plot_method_comparison(all_results, metadata, trad_results, output_dir, dpi=300):
    """Main comparison bar chart: Spearman rho for all methods."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from scipy import stats as _stats

    with open(DEFAULT_EVAL_DATA, "r") as f:
        eval_data = json.load(f)
    human_avg = [item["avg_score"] for item in eval_data]

    # Collect all methods with Spearman values
    entries = []

    for method, scores in all_results.items():
        preds = np.array([s if s >= 0 else np.nan for s in scores], dtype=float)
        refs = np.array(human_avg, dtype=float)
        valid = ~(np.isnan(preds) | np.isnan(refs))
        if valid.sum() >= 2:
            rho, _ = _stats.spearmanr(preds[valid], refs[valid])
        else:
            rho = 0.0
        display = metadata.get(method, {}).get("display_name", method)
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

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Left: LLM-based methods (main comparison)
    ax = axes[0]
    main_entries = lora_entries + llm_entries
    main_entries.sort(key=lambda x: x[1], reverse=True)

    names = [e[0] for e in main_entries]
    values = [e[1] for e in main_entries]
    colors = [PALETTE.get(e[3], "#666") for e in main_entries]

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

    # Right: Traditional metrics
    ax2 = axes[1]
    trad_entries.sort(key=lambda x: x[1], reverse=True)
    names2 = [e[0] for e in trad_entries]
    values2 = [e[1] for e in trad_entries]
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
    ax2.set_xlim(-0.7, 0.0)

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

    # Collect all methods
    entries = []
    for method, scores in all_results.items():
        meta = metadata.get(method, {})
        if meta.get("category") == "lora" and ("50" in method or "100" in method or "200" in method or "400" in method):
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
            "name": meta.get("display_name", method),
            "spearman": sp, "spearman_p": spp,
            "pearson": pr, "pearson_p": prp,
            "kendall": kt, "kendall_p": ktp,
            "mae": mae, "rmse": rmse,
            "exact": exact, "within_1": within_1, "within_2": within_2,
            "is_ours": method == "lora_balanced_simple",
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

        # Determine type
        if "LoRA" in name or "Ours" in name:
            type_str = "Fine-tuned (7B)"
        elif "Prometheus" in name:
            type_str = "Fine-tuned (7B)"
        elif "Zero-shot" in name or "G-Eval" in name:
            size = "7B" if "7B" in name else "14B"
            type_str = f"Zero-shot ({size})"
        else:
            type_str = "Traditional"

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
    lines.append(r"Training Samples & Spearman $\rho$ vs avg & Spearman $\rho$ vs consensus \\")
    lines.append(r"\midrule")

    subset_entries = []
    for size in [50, 100, 200, 400]:
        key = f"lora_balanced_simple_{size}"
        if key in all_results:
            preds = np.array([s if s >= 0 else np.nan for s in all_results[key]], dtype=float)
            refs_avg = np.array(human_avg, dtype=float)
            refs_cons = np.array([item["consensus_score"] for item in eval_data], dtype=float)
            valid_a = ~(np.isnan(preds) | np.isnan(refs_avg))
            valid_c = ~(np.isnan(preds) | np.isnan(refs_cons))
            rho_a, _ = _stats.spearmanr(preds[valid_a], refs_avg[valid_a]) if valid_a.sum() >= 2 else (0, 1)
            rho_c, _ = _stats.spearmanr(preds[valid_c], refs_cons[valid_c]) if valid_c.sum() >= 2 else (0, 1)
            subset_entries.append((size, rho_a, rho_c))

    # Full model
    if "lora_balanced_simple" in all_results:
        preds = np.array([s if s >= 0 else np.nan for s in all_results["lora_balanced_simple"]], dtype=float)
        refs_avg = np.array(human_avg, dtype=float)
        refs_cons = np.array([item["consensus_score"] for item in eval_data], dtype=float)
        valid_a = ~(np.isnan(preds) | np.isnan(refs_avg))
        valid_c = ~(np.isnan(preds) | np.isnan(refs_cons))
        rho_a, _ = _stats.spearmanr(preds[valid_a], refs_avg[valid_a]) if valid_a.sum() >= 2 else (0, 1)
        rho_c, _ = _stats.spearmanr(preds[valid_c], refs_cons[valid_c]) if valid_c.sum() >= 2 else (0, 1)
        subset_entries.append((1008, rho_a, rho_c))

    for size, rho_a, rho_c in subset_entries:
        label = str(size) + (r" (balanced)" if size == 1008 else "")
        bold = r"\textbf{" if size == 1008 else ""
        bold_end = r"}" if size == 1008 else ""
        lines.append(f"{bold}{label}{bold_end} & {bold}{rho_a:+.4f}{bold_end} & {bold}{rho_c:+.4f}{bold_end} \\\\")

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


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--eval-data", type=str, default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--all-results", type=str, default=str(DEFAULT_ALL_RESULTS))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--trad-results", type=str, default=str(DEFAULT_TRAD_RESULTS))
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--tables-dir", type=str, default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data...")
    eval_data, all_results, metadata, trad_results = load_data()
    print(f"  {len(eval_data)} eval samples, {len(all_results)} methods")

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Figures ---")

    print("1. Score distribution...")
    plot_score_distribution(eval_data, all_results, metadata, args.figures_dir, args.dpi)

    print("2. Confusion matrix (LoRA balanced_simple)...")
    if "lora_balanced_simple" in all_results:
        plot_confusion_matrix(eval_data, all_results, "lora_balanced_simple",
                              args.figures_dir, args.dpi)

    print("2b. Confusion matrix (Prometheus 2)...")
    if "prometheus2" in all_results:
        plot_confusion_matrix(eval_data, all_results, "prometheus2",
                              args.figures_dir, args.dpi)

    print("3. Method comparison bar chart...")
    plot_method_comparison(all_results, metadata, trad_results, args.figures_dir, args.dpi)

    print("\n--- Generating Tables ---")

    print("1. Main results table (LaTeX)...")
    generate_main_results_table(all_results, metadata, trad_results, args.tables_dir)

    print("2. Learning curve table (LaTeX)...")
    generate_learning_curve_table(all_results, metadata, args.tables_dir)

    print("\nDone!")
    print(f"  Figures: {args.figures_dir}")
    print(f"  Tables:  {args.tables_dir}")


if __name__ == "__main__":
    main()
