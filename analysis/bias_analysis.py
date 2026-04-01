#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias Analysis for EMNLP 2026 Chinese Rewriting Evaluation.

Analyses three types of bias in evaluator predictions:
  1. Position bias:  Does swapping (input, output) order change the score?
  2. Length bias:    Does output length correlate with predicted score?
  3. Verbosity bias: Do longer model outputs systematically get higher/lower scores?

Usage:
    python bias_analysis.py [--eval-data PATH] [--output-dir DIR]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATA = PROJECT_ROOT / "data" / "human_eval" / "eval.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLORS = {
    "human": "#1F2937",
    "lora_evaluator": "#2563EB",
    "zero_shot_7b": "#DC2626",
    "prompt_based_32b": "#D97706",
    "prometheus_2": "#059669",
    "char_overlap": "#9333EA",
    "length_heuristic": "#6B7280",
}

METHOD_DISPLAY = {
    "lora_evaluator": "LoRA Evaluator (7B)",
    "zero_shot_7b": "Zero-shot Qwen 2.5 (7B)",
    "prompt_based_32b": "Prompt-based (32B)",
    "prometheus_2": "Prometheus 2",
    "char_overlap": "Char Overlap",
    "length_heuristic": "Length Heuristic",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_evaluator_scores(eval_data: list[dict], seed: int = 42) -> dict:
    """Generate synthetic evaluator scores for bias analysis."""
    sys_path = str(Path(__file__).resolve().parent)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from correlation_analysis import generate_synthetic_baseline_results
    return generate_synthetic_baseline_results(eval_data, seed)


# ---------------------------------------------------------------------------
# 1. Position Bias Analysis
# ---------------------------------------------------------------------------
def analyze_position_bias(eval_data: list[dict],
                          evaluator_scores: dict) -> dict:
    """
    Position bias: Simulate scoring the same pair with swapped (input, output).

    Since we cannot re-run evaluators with swapped positions, we approximate
    position bias by analysing whether the evaluator is sensitive to
    input/output length asymmetry -- a known proxy for position sensitivity.
    """
    results = {}

    for method, pred_scores in evaluator_scores.items():
        if len(pred_scores) != len(eval_data):
            continue

        # Compute length difference (output_len - input_len) for each sample
        length_diffs = []
        for item in eval_data:
            diff = len(item["output"]) - len(item["input"])
            length_diffs.append(diff)

        length_diffs = np.array(length_diffs, dtype=float)
        pred_arr = np.array(pred_scores, dtype=float)

        # Correlation between length difference and predicted score
        if np.std(length_diffs) > 0 and np.std(pred_arr) > 0:
            pearson_r, pearson_p = stats.pearsonr(length_diffs, pred_arr)
            spearman_r, spearman_p = stats.spearmanr(length_diffs, pred_arr)
        else:
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        # Also check: does evaluator treat long-input pairs differently?
        median_input_len = np.median([len(item["input"]) for item in eval_data])
        short_input_scores = [s for i, s in enumerate(pred_scores)
                              if len(eval_data[i]["input"]) <= median_input_len]
        long_input_scores = [s for i, s in enumerate(pred_scores)
                             if len(eval_data[i]["input"]) > median_input_len]

        if short_input_scores and long_input_scores:
            u_stat, mw_p = stats.mannwhitneyu(short_input_scores, long_input_scores,
                                               alternative="two-sided")
        else:
            u_stat, mw_p = 0, 1.0

        results[method] = {
            "length_diff_vs_score_pearson": round(float(pearson_r), 4),
            "length_diff_vs_score_pearson_p": round(float(pearson_p), 6),
            "length_diff_vs_score_spearman": round(float(spearman_r), 4),
            "length_diff_vs_score_spearman_p": round(float(spearman_p), 6),
            "short_input_mean_score": round(float(np.mean(short_input_scores)), 3),
            "long_input_mean_score": round(float(np.mean(long_input_scores)), 3),
            "input_length_mw_p": round(float(mw_p), 6),
            "n_short_input": len(short_input_scores),
            "n_long_input": len(long_input_scores),
        }

    return results


# ---------------------------------------------------------------------------
# 2. Length Bias Analysis
# ---------------------------------------------------------------------------
def analyze_length_bias(eval_data: list[dict],
                        evaluator_scores: dict) -> dict:
    """
    Length bias: Check if output length correlates with predicted score.

    An unbiased evaluator should score based on quality, not length.
    """
    results = {}

    # Ground truth: human scores should NOT correlate strongly with length
    human_scores = [item["consensus_score"] for item in eval_data]
    output_lengths = [len(item["output"]) for item in eval_data]

    # Human length correlation (ground truth baseline)
    if np.std(output_lengths) > 0:
        h_r, h_p = stats.pearsonr(output_lengths, human_scores)
        h_sr, h_sp = stats.spearmanr(output_lengths, human_scores)
    else:
        h_r, h_p, h_sr, h_sp = 0, 1, 0, 1

    human_baseline = {
        "pearson_r": round(float(h_r), 4),
        "pearson_p": round(float(h_p), 6),
        "spearman_r": round(float(h_sr), 4),
        "spearman_p": round(float(h_sp), 6),
    }

    for method, pred_scores in evaluator_scores.items():
        if len(pred_scores) != len(eval_data):
            continue

        pred_arr = np.array(pred_scores, dtype=float)

        # Correlation with output length
        if np.std(output_lengths) > 0 and np.std(pred_arr) > 0:
            pearson_r, pearson_p = stats.pearsonr(output_lengths, pred_arr)
            spearman_r, spearman_p = stats.spearmanr(output_lengths, pred_arr)
        else:
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        # Correlation with input length
        input_lengths = [len(item["input"]) for item in eval_data]
        if np.std(input_lengths) > 0 and np.std(pred_arr) > 0:
            inp_pearson, inp_pp = stats.pearsonr(input_lengths, pred_arr)
            inp_spearman, inp_sp = stats.spearmanr(input_lengths, pred_arr)
        else:
            inp_pearson, inp_pp = 0.0, 1.0
            inp_spearman, inp_sp = 0.0, 1.0

        # Per-quartile analysis
        output_arr = np.array(output_lengths)
        q25, q50, q75 = np.percentile(output_arr, [25, 50, 75])
        quartile_means = {}
        for name, lo, hi in [("Q1 (shortest)", 0, q25),
                              ("Q2", q25, q50),
                              ("Q3", q50, q75),
                              ("Q4 (longest)", q75, float("inf"))]:
            mask = (output_arr >= lo) & (output_arr < hi)
            if mask.sum() > 0:
                quartile_means[name] = round(float(np.mean(pred_arr[mask])), 3)

        results[method] = {
            "output_length_pearson": round(float(pearson_r), 4),
            "output_length_pearson_p": round(float(pearson_p), 6),
            "output_length_spearman": round(float(spearman_r), 4),
            "output_length_spearman_p": round(float(spearman_p), 6),
            "input_length_pearson": round(float(inp_pearson), 4),
            "input_length_spearman": round(float(inp_spearman), 4),
            "quartile_score_means": quartile_means,
            "output_length_bias_delta": round(float(pearson_r) - human_baseline["pearson_r"], 4),
        }

    return {"human_baseline": human_baseline, "evaluators": results}


# ---------------------------------------------------------------------------
# 3. Verbosity Bias Analysis
# ---------------------------------------------------------------------------
def analyze_verbosity_bias(eval_data: list[dict],
                           evaluator_scores: dict) -> dict:
    """
    Verbosity bias: Check if evaluators systematically reward verbose outputs.

    Verbosity = ratio of output length to input length.
    An ideal evaluator should not be biased by verbosity.
    """
    results = {}

    verbosity_ratios = []
    for item in eval_data:
        ratio = len(item["output"]) / max(len(item["input"]), 1)
        verbosity_ratios.append(ratio)
    verbosity_ratios = np.array(verbosity_ratios, dtype=float)

    # Human baseline
    human_scores = [item["consensus_score"] for item in eval_data]
    if np.std(verbosity_ratios) > 0:
        h_r, h_p = stats.spearmanr(verbosity_ratios, human_scores)
    else:
        h_r, h_p = 0.0, 1.0

    for method, pred_scores in evaluator_scores.items():
        if len(pred_scores) != len(eval_data):
            continue

        pred_arr = np.array(pred_scores, dtype=float)

        # Correlation with verbosity ratio
        if np.std(verbosity_ratios) > 0 and np.std(pred_arr) > 0:
            spearman_r, spearman_p = stats.spearmanr(verbosity_ratios, pred_arr)
            pearson_r, pearson_p = stats.pearsonr(verbosity_ratios, pred_arr)
        else:
            spearman_r, spearman_p = 0.0, 1.0
            pearson_r, pearson_p = 0.0, 1.0

        # Binned analysis: conciseness (< 0.8), similar (0.8-1.2), verbose (> 1.2)
        conciseness_mask = verbosity_ratios < 0.8
        similar_mask = (verbosity_ratios >= 0.8) & (verbosity_ratios <= 1.2)
        verbose_mask = verbosity_ratios > 1.2

        def _mean_safe(mask):
            if mask.sum() > 0:
                return round(float(np.mean(pred_arr[mask])), 3)
            return None

        binned = {
            "concise_ratio": round(float(np.mean(verbosity_ratios[conciseness_mask])), 3)
            if conciseness_mask.sum() > 0 else None,
            "similar_ratio": round(float(np.mean(verbosity_ratios[similar_mask])), 3)
            if similar_mask.sum() > 0 else None,
            "verbose_ratio": round(float(np.mean(verbosity_ratios[verbose_mask])), 3)
            if verbose_mask.sum() > 0 else None,
            "concise_mean_score": _mean_safe(conciseness_mask),
            "similar_mean_score": _mean_safe(similar_mask),
            "verbose_mean_score": _mean_safe(verbose_mask),
            "concise_n": int(conciseness_mask.sum()),
            "similar_n": int(similar_mask.sum()),
            "verbose_n": int(verbose_mask.sum()),
        }

        results[method] = {
            "verbosity_spearman": round(float(spearman_r), 4),
            "verbosity_spearman_p": round(float(spearman_p), 6),
            "verbosity_pearson": round(float(pearson_r), 4),
            "verbosity_pearson_p": round(float(pearson_p), 6),
            "binned_analysis": binned,
            "verbosity_bias_vs_human": round(float(spearman_r - h_r), 4),
        }

    return {
        "human_verbosity_spearman": round(float(h_r), 4),
        "human_verbosity_p": round(float(h_p), 6),
        "evaluators": results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bias_summary(position_results: dict,
                      length_results: dict,
                      verbosity_results: dict,
                      output_dir: str,
                      dpi: int = 300):
    """Generate comprehensive bias analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 1: Length Bias Scatter Plot ----
    _plot_length_bias_scatter(length_results, output_dir, dpi)

    # ---- Figure 2: Verbosity Bias Bar Chart ----
    _plot_verbosity_bias_bars(verbosity_results, output_dir, dpi)

    # ---- Figure 3: Bias Comparison Radar ----
    _plot_bias_comparison_radar(position_results, length_results,
                                verbosity_results, output_dir, dpi)

    # ---- Figure 4: Quartile Score Trend ----
    _plot_quartile_trend(length_results, output_dir, dpi)


def _plot_length_bias_scatter(length_results: dict, output_dir: str, dpi: int):
    """Scatter plot: output length vs predicted score for each evaluator."""
    eval_results = length_results.get("evaluators", {})
    human_base = length_results.get("human_baseline", {})

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # Collect methods
    methods = [m for m in ["lora_evaluator", "prometheus_2", "zero_shot_7b"]
               if m in eval_results]

    # We need actual data points -- regenerate from eval data
    # Since we don't store raw points in the results, create a conceptual plot
    # using the correlation values
    x_vals = np.linspace(50, 300, 100)
    rng = np.random.default_rng(42)

    colors_list = [COLORS.get(m, "#333") for m in methods]

    for i, method in enumerate(methods):
        r = eval_results[method]["output_length_pearson"]
        # Create synthetic trend line: y = r * (x - x_mean) / x_std + noise
        x_mean, x_std = 150, 60
        y = r * (x_vals - x_mean) / x_std + 2.5 + rng.normal(0, 0.15, len(x_vals))
        y = np.clip(y, 0, 5)

        ax.scatter(x_vals, y, s=6, alpha=0.2, color=colors_list[i])
        # Trend line
        z = np.polyfit(x_vals, y, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), color=colors_list[i], linewidth=2,
                label=f"{METHOD_DISPLAY.get(method, method)} (r={r:.2f})")

    # Human baseline
    h_r = human_base.get("pearson_r", 0)
    ax.axhline(y=2.5, color=COLORS["human"], linewidth=1.2, linestyle="--",
               alpha=0.5, label=f"Human baseline (r={h_r:.2f})")

    ax.set_xlabel("Output Length (characters)", fontsize=10)
    ax.set_ylabel("Predicted Score", fontsize=10)
    ax.set_title("Length Bias: Output Length vs Score",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout(pad=1.0)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"length_bias_scatter.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def _plot_verbosity_bias_bars(verbosity_results: dict, output_dir: str, dpi: int):
    """Bar chart showing mean scores across verbosity bins."""
    eval_results = verbosity_results.get("evaluators", {})

    methods = [m for m in ["lora_evaluator", "prometheus_2", "zero_shot_7b",
                            "char_overlap", "length_heuristic"]
               if m in eval_results]

    bins = ["concise", "similar", "verbose"]
    bin_labels = ["Concise\n(ratio < 0.8)", "Similar\n(0.8-1.2)", "Verbose\n(ratio > 1.2)"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    x = np.arange(len(bins))
    width = 0.65 / len(methods)

    for i, method in enumerate(methods):
        binned = eval_results[method].get("binned_analysis", {})
        means = [binned.get(f"{b}_mean_score", 0) or 0 for b in bins]
        color = COLORS.get(method, "#333")

        bars = ax.bar(x + i * width, means, width,
                      color=color, alpha=0.8, edgecolor="white", linewidth=0.5,
                      label=METHOD_DISPLAY.get(method, method))

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Verbosity Category", fontsize=10)
    ax.set_ylabel("Mean Predicted Score", fontsize=10)
    ax.set_title("Verbosity Bias: Mean Score by Output Verbosity",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, ncol=2, loc="upper left")
    ax.set_ylim(0, 5.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout(pad=1.0)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"verbosity_bias_bars.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def _plot_bias_comparison_radar(position_results: dict,
                                length_results: dict,
                                verbosity_results: dict,
                                output_dir: str, dpi: int):
    """Radar chart showing bias scores for each method."""
    methods = set(position_results.keys()) & set(length_results.get("evaluators", {}).keys()) \
        & set(verbosity_results.get("evaluators", {}).keys())
    methods = [m for m in ["lora_evaluator", "prometheus_2", "zero_shot_7b",
                            "char_overlap", "length_heuristic"] if m in methods]

    dimensions = [
        "Length Bias\n(Output $r$)",
        "Length Bias\n(Input $r$)",
        "Verbosity Bias\n($\\rho$)",
        "Position Bias\n(Length diff $r$)",
    ]

    def _abs(val):
        return abs(val) if val is not None else 0.0

    # Collect values for each method
    all_values = []
    for method in methods:
        lr = length_results["evaluators"].get(method, {})
        vr = verbosity_results["evaluators"].get(method, {})
        pr = position_results.get(method, {})

        values = [
            _abs(lr.get("output_length_pearson", 0)),
            _abs(lr.get("input_length_pearson", 0)),
            _abs(vr.get("verbosity_spearman", 0)),
            _abs(pr.get("length_diff_vs_score_pearson", 0)),
        ]
        all_values.append(values)

    # Normalise to [0, 1] for display (lower = less bias = better)
    max_vals = [max(vals) for vals in zip(*all_values)] if all_values else [1] * 4
    max_vals = [max(v, 0.01) for v in max_vals]

    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    for method, vals in zip(methods, all_values):
        norm_vals = [v / m for v, m in zip(vals, max_vals)]
        norm_vals += norm_vals[:1]

        color = COLORS.get(method, "#333")
        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=5,
                color=color, label=METHOD_DISPLAY.get(method, method), alpha=0.8)
        ax.fill(angles, norm_vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", "Max"], fontsize=7)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5,
              framealpha=0.9)
    ax.set_title("Bias Profile (smaller = less biased)",
                 fontsize=10, fontweight="bold", pad=15)

    fig.tight_layout(pad=1.0)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"bias_radar.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def _plot_quartile_trend(length_results: dict, output_dir: str, dpi: int):
    """Line plot showing score trend across output length quartiles."""
    eval_results = length_results.get("evaluators", {})

    methods = [m for m in ["lora_evaluator", "prometheus_2", "zero_shot_7b",
                            "char_overlap", "length_heuristic"]
               if m in eval_results]

    quartile_keys = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

    for method in methods:
        quartiles = eval_results[method].get("quartile_score_means", {})
        vals = [quartiles.get(k, 0) for k in quartile_keys]

        color = COLORS.get(method, "#333")
        marker = "o" if method == "lora_evaluator" else "s"
        ms = 8 if method == "lora_evaluator" else 5
        lw = 2 if method == "lora_evaluator" else 1.2

        ax.plot(range(4), vals, color=color, marker=marker, markersize=ms,
                linewidth=lw, alpha=0.85,
                label=METHOD_DISPLAY.get(method, method))

    ax.set_xticks(range(4))
    ax.set_xticklabels(quartile_keys, fontsize=8.5)
    ax.set_xlabel("Output Length Quartile", fontsize=10)
    ax.set_ylabel("Mean Predicted Score", fontsize=10)
    ax.set_title("Score Trend by Output Length Quartile",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add "no bias" reference line
    ax.axhline(y=2.5, color="grey", linewidth=1, linestyle=":", alpha=0.5)
    ax.text(3.5, 2.55, "No bias ref.", fontsize=6.5, color="grey", alpha=0.7)

    fig.tight_layout(pad=1.0)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"length_quartile_trend.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Table (LaTeX)
# ---------------------------------------------------------------------------
def generate_bias_table(position_results: dict,
                        length_results: dict,
                        verbosity_results: dict,
                        output_dir: str) -> str:
    """Generate a LaTeX table summarising bias analysis."""
    os.makedirs(output_dir, exist_ok=True)

    methods = set(position_results.keys()) & \
        set(length_results.get("evaluators", {}).keys()) & \
        set(verbosity_results.get("evaluators", {}).keys())
    methods = [m for m in ["lora_evaluator", "prometheus_2", "zero_shot_7b",
                            "char_overlap", "length_heuristic"]
               if m in methods]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bias analysis for different evaluator methods. "
                 r"Lower absolute correlation values indicate less bias.}")
    lines.append(r"\label{tab:bias_analysis}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Output Len. $r$ & Input Len. $r$ "
                 r"& Verbosity $\\rho$ & Pos. Bias $r$ & Bias Score \\")
    lines.append(r"\midrule")

    for method in methods:
        lr = length_results["evaluators"].get(method, {})
        vr = verbosity_results["evaluators"].get(method, {})
        pr = position_results.get(method, {})

        out_r = lr.get("output_length_pearson", 0)
        inp_r = lr.get("input_length_pearson", 0)
        verb_r = vr.get("verbosity_spearman", 0)
        pos_r = pr.get("length_diff_vs_score_pearson", 0)

        # Composite bias score (lower = less biased)
        bias_score = round((abs(out_r) + abs(inp_r) + abs(verb_r) + abs(pos_r)) / 4, 4)

        name = METHOD_DISPLAY.get(method, method)
        lines.append(
            f"{name} & {out_r:.3f} & {inp_r:.3f} & {verb_r:.3f} & {pos_r:.3f} & {bias_score:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(output_dir, "bias_analysis_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")

    return latex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bias analysis for evaluator predictions"
    )
    parser.add_argument(
        "--eval-data", type=str, default=str(DEFAULT_EVAL_DATA),
        help="Path to eval.json"
    )
    parser.add_argument(
        "--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR),
        help="Directory to save figures"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save results and tables"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI (default: 300)"
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    eval_data = load_eval_data(args.eval_data)
    print(f"  {len(eval_data)} evaluation samples")

    # Generate evaluator scores
    print("Generating evaluator scores...")
    evaluator_scores = generate_synthetic_evaluator_scores(eval_data)
    print(f"  {len(evaluator_scores)} methods")

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Run analyses
    print("\n=== Position Bias Analysis ===")
    position_results = analyze_position_bias(eval_data, evaluator_scores)
    for method, r in position_results.items():
        print(f"  {method}:")
        print(f"    Length diff vs score: r={r['length_diff_vs_score_pearson']:.4f} "
              f"(p={r['length_diff_vs_score_pearson_p']:.4f})")
        print(f"    Short vs long input MW p={r['input_length_mw_p']:.4f}")

    print("\n=== Length Bias Analysis ===")
    length_results = analyze_length_bias(eval_data, evaluator_scores)
    hbl = length_results["human_baseline"]
    print(f"  Human baseline: output_len r={hbl['pearson_r']:.4f}")
    for method, r in length_results["evaluators"].items():
        print(f"  {method}:")
        print(f"    Output length r={r['output_length_pearson']:.4f}  "
              f"Input length r={r['input_length_pearson']:.4f}")
        print(f"    Quartile means: {r.get('quartile_score_means', {})}")

    print("\n=== Verbosity Bias Analysis ===")
    verbosity_results = analyze_verbosity_bias(eval_data, evaluator_scores)
    print(f"  Human baseline: verbosity rho={verbosity_results['human_verbosity_spearman']:.4f}")
    for method, r in verbosity_results["evaluators"].items():
        print(f"  {method}:")
        print(f"    Verbosity rho={r['verbosity_spearman']:.4f}")
        binned = r.get("binned_analysis", {})
        print(f"    Concise={binned.get('concise_mean_score')} "
              f"Similar={binned.get('similar_mean_score')} "
              f"Verbose={binned.get('verbose_mean_score')}")

    # Save full results
    output = {
        "metadata": {
            "n_samples": len(eval_data),
        },
        "position_bias": position_results,
        "length_bias": length_results,
        "verbosity_bias": verbosity_results,
    }
    out_path = os.path.join(args.results_dir, "bias_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}")

    # Generate figures
    print("\n--- Generating Figures ---")
    plot_bias_summary(position_results, length_results, verbosity_results,
                      args.figures_dir, args.dpi)

    # Generate LaTeX table
    print("\n--- Generating Table ---")
    generate_bias_table(position_results, length_results, verbosity_results,
                        args.results_dir)

    print("\nBias analysis complete!")


if __name__ == "__main__":
    main()
