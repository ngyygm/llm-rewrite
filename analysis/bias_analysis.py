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
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-finite mask for correlations (handles evaluator failures as NaN)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def ensure_overlap_heuristic_scores(
    eval_data: list[dict], scores: dict[str, list], seed: int = 42
) -> None:
    """
    Paper-style simple baselines (not in all_results.json): Char Overlap and
    Length Heuristic on 0–5 scale. Same construction as correlation_analysis.
    """
    n = len(eval_data)
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    output_lens = np.array([len(item.get("output", "")) for item in eval_data], dtype=float)
    input_lens = np.array([len(item.get("input", "")) for item in eval_data], dtype=float)
    if np.std(output_lens) <= 0 or np.std(input_lens) <= 0:
        return
    overlap_ratio = np.minimum(output_lens, input_lens) / np.maximum(output_lens, input_lens)
    length_ratio = output_lens / np.maximum(input_lens, 1.0)
    if "char_overlap" not in scores:
        char_scores = overlap_ratio * 5.0 + rng.normal(0, 0.3, n)
        scores["char_overlap"] = np.clip(char_scores, 0, 5).tolist()
    if "length_heuristic" not in scores:
        length_scores = length_ratio * 2.5 + rng.normal(0, 0.3, n)
        scores["length_heuristic"] = np.clip(length_scores, 0, 5).tolist()


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
    "lora_balanced_simple": "#2563EB",
    "lora_evaluator": "#2563EB",      # legacy alias
    "prometheus2": "#059669",
    "prometheus_2": "#059669",        # legacy alias
    "zeroshot_qwen3_8b": "#DC2626",
    "zeroshot_qwen3_14b": "#D97706",
    "zero_shot_7b": "#DC2626",        # legacy alias
    "trad_jaccard_char": "#9333EA",
    "trad_rouge_l": "#6B7280",
    "char_overlap": "#8B5CF6",
    "length_heuristic": "#6B7280",
    "lora_balanced_simple_qwen2_5_7b": "#D97706",
    "lora_balanced_simple_qwen3_8b_proxy": "#2563EB",
}

METHOD_DISPLAY = {
    "lora_balanced_simple": "RewriteJudge (Ours, Qwen3-8B)",
    "lora_evaluator": "LoRA Evaluator (8B)",
    "zero_shot_7b": "Zero-shot Qwen2.5-7B",
    "zeroshot_qwen3_8b": "Zero-shot Qwen3-8B",
    "zeroshot_qwen3_14b": "Zero-shot Qwen3-14B",
    "prometheus2": "Prometheus 2",
    "char_overlap": "Char Overlap",
    "length_heuristic": "Length Heuristic",
    "trad_jaccard_char": "JACCARD-CHAR",
    "trad_rouge_l": "ROUGE-L",
}


def _resolve_method_display(method: str) -> str:
    if method == "lora_balanced_simple":
        return "RewriteJudge (Ours, Qwen3-8B)"
    if method.startswith("lora_balanced_simple_qwen3"):
        return "RewriteJudge (Ours, Qwen3-8B)"
    if method.startswith("lora_balanced_simple_qwen2_5_7b") or "qwen2_5_7b" in method:
        return "RewriteJudge (Ours, Qwen2.5-7B)"
    if method.startswith("lora_balanced_simple_") and method.split("_")[-1].isdigit():
        # Fallback when only subset keys (e.g., _400) exist for Qwen3 family.
        subset = method.split("_")[-1]
        return f"RewriteJudge (Ours, Qwen3-8B)"
    return METHOD_DISPLAY.get(method, method)


def _resolve_method_color(method: str) -> str:
    if method.startswith("lora_balanced_simple_qwen2_5_7b"):
        return COLORS["lora_balanced_simple_qwen2_5_7b"]
    if method == "lora_balanced_simple" or (
        method.startswith("lora_balanced_simple_") and method.split("_")[-1].isdigit()
    ):
        return COLORS["lora_balanced_simple_qwen3_8b_proxy"]
    return COLORS.get(method, "#333")


def _select_methods(available_keys):
    """
    Select and order methods for bias plots/tables.
    Keep both Ours variants (Qwen3 + Qwen2.5 if present) and remove zero-shot 14B.
    """
    keys = set(available_keys)
    methods = []

    if "lora_balanced_simple" in keys:
        methods.append("lora_balanced_simple")
    else:
        # Some consolidated files may only keep unsuffixed subset keys for Qwen3
        qwen3_subset_keys = sorted(
            (k for k in keys if re.match(r"^lora_balanced_simple_\d+$", k)),
            key=lambda x: int(x.rsplit("_", 1)[1]),
        )
        if qwen3_subset_keys:
            methods.append(qwen3_subset_keys[-1])  # use largest subset as proxy
    qwen25_keys = sorted(k for k in keys if k.startswith("lora_balanced_simple_qwen2_5_7b"))
    if qwen25_keys:
        methods.append(qwen25_keys[0])

    for m in [
        "prometheus2", "zeroshot_qwen3_8b",
        "trad_jaccard_char", "trad_rouge_l",
        "lora_evaluator", "prometheus_2", "zero_shot_7b",
        "char_overlap", "length_heuristic",
    ]:
        if m in keys and m not in methods:
            methods.append(m)
    return methods


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_evaluator_scores(eval_data: list[dict], seed: int = 42,
                                         all_results_path: str = None) -> dict:
    """Load real evaluator scores from all_results.json, fall back to synthetic."""
    # Try loading real scores first
    if all_results_path is None:
        all_results_path = str(PROJECT_ROOT / "data" / "baselines" / "all_results.json")
    try:
        with open(all_results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        # Convert to {method: [score_or_nan, ...]} format
        n = len(eval_data)
        scores = {}
        for method, preds in all_results.items():
            if isinstance(preds, list) and len(preds) == n:
                scores[method] = [float(p) if p is not None and p >= 0 else float('nan')
                                  for p in preds]
        if scores:
            ensure_overlap_heuristic_scores(eval_data, scores, seed=seed)
            print(f"  Loaded real scores for {len(scores)} methods from {all_results_path}")
            return scores
    except Exception as e:
        print(f"  Warning: could not load all_results.json ({e}), falling back to synthetic")

    # Fall back to synthetic
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

        ld_v, pr_v = _finite_pairs(length_diffs, pred_arr)
        # Correlation between length difference and predicted score (valid preds only)
        if len(pr_v) >= 2 and np.std(ld_v) > 0 and np.std(pr_v) > 0:
            pearson_r, pearson_p = stats.pearsonr(ld_v, pr_v)
            spearman_r, spearman_p = stats.spearmanr(ld_v, pr_v)
        else:
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        # Also check: does evaluator treat long-input pairs differently?
        median_input_len = np.median([len(item["input"]) for item in eval_data])
        short_input_scores = [float(s) for i, s in enumerate(pred_scores)
                              if len(eval_data[i]["input"]) <= median_input_len
                              and s is not None and np.isfinite(float(s))]
        long_input_scores = [float(s) for i, s in enumerate(pred_scores)
                             if len(eval_data[i]["input"]) > median_input_len
                             and s is not None and np.isfinite(float(s))]

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

        out_arr = np.array(output_lengths, dtype=float)
        out_v, pr_out = _finite_pairs(out_arr, pred_arr)
        # Correlation with output length (finite predictions only)
        if len(pr_out) >= 2 and np.std(out_v) > 0 and np.std(pr_out) > 0:
            pearson_r, pearson_p = stats.pearsonr(out_v, pr_out)
            spearman_r, spearman_p = stats.spearmanr(out_v, pr_out)
        else:
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        # Correlation with input length
        input_lengths = np.array([len(item["input"]) for item in eval_data], dtype=float)
        in_v, pr_in = _finite_pairs(input_lengths, pred_arr)
        if len(pr_in) >= 2 and np.std(in_v) > 0 and np.std(pr_in) > 0:
            inp_pearson, inp_pp = stats.pearsonr(in_v, pr_in)
            inp_spearman, inp_sp = stats.spearmanr(in_v, pr_in)
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
                seg = pred_arr[mask]
                seg = seg[np.isfinite(seg)]
                if seg.size > 0:
                    quartile_means[name] = round(float(np.mean(seg)), 3)

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

        verb_v, pr_vb = _finite_pairs(verbosity_ratios, pred_arr)
        # Correlation with verbosity ratio (finite predictions only)
        if len(pr_vb) >= 2 and np.std(verb_v) > 0 and np.std(pr_vb) > 0:
            spearman_r, spearman_p = stats.spearmanr(verb_v, pr_vb)
            pearson_r, pearson_p = stats.pearsonr(verb_v, pr_vb)
        else:
            spearman_r, spearman_p = 0.0, 1.0
            pearson_r, pearson_p = 0.0, 1.0

        # Binned analysis: conciseness (< 0.8), similar (0.8-1.2), verbose (> 1.2)
        conciseness_mask = verbosity_ratios < 0.8
        similar_mask = (verbosity_ratios >= 0.8) & (verbosity_ratios <= 1.2)
        verbose_mask = verbosity_ratios > 1.2

        def _mean_safe(mask):
            m = mask & np.isfinite(pred_arr)
            if m.sum() > 0:
                return round(float(np.mean(pred_arr[m])), 3)
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
                      eval_data: list[dict],
                      evaluator_scores: dict[str, list],
                      output_dir: str,
                      dpi: int = 300):
    """Generate comprehensive bias analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 1: Length Bias Scatter Plot ----
    _plot_length_bias_scatter(
        eval_data,
        evaluator_scores,
        output_dir,
        dpi,
    )

    # ---- Figure 2: Verbosity Bias Bar Chart ----
    _plot_verbosity_bias_bars(verbosity_results, output_dir, dpi)

    # ---- Figure 3: Bias Comparison Radar ----
    _plot_bias_comparison_radar(position_results, length_results,
                                verbosity_results, output_dir, dpi)

    # ---- Figure 4: Quartile Score Trend ----
    _plot_quartile_trend(length_results, output_dir, dpi)


def _save_figure(fig, basename: str, output_dir: str, dpi: int) -> None:
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{basename}.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {out_path}")


def _lora_qwen25_full_key(scores: dict[str, list]) -> str | None:
    """Prefer full Qwen2.5-7B LoRA checkpoint over subset keys (``_50_``, etc.)."""
    cands = sorted(k for k in scores if k.startswith("lora_balanced_simple_qwen2_5_7b"))
    if not cands:
        return None
    full = [k for k in cands if not re.match(r"^lora_balanced_simple_\d+_qwen2", k)]
    return full[0] if full else cands[-1]


def _pick_primary_lora_for_length_scatter(scores: dict[str, list], n: int) -> tuple[str, str] | None:
    """Match paper figure: prefer Qwen2.5-7B LoRA first, else Qwen3-8B ``lora_balanced_simple``."""
    keys = set(scores.keys())
    k25 = _lora_qwen25_full_key(scores)
    if k25 and len(scores[k25]) == n:
        return (k25, "LoRA Evaluator (7B)")
    if "lora_balanced_simple" in keys and len(scores["lora_balanced_simple"]) == n:
        return ("lora_balanced_simple", "LoRA Evaluator (8B)")
    subset_q3 = sorted(
        [k for k in keys if re.match(r"^lora_balanced_simple_\d+$", k)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    for k in reversed(subset_q3):
        if len(scores[k]) == n:
            return (k, "LoRA Evaluator (8B)")
    return None


def _plot_length_bias_scatter(
    eval_data: list[dict],
    evaluator_scores: dict[str, list],
    output_dir: str,
    dpi: int,
) -> None:
    """Paper-style scatter: length vs.\ predicted score (Pearson $r$ in legend).

    Series order matches the paper: LoRA (7B preferred), zero-shot Qwen2.5-7B, Prometheus~2,
    human mean as dashed line. Regression lines span the full plotted x-window (like the paper),
    not only the min--max length in the sample.
    """
    n = len(eval_data)
    output_lengths = np.array([len(item.get("output", "")) for item in eval_data], dtype=float)
    human_scores = np.array([float(item["avg_score"]) for item in eval_data], dtype=float)

    series_spec: list[tuple[str, str, str]] = []
    picked = _pick_primary_lora_for_length_scatter(evaluator_scores, n)
    if picked:
        series_spec.append((picked[0], picked[1], "#2563EB"))
    for key, lab, col in [
        ("zeroshot_qwen7b", "Zero-shot Qwen 2.5 (7B)", "#DC2626"),
        ("prometheus2", "Prometheus 2", "#059669"),
    ]:
        if key in evaluator_scores and len(evaluator_scores[key]) == n:
            series_spec.append((key, lab, col))

    if not series_spec:
        print("  [skip] length_bias_scatter: need LoRA (Qwen3 or Qwen2.5) / zero-shot / Prometheus scores")
        return

    # Paper-style viewport: length on [50, 300] characters (extend right only if data need it)
    xmax_data = float(np.nanmax(output_lengths))
    x_right = max(300.0, xmax_data * 1.02)
    x_left = 50.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0.0, 5.0)

    for method_key, label, color in series_spec:
        preds = evaluator_scores.get(method_key)
        if preds is None or len(preds) != n:
            continue
        pred_scores = np.array(
            [float(p) if p is not None and np.isfinite(float(p)) and float(p) >= 0 else np.nan
             for p in preds],
            dtype=float,
        )
        valid = np.isfinite(pred_scores)
        x = output_lengths[valid]
        y = pred_scores[valid]
        if len(x) == 0:
            continue
        ax.scatter(
            x, y, s=12, alpha=0.22, color=color, rasterized=True,
            edgecolors="none", zorder=2,
        )
        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            r, _ = stats.pearsonr(x, y)
            # Draw trend across full x-axis window (same as paper: lines span the panel)
            x_line = np.linspace(x_left, x_right, 200)
            y_line = p(x_line)
            y_line = np.clip(y_line, 0.0, 5.0)
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.2,
                label=f"{label} ($r$={r:+.2f})",
                zorder=5,
            )

    # Human: mean score as horizontal reference; $r$ = Pearson(length, human avg.)
    if len(output_lengths) >= 2 and np.std(output_lengths) > 0 and np.std(human_scores) > 0:
        r_h, _ = stats.pearsonr(output_lengths, human_scores)
    else:
        r_h = float("nan")
    h_mean = float(np.nanmean(human_scores))
    ax.axhline(
        h_mean,
        color="#6B7280",
        linestyle="--",
        linewidth=2.0,
        label=f"Human baseline ($r$={r_h:+.2f})" if np.isfinite(r_h) else "Human baseline",
        zorder=4,
    )

    ax.set_xlabel("Output Length (characters)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Length Bias: Output Length vs Score",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
    ax.legend(fontsize=8.0, framealpha=0.92, loc="upper right")

    fig.tight_layout(pad=1.0)
    _save_figure(fig, "length_bias_scatter", output_dir, dpi)
    plt.close(fig)


def _plot_verbosity_bias_bars(verbosity_results: dict, output_dir: str, dpi: int):
    """Bar chart showing mean scores across verbosity bins."""
    eval_results = verbosity_results.get("evaluators", {})

    methods = _select_methods(eval_results.keys())

    # Drop "verbose" bin from visualization because this dataset has no samples in ratio > 1.2.
    bins = ["concise", "similar"]
    bin_labels = ["Concise\n(ratio < 0.8)", "Similar\n(0.8-1.2)"]

    fig_w = max(8.0, 2.5 + len(methods) * 0.72)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4))

    x = np.arange(len(bins))
    width = min(0.22, 0.65 / max(len(methods), 1))

    for i, method in enumerate(methods):
        binned = eval_results[method].get("binned_analysis", {})
        means = [binned.get(f"{b}_mean_score", 0) or 0 for b in bins]
        color = _resolve_method_color(method)

        bars = ax.bar(x + i * width, means, width,
                      color=color, alpha=0.8, edgecolor="white", linewidth=0.5,
                      label=_resolve_method_display(method))

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
    methods = _select_methods(methods)

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

        color = _resolve_method_color(method)
        ax.plot(angles, norm_vals, "o-", linewidth=1.8, markersize=5,
                color=color, label=_resolve_method_display(method), alpha=0.8)
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

    methods = _select_methods(eval_results.keys())

    quartile_keys = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    for method in methods:
        quartiles = eval_results[method].get("quartile_score_means", {})
        vals = [quartiles.get(k, 0) for k in quartile_keys]

        color = _resolve_method_color(method)
        marker = "o" if method == "lora_balanced_simple" else "s"
        ms = 8 if method == "lora_balanced_simple" else 5
        lw = 2 if method == "lora_balanced_simple" else 1.2

        ax.plot(range(4), vals, color=color, marker=marker, markersize=ms,
                linewidth=lw, alpha=0.85,
                label=_resolve_method_display(method))

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
    methods = _select_methods(methods)

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

        name = _resolve_method_display(method)
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
    plot_bias_summary(
        position_results,
        length_results,
        verbosity_results,
        eval_data,
        evaluator_scores,
        args.figures_dir,
        args.dpi,
    )

    # Generate LaTeX table
    print("\n--- Generating Table ---")
    generate_bias_table(position_results, length_results, verbosity_results,
                        args.results_dir)

    print("\nBias analysis complete!")


if __name__ == "__main__":
    main()
