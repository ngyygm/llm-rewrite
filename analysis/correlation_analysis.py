#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Analysis for EMNLP 2026 Chinese Rewriting Evaluation.

Computes correlation metrics between evaluator predictions and human annotations:
- Spearman rho, Pearson r, Kendall tau
- MAE, RMSE
- Agreement: Exact %, +/-1 %, +/-2 %
- Per-score-level analysis
- Confusion matrix data

Usage:
    python correlation_analysis.py [--eval-data PATH] [--results PATH] [--output-dir DIR]
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATA = PROJECT_ROOT / "data" / "human_eval" / "eval.json"
DEFAULT_ALL_RESULTS = PROJECT_ROOT / "data" / "baselines" / "all_results.json"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "baselines" / "method_metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results"


def compute_rank_correlations(human_scores, pred_scores):
    h = np.array(human_scores, dtype=float)
    p = np.array(pred_scores, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(h))
    if valid.sum() < 2:
        return {"spearman_rho": 0.0, "spearman_p": 1.0,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "kendall_tau": 0.0, "kendall_p": 1.0}
    hv, pv = h[valid], p[valid]
    sp, spp = stats.spearmanr(hv, pv)
    pr, prp = stats.pearsonr(hv, pv)
    kt, ktp = stats.kendalltau(hv, pv)
    return {
        "spearman_rho": round(float(sp), 4), "spearman_p": round(float(spp), 6),
        "pearson_r": round(float(pr), 4), "pearson_p": round(float(prp), 6),
        "kendall_tau": round(float(kt), 4), "kendall_p": round(float(ktp), 6),
    }


def compute_error_metrics(human_scores, pred_scores):
    h = np.array(human_scores, dtype=float)
    p = np.array(pred_scores, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(h))
    hv, pv = h[valid], p[valid]
    n = int(valid.sum())
    if n < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "n_samples": 0,
                "exact_agreement_pct": 0, "within_1_pct": 0, "within_2_pct": 0}
    mae = float(np.mean(np.abs(hv - pv)))
    rmse = float(np.sqrt(np.mean((hv - pv) ** 2)))
    exact = float(np.mean(hv == pv)) * 100
    within_1 = float(np.mean(np.abs(hv - pv) <= 1)) * 100
    within_2 = float(np.mean(np.abs(hv - pv) <= 2)) * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "n_samples": n,
            "exact_agreement_pct": round(exact, 2),
            "within_1_pct": round(within_1, 2),
            "within_2_pct": round(within_2, 2)}


def build_confusion_matrix(human_scores, pred_scores, labels=None):
    if labels is None:
        all_vals = sorted(set(int(s) for s in human_scores if not np.isnan(s)) |
                          set(int(s) for s in pred_scores if not np.isnan(s) and s >= 0))
        labels = all_vals

    matrix = defaultdict(lambda: defaultdict(int))
    for h, p in zip(human_scores, pred_scores):
        if np.isnan(h) or np.isnan(p) or p < 0:
            continue
        h_int, p_int = int(round(h)), int(round(p))
        if h_int in labels and p_int in labels:
            matrix[h_int][p_int] += 1

    matrix_pct = {}
    for h_int in labels:
        row_total = sum(matrix[h_int].get(p, 0) for p in labels)
        matrix_pct[h_int] = {}
        for p_int in labels:
            count = matrix[h_int].get(p_int, 0)
            matrix_pct[h_int][p_int] = round(count / row_total * 100, 1) if row_total > 0 else 0.0

    return {
        "labels": labels,
        "counts": {str(k): dict(v) for k, v in matrix.items()},
        "percentages": {str(k): v for k, v in matrix_pct.items()},
    }


def per_score_level_analysis(human_scores, pred_scores):
    h = np.array(human_scores, dtype=float)
    p = np.array(pred_scores, dtype=float)
    all_levels = sorted(set(int(s) for s in human_scores if not np.isnan(s)))
    analysis = {}
    for level in all_levels:
        mask = (np.abs(h - level) < 0.5) & ~np.isnan(p)
        pred_at_level = p[mask]
        n = int(mask.sum())
        pred_counter = Counter(int(round(x)) for x in pred_at_level if x >= 0)
        analysis[str(level)] = {
            "n_samples": n,
            "pred_mean": round(float(np.mean(pred_at_level[pred_at_level >= 0])), 3) if n > 0 else None,
            "pred_std": round(float(np.std(pred_at_level[pred_at_level >= 0])), 3) if n > 0 else None,
            "pred_distribution": dict(sorted(pred_counter.items())),
        }
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument("--eval-data", type=str, default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--all-results", type=str, default=str(DEFAULT_ALL_RESULTS))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    # Load data
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(args.all_results, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n = len(eval_data)
    print(f"Loaded {n} eval samples, {len(all_results)} methods")

    human_consensus = [item["consensus_score"] for item in eval_data]
    human_avg = [item["avg_score"] for item in eval_data]

    # Compute metrics for each method
    method_results = {}
    for method_name, scores in all_results.items():
        preds = np.array([s if s >= 0 else np.nan for s in scores], dtype=float)
        refs_consensus = np.array(human_consensus, dtype=float)
        refs_avg = np.array(human_avg, dtype=float)

        corr = compute_rank_correlations(refs_avg, preds)
        err = compute_error_metrics(refs_consensus, preds)
        confusion = build_confusion_matrix(refs_consensus, preds)
        per_score = per_score_level_analysis(refs_consensus, preds)

        method_results[method_name] = {
            "correlations": corr,
            "error_metrics": err,
            "confusion_matrix": confusion,
            "per_score_analysis": per_score,
        }

        print(f"\n--- {metadata.get(method_name, {}).get('display_name', method_name)} ---")
        print(f"  Spearman={corr['spearman_rho']:+.4f}  Pearson={corr['pearson_r']:+.4f}  Kendall={corr['kendall_tau']:+.4f}")
        print(f"  MAE={err['mae']:.4f}  RMSE={err['rmse']:.4f}")
        print(f"  Exact={err['exact_agreement_pct']:.1f}%  +/-1={err['within_1_pct']:.1f}%  +/-2={err['within_2_pct']:.1f}%")

    # Save
    output = {
        "metadata": {
            "n_eval_samples": n,
            "methods": list(all_results.keys()),
        },
        "human_score_distribution": dict(Counter(human_consensus)),
        "results": method_results,
        "method_metadata": metadata,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "correlation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
