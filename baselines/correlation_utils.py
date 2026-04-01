"""
Shared correlation analysis utilities for baseline evaluation.

Provides comprehensive correlation and agreement metrics between
predicted scores and human annotations for the EMNLP 2026 Chinese
rewriting evaluation project.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "data" / "human_eval" / "eval.json"
RESULTS_DIR = BASE_DIR / "data" / "baselines"


def load_eval_data(path: Optional[str] = None) -> List[Dict]:
    """Load the human evaluation dataset.

    Args:
        path: Path to eval.json. Defaults to data/human_eval/eval.json.

    Returns:
        List of dicts with keys: input, output, annotator_scores,
        consensus_score, avg_score, std_score.
    """
    if path is None:
        path = str(EVAL_PATH)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_correlations(
    predicted_scores: List[float],
    ground_truth_scores: List[float],
    method_name: str = "method",
    round_predictions: bool = False,
    score_range: Tuple[int, int] = (0, 5),
) -> Dict:
    """Compute all correlation and agreement metrics.

    Args:
        predicted_scores: List of predicted scores from the evaluator.
        ground_truth_scores: List of human consensus scores (integers 0-5).
        method_name: Name of the evaluation method for reporting.
        round_predictions: If True, round predictions to integers before
            computing agreement metrics.
        score_range: (min, max) possible score values for agreement metrics.

    Returns:
        Dict containing all metrics.
    """
    predicted = np.array(predicted_scores, dtype=float)
    ground_truth = np.array(ground_truth_scores, dtype=float)

    assert len(predicted) == len(ground_truth), (
        f"Length mismatch: predicted={len(predicted)}, "
        f"ground_truth={len(ground_truth)}"
    )

    # Filter out any NaN values
    valid_mask = ~(np.isnan(predicted) | np.isnan(ground_truth))
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]

    if len(pred_valid) < 5:
        return {
            "method": method_name,
            "n_valid": len(pred_valid),
            "error": "Too few valid samples for correlation",
        }

    results = {"method": method_name, "n_valid": int(valid_mask.sum()), "n_total": len(predicted)}

    # --- Rank-based correlations ---

    # Spearman rho (primary metric)
    if np.std(pred_valid) == 0 or np.std(gt_valid) == 0:
        results["spearman_rho"] = 0.0
        results["spearman_p"] = 1.0
    else:
        rho, p_value = stats.spearmanr(pred_valid, gt_valid)
        results["spearman_rho"] = float(rho)
        results["spearman_p"] = float(p_value)

    # Kendall tau
    try:
        tau, tau_p = stats.kendalltau(pred_valid, gt_valid)
        results["kendall_tau"] = float(tau)
        results["kendall_p"] = float(tau_p)
    except Exception:
        results["kendall_tau"] = 0.0
        results["kendall_p"] = 1.0

    # --- Linear correlation ---

    # Pearson r
    if np.std(pred_valid) == 0 or np.std(gt_valid) == 0:
        results["pearson_r"] = 0.0
        results["pearson_p"] = 1.0
    else:
        r, r_p = stats.pearsonr(pred_valid, gt_valid)
        results["pearson_r"] = float(r)
        results["pearson_p"] = float(r_p)

    # --- Error metrics ---
    results["mae"] = float(np.mean(np.abs(pred_valid - gt_valid)))
    results["rmse"] = float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))

    # --- Agreement metrics ---
    if round_predictions:
        pred_rounded = np.round(pred_valid).astype(int)
        gt_int = gt_valid.astype(int)
    else:
        pred_rounded = pred_valid
        gt_int = gt_valid

    diff = np.abs(pred_rounded - gt_int)
    results["exact_agreement_pct"] = float(np.mean(diff == 0) * 100)
    results["within_1_pct"] = float(np.mean(diff <= 1) * 100)
    results["within_2_pct"] = float(np.mean(diff <= 2) * 100)

    # --- Predicted score statistics ---
    results["pred_mean"] = float(np.mean(pred_valid))
    results["pred_std"] = float(np.std(pred_valid))
    results["pred_min"] = float(np.min(pred_valid))
    results["pred_max"] = float(np.max(pred_valid))

    return results


def per_score_level_analysis(
    predicted_scores: List[float],
    ground_truth_scores: List[float],
    method_name: str = "method",
) -> Dict:
    """Compute per-score-level (0-5) breakdown of predictions.

    For each true score level, compute the mean and std of predicted scores,
    count of samples, and accuracy metrics.

    Args:
        predicted_scores: List of predicted scores.
        ground_truth_scores: List of human consensus scores (integers 0-5).
        method_name: Name of the evaluation method.

    Returns:
        Dict keyed by score level with prediction statistics.
    """
    predicted = np.array(predicted_scores, dtype=float)
    ground_truth = np.array(ground_truth_scores, dtype=int)

    level_analysis = {}
    for score_val in range(6):
        mask = ground_truth == score_val
        if mask.sum() == 0:
            level_analysis[str(score_val)] = {
                "count": 0,
                "pred_mean": None,
                "pred_std": None,
                "exact_agreement_pct": None,
            }
            continue

        preds_at_level = predicted[mask]
        rounded_preds = np.round(preds_at_level).astype(int)
        exact_match = (rounded_preds == score_val).sum()

        level_analysis[str(score_val)] = {
            "count": int(mask.sum()),
            "pred_mean": float(np.mean(preds_at_level)),
            "pred_std": float(np.std(preds_at_level)),
            "pred_median": float(np.median(preds_at_level)),
            "exact_agreement_pct": float(exact_match / mask.sum() * 100),
            "within_1_pct": float(
                (np.abs(rounded_preds - score_val) <= 1).sum() / mask.sum() * 100
            ),
        }

    level_analysis["_method"] = method_name
    return level_analysis


def print_correlation_table(results: Dict) -> str:
    """Format correlation results as a readable table row.

    Args:
        results: Dict returned by compute_correlations.

    Returns:
        Formatted string for printing.
    """
    name = results.get("method", "unknown")
    n = results.get("n_valid", 0)
    rho = results.get("spearman_rho", 0)
    rho_p = results.get("spearman_p", 1)
    tau = results.get("kendall_tau", 0)
    pearson = results.get("pearson_r", 0)
    mae = results.get("mae", float("inf"))
    rmse = results.get("rmse", float("inf"))
    exact = results.get("exact_agreement_pct", 0)
    w1 = results.get("within_1_pct", 0)

    p_sig = "***" if rho_p < 0.001 else "**" if rho_p < 0.01 else "*" if rho_p < 0.05 else ""

    line = (
        f"  {name:<30s}  n={n:>3d}  "
        f"Spearman={rho:+.4f}{p_sig:<3s}  "
        f"Kendall={tau:+.4f}  "
        f"Pearson={pearson:+.4f}  "
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"Exact={exact:.1f}%  +/-1={w1:.1f}%"
    )
    return line


def print_level_analysis(level_analysis: Dict) -> str:
    """Format per-score-level analysis as a readable table.

    Args:
        level_analysis: Dict returned by per_score_level_analysis.

    Returns:
        Formatted string for printing.
    """
    method = level_analysis.pop("_method", "method")
    lines = [f"\n  Per-score-level analysis for {method}:"]
    lines.append(f"  {'Score':>5s}  {'Count':>5s}  {'Pred Mean':>9s}  "
                 f"{'Pred Std':>8s}  {'Exact%':>7s}  {'Within1%':>8s}")
    lines.append("  " + "-" * 50)
    for score_val in range(6):
        info = level_analysis.get(str(score_val), {})
        if info.get("count", 0) == 0:
            lines.append(f"  {score_val:>5d}  {'N/A':>5s}")
        else:
            lines.append(
                f"  {score_val:>5d}  {info['count']:>5d}  "
                f"{info['pred_mean']:>9.3f}  {info['pred_std']:>8.3f}  "
                f"{info['exact_agreement_pct']:>6.1f}%  "
                f"{info['within_1_pct']:>7.1f}%"
            )
    return "\n".join(lines)


def save_results(
    results: Dict,
    filename: str,
    subdir: Optional[str] = None,
) -> str:
    """Save results dict to a JSON file.

    Args:
        results: Results dictionary to save.
        filename: Output filename (e.g., "traditional_metrics.json").
        subdir: Optional subdirectory under RESULTS_DIR.

    Returns:
        Path to the saved file.
    """
    out_dir = RESULTS_DIR
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return str(out_path)


def load_results(
    filename: str,
    subdir: Optional[str] = None,
) -> Dict:
    """Load previously saved results.

    Args:
        filename: Results filename.
        subdir: Optional subdirectory under RESULTS_DIR.

    Returns:
        Loaded results dictionary.
    """
    in_dir = RESULTS_DIR
    if subdir:
        in_dir = in_dir / subdir
    in_path = in_dir / filename
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_results(
    result_dicts: List[Dict],
    group_name: str = "combined",
) -> Dict:
    """Combine multiple result dicts into one summary.

    Args:
        result_dicts: List of individual result dicts from compute_correlations.
        group_name: Name for the combined group.

    Returns:
        Combined results dict.
    """
    combined = {"group": group_name, "methods": {}}
    for r in result_dicts:
        name = r.get("method", "unknown")
        combined["methods"][name] = r

    # Summary statistics across methods
    if result_dicts:
        rhos = [r.get("spearman_rho", 0) for r in result_dicts if "spearman_rho" in r]
        if rhos:
            combined["summary"] = {
                "best_spearman": max(rhos),
                "best_method": max(
                    zip(rhos, [r.get("method", "") for r in result_dicts]),
                    key=lambda x: x[0]
                )[1],
                "mean_spearman": float(np.mean(rhos)),
                "n_methods": len(rhos),
            }

    return combined


if __name__ == "__main__":
    # Quick self-test with synthetic data
    data = load_eval_data()
    gt_scores = [d["consensus_score"] for d in data]
    n = len(gt_scores)

    # Simulate a decent predictor (human-like with noise)
    rng = np.random.RandomState(42)
    pred_scores = np.array(gt_scores, dtype=float) + rng.normal(0, 0.5, n)
    pred_scores = np.clip(pred_scores, 0, 5).tolist()

    results = compute_correlations(pred_scores, gt_scores, "synthetic_test")
    print(print_correlation_table(results))

    level = per_score_level_analysis(pred_scores, gt_scores, "synthetic_test")
    print(print_level_analysis(level))

    print(f"\nEval data loaded: {n} samples")
    print(f"Ground truth score distribution:")
    for s in range(6):
        count = sum(1 for d in data if d["consensus_score"] == s)
        print(f"  {s}: {count} ({count/n*100:.1f}%)")
