"""
Master runner script for all baseline evaluations.

Runs all baseline methods and collects combined results:
1. Traditional metrics (BLEU, ROUGE-L, Jaccard, TF-IDF, SBERT, W2V)
2. LLM-based evaluators (G-Eval, Zero-shot Qwen2.5-7B)
3. Fine-tuned evaluators (Prometheus 2, M-Prometheus)
4. ParaScore

Saves combined results to data/baselines/all_results.json and prints
a summary comparison table.

Usage:
    # Run everything (warning: LLM evaluators take hours)
    python baselines/run_all_baselines.py

    # Run only traditional metrics (fast)
    python baselines/run_all_baselines.py --traditional-only

    # Run traditional + ParaScore (no LLM, moderate speed)
    python baselines/run_all_baselines.py --no-llm

    # Run with a specific LLM model
    python baselines/run_all_baselines.py --llm-model Qwen/Qwen2.5-7B-Instruct

    # Quick debug run (5 samples only)
    python baselines/run_all_baselines.py --max-samples 5
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from baselines.correlation_utils import (
    load_eval_data,
    print_correlation_table,
    save_results,
    combine_results,
)

EVAL_PATH = BASE_DIR / "data" / "human_eval" / "eval.json"
RESULTS_DIR = BASE_DIR / "data" / "baselines"


# ============================================================
# Individual runners
# ============================================================

def run_traditional(data: List[Dict], skip_embedding: bool = False, max_samples: Optional[int] = None) -> Dict:
    """Run traditional metrics."""
    from baselines.run_traditional import run_all_traditional_metrics

    eval_data = data[:max_samples] if max_samples else data
    return run_all_traditional_metrics(eval_data, skip_embedding=skip_embedding)


def run_llm_evaluators(
    data: List[Dict],
    model_name: str,
    eval_types: List[str],
    temperature: float,
    load_in_4bit: bool,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Run LLM-based evaluators."""
    from baselines.run_llm_evaluators import run_llm_evaluator

    results = []
    for eval_type in eval_types:
        result = run_llm_evaluator(
            model_name=model_name,
            data=data,
            eval_type=eval_type,
            temperature=temperature,
            load_in_4bit=load_in_4bit,
            max_samples=max_samples,
        )
        results.append(result)
    return results


def run_fine_tuned(
    data: List[Dict],
    temperature: float,
    load_in_4bit: bool,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Run fine-tuned evaluators."""
    from baselines.run_fine_tuned_evaluators import run_all_fine_tuned

    return run_all_fine_tuned(
        data,
        temperature=temperature,
        load_in_4bit=load_in_4bit,
        max_samples=max_samples,
    )


def run_parascore(data: List[Dict], max_samples: Optional[int] = None) -> Dict:
    """Run ParaScore evaluation."""
    from baselines.run_parascore import run_parascore_evaluation

    eval_data = data[:max_samples] if max_samples else data
    return run_parascore_evaluation(eval_data)


# ============================================================
# Summary table
# ============================================================

def print_summary_table(all_results: Dict) -> str:
    """Print a formatted summary table of all baseline results.

    Args:
        all_results: Combined results dict.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append("BASELINE EVALUATION SUMMARY")
    lines.append("=" * 120)
    lines.append("")
    lines.append(
        f"{'Method':<40s}  {'Spearman':>9s}  {'p-value':>9s}  "
        f"{'Kendall':>9s}  {'Pearson':>9s}  "
        f"{'MAE':>6s}  {'RMSE':>6s}  "
        f"{'Exact%':>7s}  {'+/-1%':>6s}  {'+/-2%':>6s}"
    )
    lines.append("-" * 120)

    # Collect all correlation results
    method_correlations = []

    # Traditional metrics
    if "traditional" in all_results:
        for metric_name, corr in all_results["traditional"].get("correlations", {}).items():
            method_correlations.append((metric_name, corr))

    # LLM evaluators
    if "llm_evaluators" in all_results:
        for r in all_results["llm_evaluators"]:
            if "correlations" in r:
                corr = r["correlations"]
                method_correlations.append((corr.get("method", "unknown"), corr))

    # Fine-tuned evaluators
    if "fine_tuned" in all_results:
        for r in all_results["fine_tuned"]:
            if "correlations" in r:
                corr = r["correlations"]
                method_correlations.append((corr.get("method", "unknown"), corr))

    # ParaScore
    if "parascore" in all_results:
        for mode_name, corr in all_results["parascore"].get("correlations", {}).items():
            method_correlations.append((mode_name, corr))

    # Sort by Spearman rho (descending)
    method_correlations.sort(key=lambda x: x[1].get("spearman_rho", 0), reverse=True)

    for method_name, corr in method_correlations:
        rho = corr.get("spearman_rho", 0)
        p_val = corr.get("spearman_p", 1)
        tau = corr.get("kendall_tau", 0)
        pearson = corr.get("pearson_r", 0)
        mae = corr.get("mae", 0)
        rmse = corr.get("rmse", 0)
        exact = corr.get("exact_agreement_pct", 0)
        w1 = corr.get("within_1_pct", 0)
        w2 = corr.get("within_2_pct", 0)

        # Significance markers
        p_sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "   "

        lines.append(
            f"{method_name:<40s}  {rho:>+8.4f}{p_sig:<4s} {p_val:>8.4f}  "
            f"{tau:>+8.4f}  {pearson:>+8.4f}  "
            f"{mae:>5.3f}  {rmse:>5.3f}  "
            f"{exact:>6.1f}%  {w1:>5.1f}%  {w2:>5.1f}%"
        )

    lines.append("=" * 120)

    # Legend
    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    lines.append("Primary metric: Spearman rho (higher is better)")
    lines.append("")

    return "\n".join(lines)


def print_best_methods(all_results: Dict) -> str:
    """Print the best method per category."""
    lines = []
    lines.append("")
    lines.append("BEST METHODS BY CATEGORY:")
    lines.append("-" * 60)

    # Traditional
    if "traditional" in all_results:
        trad_corrs = all_results["traditional"].get("correlations", {})
        if trad_corrs:
            best_trad = max(trad_corrs.items(), key=lambda x: x[1].get("spearman_rho", 0))
            lines.append(
                f"  Traditional:      {best_trad[0]} "
                f"(rho={best_trad[1].get('spearman_rho', 0):.4f})"
            )

    # LLM
    if "llm_evaluators" in all_results:
        llm_corrs = [
            (r["correlations"].get("method", ""), r["correlations"])
            for r in all_results["llm_evaluators"]
            if "correlations" in r
        ]
        if llm_corrs:
            best_llm = max(llm_corrs, key=lambda x: x[1].get("spearman_rho", 0))
            lines.append(
                f"  LLM-based:        {best_llm[0]} "
                f"(rho={best_llm[1].get('spearman_rho', 0):.4f})"
            )

    # Fine-tuned
    if "fine_tuned" in all_results:
        ft_corrs = [
            (r["correlations"].get("method", ""), r["correlations"])
            for r in all_results["fine_tuned"]
            if "correlations" in r
        ]
        if ft_corrs:
            best_ft = max(ft_corrs, key=lambda x: x[1].get("spearman_rho", 0))
            lines.append(
                f"  Fine-tuned:       {best_ft[0]} "
                f"(rho={best_ft[1].get('spearman_rho', 0):.4f})"
            )

    # ParaScore
    if "parascore" in all_results:
        ps_corrs = all_results["parascore"].get("correlations", {})
        if ps_corrs:
            best_ps = max(ps_corrs.items(), key=lambda x: x[1].get("spearman_rho", 0))
            lines.append(
                f"  ParaScore:        {best_ps[0]} "
                f"(rho={best_ps[1].get('spearman_rho', 0):.4f})"
            )

    # Overall best
    all_methods = []
    for category, key in [
        ("traditional", "traditional"),
        ("llm", "llm_evaluators"),
        ("finetuned", "fine_tuned"),
        ("parascore", "parascore"),
    ]:
        if key in all_results:
            corrs = all_results[key].get("correlations", {})
            if isinstance(corrs, dict):
                for name, corr in corrs.items():
                    if isinstance(corr, dict):
                        all_methods.append((f"[{category}] {name}", corr))
            elif isinstance(corrs, list):
                for r in corrs:
                    if isinstance(r, dict) and "correlations" in r:
                        c = r["correlations"]
                        if isinstance(c, dict):
                            all_methods.append(
                                (f"[{category}] {c.get('method', '')}", c)
                            )

    if all_methods:
        best_overall = max(
            all_methods, key=lambda x: x[1].get("spearman_rho", 0)
        )
        lines.append("")
        lines.append(f"  OVERALL BEST:     {best_overall[0]} "
                      f"(rho={best_overall[1].get('spearman_rho', 0):.4f})")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    """Main entry point for running all baselines."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all baseline evaluations and produce summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines
  python baselines/run_all_baselines.py

  # Run only traditional metrics (fast, no GPU needed for most)
  python baselines/run_all_baselines.py --traditional-only

  # Run traditional + ParaScore (no LLMs)
  python baselines/run_all_baselines.py --no-llm

  # Debug with 5 samples
  python baselines/run_all_baselines.py --max-samples 5 --traditional-only
        """
    )
    parser.add_argument(
        "--eval-path", type=str, default=str(EVAL_PATH),
        help="Path to eval.json"
    )
    parser.add_argument(
        "--traditional-only", action="store_true",
        help="Only run traditional metrics (fast)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM-based evaluators (run traditional + ParaScore)"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="Skip embedding-based traditional metrics (BLEU/ROUGE/Jaccard/TF-IDF only)"
    )
    parser.add_argument(
        "--skip-fine-tuned", action="store_true",
        help="Skip fine-tuned evaluators (Prometheus)"
    )
    parser.add_argument(
        "--skip-parascore", action="store_true",
        help="Skip ParaScore"
    )
    parser.add_argument(
        "--llm-model", type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for LLM-based evaluators"
    )
    parser.add_argument(
        "--llm-eval-types", nargs="+",
        default=["zero_shot", "geval"],
        choices=["zero_shot", "geval"],
        help="LLM evaluation types to run"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization for LLM models"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to N samples (for debugging)"
    )
    args = parser.parse_args()

    print("=" * 120)
    print("EMNLP 2026 - Chinese Rewriting Evaluation: All Baselines")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 120)

    # Load data
    data = load_eval_data(args.eval_path)
    n_eval = len(data)
    if args.max_samples:
        data_eval = data[:args.max_samples]
        n_eval = args.max_samples
    else:
        data_eval = data

    print(f"\nEvaluation data: {n_eval} samples")
    print(f"Ground truth: consensus_score (0-5, integer)")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_eval,
        "config": {
            "eval_path": str(args.eval_path),
            "skip_embedding": args.skip_embedding,
            "llm_model": args.llm_model,
            "llm_eval_types": args.llm_eval_types,
            "temperature": args.temperature,
            "load_in_4bit": not args.no_4bit,
            "max_samples": args.max_samples,
        },
    }

    total_start = time.time()

    # ============================================================
    # 1. Traditional metrics
    # ============================================================
    print("\n" + "#" * 60)
    print("# 1. Traditional Metrics")
    print("#" * 60)

    try:
        t0 = time.time()
        trad_results = run_traditional(
            data_eval, skip_embedding=args.skip_embedding
        )
        all_results["traditional"] = {
            "correlations": trad_results.get("correlations", {}),
            "level_analysis": trad_results.get("level_analysis", {}),
            "metrics_run": trad_results.get("metrics_run", []),
            "time_seconds": round(time.time() - t0, 1),
        }
        print(f"  Traditional metrics completed in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  ERROR running traditional metrics: {e}")
        all_results["traditional"] = {"error": str(e)}

    # ============================================================
    # 2. LLM-based evaluators
    # ============================================================
    if not args.traditional_only and not args.no_llm:
        print("\n" + "#" * 60)
        print("# 2. LLM-based Evaluators")
        print("#" * 60)

        try:
            t0 = time.time()
            llm_results = run_llm_evaluators(
                data=data_eval,
                model_name=args.llm_model,
                eval_types=args.llm_eval_types,
                temperature=args.temperature,
                load_in_4bit=not args.no_4bit,
                max_samples=args.max_samples,
            )
            all_results["llm_evaluators"] = llm_results
            print(f"  LLM evaluators completed in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR running LLM evaluators: {e}")
            all_results["llm_evaluators"] = {"error": str(e)}
    elif args.no_llm:
        print("\n  Skipping LLM evaluators (--no-llm)")

    # ============================================================
    # 3. Fine-tuned evaluators
    # ============================================================
    if not args.traditional_only and not args.no_llm and not args.skip_fine_tuned:
        print("\n" + "#" * 60)
        print("# 3. Fine-tuned Evaluators")
        print("#" * 60)

        try:
            t0 = time.time()
            ft_results = run_fine_tuned(
                data=data_eval,
                temperature=args.temperature,
                load_in_4bit=not args.no_4bit,
                max_samples=args.max_samples,
            )
            all_results["fine_tuned"] = ft_results
            print(f"  Fine-tuned evaluators completed in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR running fine-tuned evaluators: {e}")
            all_results["fine_tuned"] = {"error": str(e)}
    elif args.skip_fine_tuned:
        print("\n  Skipping fine-tuned evaluators (--skip-fine-tuned)")

    # ============================================================
    # 4. ParaScore
    # ============================================================
    if not args.traditional_only and not args.skip_parascore:
        print("\n" + "#" * 60)
        print("# 4. ParaScore")
        print("#" * 60)

        try:
            t0 = time.time()
            ps_results = run_parascore(data_eval)
            all_results["parascore"] = {
                "correlations": ps_results.get("correlations", {}),
                "level_analysis": ps_results.get("level_analysis", {}),
                "modes": ps_results.get("modes", []),
                "time_seconds": round(time.time() - t0, 1),
            }
            print(f"  ParaScore completed in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR running ParaScore: {e}")
            all_results["parascore"] = {"error": str(e)}
    elif args.skip_parascore:
        print("\n  Skipping ParaScore (--skip-parascore)")

    total_time = time.time() - total_start

    # ============================================================
    # Summary
    # ============================================================
    summary = print_summary_table(all_results)
    print(summary)

    best = print_best_methods(all_results)
    print(best)

    print(f"Total runtime: {total_time:.1f}s")

    # ============================================================
    # Save combined results
    # ============================================================
    # Remove sample_results to keep the combined file manageable
    combined_save = json.loads(json.dumps(all_results, default=str))
    # Deep clean sample_results
    for key in ["traditional", "parascore"]:
        if key in combined_save and isinstance(combined_save[key], dict):
            combined_save[key].pop("sample_results", None)
    for key in ["llm_evaluators", "fine_tuned"]:
        if key in combined_save and isinstance(combined_save[key], list):
            for item in combined_save[key]:
                if isinstance(item, dict):
                    item.pop("sample_results", None)
                    item.pop("raw_responses", None)

    out_path = save_results(combined_save, "all_results.json")
    print(f"Combined results saved to: {out_path}")

    # Also save the summary table as text
    summary_path = RESULTS_DIR / "summary_table.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"EMNLP 2026 - Chinese Rewriting Evaluation: Baseline Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {n_eval}\n\n")
        f.write(summary)
        f.write(best)
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
