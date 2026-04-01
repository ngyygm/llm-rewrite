#!/usr/bin/env python3
"""
End-to-end downstream validation pipeline.

Simplified approach:
1. Generate source texts & rewrites (or use existing)
2. Score all rewrites with LoRA evaluator
3. Create filtered datasets with different strategies
4. For each strategy, compute quality metrics on the filtered set

Since full SFT training is expensive, this script focuses on:
- Evaluator-guided data quality analysis
- Comparison of filtering strategies via score distributions
- Statistical validation of evaluator rankings

EMNLP 2026
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated_rewrites"
FILTERED_DIR = GENERATED_DIR / "filtered"


def analyze_score_distribution(scored_rewrites, output_dir):
    """Analyze score distribution of evaluator-scored rewrites."""
    valid_scores = [r["predicted_score"] for r in scored_rewrites if r["predicted_score"] is not None]
    if not valid_scores:
        print("No valid scores to analyze")
        return

    dist = Counter(valid_scores)
    print(f"\nScore Distribution ({len(valid_scores)} valid):")
    print(f"  Mean: {np.mean(valid_scores):.2f}")
    print(f"  Std:  {np.std(valid_scores):.2f}")
    print(f"  Min:  {min(valid_scores)}, Max: {max(valid_scores)}")
    for s in range(6):
        count = dist.get(s, 0)
        pct = count / len(valid_scores) * 100
        bar = "#" * int(pct / 2)
        print(f"  Score {s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Analyze by prompt type
    by_prompt = {}
    for r in scored_rewrites:
        pt = r.get("prompt_type", "unknown")
        score = r["predicted_score"]
        if score is not None:
            if pt not in by_prompt:
                by_prompt[pt] = []
            by_prompt[pt].append(score)

    print(f"\nScore by prompt type:")
    for pt, scores in sorted(by_prompt.items()):
        print(f"  Prompt {pt}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, n={len(scores)}")

    # Kendall W (concordance) between prompt types for same source
    from scipy import stats
    sources = {}
    for r in scored_rewrites:
        sh = r.get("source_hash", "")
        if sh not in sources:
            sources[sh] = {}
        pt = r.get("prompt_type", -1)
        score = r["predicted_score"]
        if score is not None:
            sources[sh][pt] = score

    multi_prompt_sources = {sh: scores for sh, scores in sources.items() if len(scores) >= 2}
    if len(multi_prompt_sources) >= 10:
        print(f"\nEvaluator discrimination (sources with 2+ rewrites): {len(multi_prompt_sources)}")
        # Check if evaluator gives different scores to different rewrites of same source
        discrim_counts = []
        for sh, scores in multi_prompt_sources.items():
            score_vals = list(scores.values())
            if len(score_vals) >= 2:
                range_val = max(score_vals) - min(score_vals)
                discrim_counts.append(range_val)
        if discrim_counts:
            print(f"  Score range per source: mean={np.mean(discrim_counts):.2f}, "
                  f"median={np.median(discrim_counts):.1f}")
            same_score_pct = sum(1 for d in discrim_counts if d == 0) / len(discrim_counts) * 100
            print(f"  Same score for all variants: {same_score_pct:.1f}%")

    results = {
        "n_total": len(scored_rewrites),
        "n_valid": len(valid_scores),
        "mean_score": round(float(np.mean(valid_scores)), 4),
        "std_score": round(float(np.std(valid_scores)), 4),
        "distribution": {str(k): v for k, v in dist.items()},
        "by_prompt_type": {str(k): {"mean": round(float(np.mean(v)), 4),
                                      "std": round(float(np.std(v)), 4),
                                      "n": len(v)}
                             for k, v in by_prompt.items()},
    }

    out_path = output_dir / "score_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nScore analysis saved to: {out_path}")
    return results


def create_filtered_datasets(scored_rewrites, output_dir):
    """Create filtered datasets with different strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import random

    valid = [r for r in scored_rewrites if r["predicted_score"] is not None]
    n_total = len(valid)
    print(f"\nCreating filtered datasets from {n_total} valid rewrites...")

    strategies = {}

    # 1. All data (no filtering)
    strategies["all"] = valid

    # 2. Random sample (50%)
    rng = random.Random(42)
    strategies["random_50pct"] = rng.sample(valid, n_total // 2)

    # 3. Evaluator top-K (top 50%)
    sorted_by_score = sorted(valid, key=lambda x: x["predicted_score"], reverse=True)
    strategies["evaluator_top_50pct"] = sorted_by_score[:n_total // 2]

    # 4. Evaluator top-K (top 30%)
    strategies["evaluator_top_30pct"] = sorted_by_score[:n_total * 3 // 10]

    # 5. Evaluator threshold (score >= 3)
    strategies["evaluator_thresh_3"] = [r for r in valid if r["predicted_score"] >= 3]

    # 6. Evaluator threshold (score >= 4)
    strategies["evaluator_thresh_4"] = [r for r in valid if r["predicted_score"] >= 4]

    # 7. Low quality (score <= 1) - for comparison
    strategies["evaluator_low"] = [r for r in valid if r["predicted_score"] <= 1]

    # 8. BLEU-based filtering (medium overlap)
    def char_bleu(hyp, ref):
        from collections import Counter
        hc = Counter(list(hyp))
        rc = Counter(list(ref))
        if not hc:
            return 0.0
        clipped = sum(min(hc[c], rc[c]) for c in hc)
        return clipped / len(hc)

    bleu_scored = []
    for r in valid:
        bleu = char_bleu(r["rewrite_text"], r["source_text"])
        bleu_scored.append((r, bleu))
    bleu_mid = [r for r, b in bleu_scored if 0.2 <= b <= 0.6]
    strategies["bleu_mid_range"] = bleu_mid

    for name, filtered in strategies.items():
        scores = [r["predicted_score"] for r in filtered]
        mean_s = np.mean(scores) if scores else 0
        std_s = np.std(scores) if scores else 0
        print(f"  {name:25s}: {len(filtered):4d} samples, mean_score={mean_s:.2f}, std={std_s:.2f}")

        # Save as SFT format
        sft_data = format_for_sft(filtered)
        out_path = output_dir / f"sft_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

    return strategies


def format_for_sft(rewrites):
    """Format rewrites for SFT training."""
    system_prompt = "你是一个专业的中文文本改写助手。请根据用户提供的原文，生成一段高质量的改写文本。改写应保留原文核心语义，使用不同的词汇和句式表达。"
    sft_data = []
    for r in rewrites:
        sft_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请改写以下文本：\n{r['source_text']}"},
                {"role": "assistant", "content": r["rewrite_text"]},
            ],
        })
    return sft_data


def compute_quality_metrics(scored_rewrites, strategy_name, filtered):
    """Compute quality metrics for a filtered dataset."""
    scores = [r["predicted_score"] for r in filtered]
    if not scores:
        return {}

    return {
        "strategy": strategy_name,
        "n_samples": len(filtered),
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "median_score": round(float(np.median(scores)), 4),
        "min_score": min(scores),
        "max_score": max(scores),
        "pct_score_ge_3": round(sum(1 for s in scores if s >= 3) / len(scores) * 100, 1),
        "pct_score_ge_4": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
    }


def generate_downstream_table(strategies, output_dir):
    """Generate downstream validation results table (LaTeX)."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Data quality analysis under different evaluator-guided filtering strategies. "
                 r"Higher mean evaluator scores indicate better predicted rewrite quality.}")
    lines.append(r"\label{tab:downstream_filtering}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Filtering Strategy & $N$ & Mean Score & $\sigma$ & \% Score $\geq$ 3 \\")
    lines.append(r"\midrule")

    # Compute metrics for each strategy
    entries = []
    for name, filtered in strategies.items():
        scores = [r["predicted_score"] for r in filtered]
        if not scores:
            continue
        entries.append({
            "name": name.replace("_", " ").title(),
            "n": len(filtered),
            "mean": round(np.mean(scores), 2),
            "std": round(np.std(scores), 2),
            "pct_ge3": round(sum(1 for s in scores if s >= 3) / len(scores) * 100, 1),
        })

    # Sort by mean score
    entries.sort(key=lambda e: e["mean"], reverse=True)

    for e in entries:
        is_ours = "evaluator" in e["name"].lower() and "top" in e["name"].lower()
        name = r"\textbf{" + e["name"] + r"}" if is_ours else e["name"]
        mean_str = f"\\textbf{{{e['mean']}}}" if is_ours else str(e["mean"])
        lines.append(f"{name} & {e['n']} & {mean_str} & {e['std']} & {e['pct_ge3']}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = output_dir / "downstream_filtering_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {out_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Downstream validation pipeline")
    parser.add_argument("--scored_path", type=str, default=str(GENERATED_DIR / "scored_rewrites.json"))
    parser.add_argument("--output_dir", type=str, default=str(FILTERED_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scored rewrites
    print(f"Loading scored rewrites from {args.scored_path}...")
    with open(args.scored_path, "r", encoding="utf-8") as f:
        scored_rewrites = json.load(f)
    print(f"  Loaded {len(scored_rewrites)} scored rewrites")

    # Step 1: Analyze score distribution
    print("\n" + "=" * 60)
    print("Step 1: Score Distribution Analysis")
    print("=" * 60)
    analysis = analyze_score_distribution(scored_rewrites, output_dir)

    # Step 2: Create filtered datasets
    print("\n" + "=" * 60)
    print("Step 2: Create Filtered Datasets")
    print("=" * 60)
    strategies = create_filtered_datasets(scored_rewrites, output_dir)

    # Step 3: Generate table
    print("\n" + "=" * 60)
    print("Step 3: Generate Results Table")
    print("=" * 60)
    generate_downstream_table(strategies, output_dir)

    # Step 4: Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    valid = [r for r in scored_rewrites if r["predicted_score"] is not None]
    if valid:
        sorted_valid = sorted(valid, key=lambda x: x["predicted_score"], reverse=True)
        top_50 = sorted_valid[:len(sorted_valid) // 2]
        random_half = valid[len(valid) // 2:]  # Rough random split

        top_mean = np.mean([r["predicted_score"] for r in top_50])
        rand_mean = np.mean([r["predicted_score"] for r in random_half])
        print(f"  Top-50% evaluator score: {top_mean:.2f}")
        print(f"  Bottom-50% evaluator score: {rand_mean:.2f}")
        print(f"  Improvement: +{top_mean - rand_mean:.2f} points ({(top_mean - rand_mean) / rand_mean * 100:.1f}%)")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
