#!/usr/bin/env python3
"""
Error Analysis: Why Traditional Metrics Negatively Correlate with Human Judgment.

Identifies cases where:
1. High metric score but low human score (metric rewards bad rewrites)
2. Low metric score but high human score (metric penalizes good rewrites)

Categorizes the error patterns and outputs examples for the paper.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_data():
    """Load eval data and traditional metric results."""
    eval_path = PROJECT_ROOT / "data" / "human_eval" / "eval.json"
    trad_path = PROJECT_ROOT / "data" / "baselines" / "all_results_traditional.json"

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(trad_path, "r", encoding="utf-8") as f:
        trad_results = json.load(f)

    return eval_data, trad_results


def analyze_disagreements(eval_data, trad_results, metric_key, n_examples=5):
    """Find cases where metric and human score disagree most."""
    n = len(eval_data)

    # Collect (metric_score, human_score, index) tuples
    disagreements = []
    for i in range(n):
        human = eval_data[i]["avg_score"]
        metric = trad_results[metric_key][i]
        disagreements.append((metric, human, i))

    # Sort by metric-human gap (positive = metric thinks it's better than human does)
    disagreements.sort(key=lambda x: x[0] - x[1], reverse=True)

    # High metric, low human: metric rewards bad rewrites
    high_metric_low_human = [(m, h, idx) for m, h, idx in disagreements if m > 0.5 and h <= 1.0]

    # Low metric, high human: metric penalizes good rewrites
    low_metric_high_human = [(m, h, idx) for m, h, idx in reversed(disagreements) if m < 0.3 and h >= 3.0]

    return high_metric_low_human[:n_examples], low_metric_high_human[:n_examples]


def categorize_error(source, rewrite, metric_score, human_score):
    """Categorize why the metric disagrees with human judgment."""
    reasons = []

    # Check surface similarity
    source_chars = set(source)
    rewrite_chars = set(rewrite)
    common_chars = source_chars & rewrite_chars
    char_overlap = len(common_chars) / max(len(source_chars), 1)

    # Check length ratio
    len_ratio = len(rewrite) / max(len(source), 1)

    if char_overlap > 0.7 and human_score <= 1:
        reasons.append("高表面相似度但低质量: 改写几乎复制原文，没有实质性改进")
    elif char_overlap < 0.5 and human_score >= 3:
        reasons.append("低表面相似度但高质量: 改写大幅重构但保留了语义")
    elif char_overlap > 0.6 and human_score >= 3:
        reasons.append("高表面相似度且高质量: 改写保留了结构但改善了表达")
    elif len_ratio > 1.3 and human_score <= 1:
        reasons.append("过度膨胀: 改写大幅增加长度但没有实质内容")
    elif len_ratio < 0.7 and human_score <= 1:
        reasons.append("过度压缩: 改写丢失了重要信息")
    else:
        reasons.append("其他: 指标与人类判断在细微质量差异上不一致")

    return reasons


def main():
    eval_data, trad_results = load_data()
    n = len(eval_data)
    print(f"Eval samples: {n}")
    print()

    # Use BLEU as the primary metric for analysis
    metric_key = "trad_bleu"
    metric_name = "BLEU"

    high_low, low_high = analyze_disagreements(eval_data, trad_results, metric_key, n_examples=5)

    print(f"=== {metric_name}: High Metric / Low Human (metric rewards bad rewrites) ===\n")
    for rank, (metric_sc, human_sc, idx) in enumerate(high_low, 1):
        item = eval_data[idx]
        reasons = categorize_error(item["input"], item["output"], metric_sc, human_sc)
        print(f"Example {rank} (idx={idx}):")
        print(f"  BLEU={metric_sc:.4f}, Human={human_sc:.1f}, Gap={metric_sc-human_sc:+.2f}")
        print(f"  Source:  {item['input'][:80]}...")
        print(f"  Rewrite: {item['output'][:80]}...")
        print(f"  Reason:  {reasons[0]}")
        print()

    print(f"=== {metric_name}: Low Metric / High Human (metric penalizes good rewrites) ===\n")
    for rank, (metric_sc, human_sc, idx) in enumerate(low_high, 1):
        item = eval_data[idx]
        reasons = categorize_error(item["input"], item["output"], metric_sc, human_sc)
        print(f"Example {rank} (idx={idx}):")
        print(f"  BLEU={metric_sc:.4f}, Human={human_sc:.1f}, Gap={metric_sc-human_sc:+.2f}")
        print(f"  Source:  {item['input'][:80]}...")
        print(f"  Rewrite: {item['output'][:80]}...")
        print(f"  Reason:  {reasons[0]}")
        print()

    # Compute statistics
    print("=== Overall Statistics ===\n")

    # Compute Spearman for each metric
    from scipy.stats import spearmanr
    human_scores = [item["avg_score"] for item in eval_data]

    print(f"{'Metric':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for key, meta in get_metric_metadata().items():
        if key in trad_results:
            rho, pval = spearmanr(trad_results[key], human_scores)
            print(f"{meta['name']:<20} {rho:>12.4f} {pval:>12.2e}")

    print()

    # Distribution analysis
    print("=== Score Distribution Analysis ===\n")
    score_bins = defaultdict(int)
    for item in eval_data:
        bucket = int(item["avg_score"])
        score_bins[bucket] += 1

    for bucket in sorted(score_bins.keys()):
        bar = "#" * (score_bins[bucket] * 2)
        print(f"  Score {bucket}: {score_bins[bucket]:>3} {bar}")


def get_metric_metadata():
    return {
        "trad_bleu": {"name": "BLEU"},
        "trad_rouge_l": {"name": "ROUGE-L"},
        "trad_sbert_cosine": {"name": "SBERT-COSINE"},
        "trad_tfidf_cosine": {"name": "TFIDF-COSINE"},
        "trad_w2v_cosine": {"name": "W2V-COSINE"},
        "trad_jaccard_word": {"name": "JACCARD-WORD"},
        "trad_jaccard_char": {"name": "JACCARD-CHAR"},
    }


if __name__ == "__main__":
    main()
