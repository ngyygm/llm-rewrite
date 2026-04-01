"""
Filter generated rewrite data using evaluator scores.

Supports multiple filtering strategies:
1. Top-K: Select top K rewrites by evaluator score
2. Threshold: Select rewrites with score >= threshold
3. Random: Baseline (no filtering)
4. BLEU-filtered: Select by BLEU range
5. Evaluator-filtered: Use trained LoRA evaluator scores
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "data" / "generated_rewrites"
FILTERED_DIR = GENERATED_DIR / "filtered"
EVALUATOR_DIR = BASE_DIR / "evaluator"

SEED = 42


def load_rewrites(rewrites_path: Optional[str] = None) -> List[Dict]:
    """Load generated rewrites."""
    path = Path(rewrites_path) if rewrites_path else GENERATED_DIR / "all_rewrites.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluator_scores(scores_path: str) -> Dict[str, float]:
    """Load evaluator scores (source_hash -> score mapping)."""
    with open(scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build hash -> score mapping
    scores = {}
    for item in data.get("results", data if isinstance(data, list) else []):
        key = item.get("source_hash", item.get("hash", ""))
        if key:
            scores[key] = item.get("score", item.get("predicted_score", 0))
    return scores


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute simple BLEU score (character-level for Chinese)."""
    from collections import Counter

    hyp_chars = list(hypothesis)
    ref_chars = list(reference)

    if len(hyp_chars) == 0:
        return 0.0

    # Unigram precision
    hyp_counts = Counter(hyp_chars)
    ref_counts = Counter(ref_chars)

    clipped = sum(min(hyp_counts[c], ref_counts[c]) for c in hyp_counts)
    precision = clipped / len(hyp_chars) if len(hyp_chars) > 0 else 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_chars) / len(hyp_chars))) if len(hyp_chars) > 0 else 0.0

    return bp * precision


import math


def filter_top_k(rewrites: List[Dict], scores: Dict[str, float], k: int) -> List[Dict]:
    """Select top K rewrites by evaluator score."""
    scored = [(r, scores.get(r["source_hash"], 0)) for r in rewrites]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, s in scored[:k]]


def filter_by_threshold(rewrites: List[Dict], scores: Dict[str, float], threshold: float) -> List[Dict]:
    """Select rewrites with score >= threshold."""
    return [r for r in rewrites if scores.get(r["source_hash"], 0) >= threshold]


def filter_random(rewrites: List[Dict], k: int, seed: int = SEED) -> List[Dict]:
    """Random selection baseline."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(rewrites)), min(k, len(rewrites)))
    return [rewrites[i] for i in indices]


def filter_by_bleu_range(rewrites: List[Dict], min_bleu: float = 0.3, max_bleu: float = 0.8) -> List[Dict]:
    """Select rewrites with BLEU score in specified range."""
    filtered = []
    for r in rewrites:
        bleu = compute_bleu(r["rewrite_text"], r["source_text"])
        if min_bleu <= bleu <= max_bleu:
            filtered.append(r)
    return filtered


def format_for_sft(rewrites: List[Dict]) -> List[Dict]:
    """Format filtered rewrites as SFT training data.

    Format: {"messages": [{"role": "system", ...}, {"role": "user", "input"}, {"role": "assistant", "output"}]}
    """
    system_prompt = "你是一个专业的中文文本改写助手。请根据用户提供的原文，生成一段高质量的改写文本。改写应保留原文核心语义，使用不同的词汇和句式表达。"

    sft_data = []
    for r in rewrites:
        sft_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请改写以下文本：\n{r['source_text']}"},
                {"role": "assistant", "content": r["rewrite_text"]},
            ],
            "metadata": {
                "source_category": r.get("source_category", ""),
                "quality_level": r.get("quality_level", ""),
            },
        })
    return sft_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter rewrite data for SFT")
    parser.add_argument("--rewrites_path", type=str, default="",
                        help="Path to rewrites JSON")
    parser.add_argument("--scores_path", type=str, default="",
                        help="Path to evaluator scores JSON")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["top_k", "threshold", "random", "bleu", "all"],
                        help="Filtering strategy")
    parser.add_argument("--k", type=int, default=2000,
                        help="Number of samples for top_k/random")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Score threshold")
    parser.add_argument("--output_dir", type=str, default=str(FILTERED_DIR),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rewrites...")
    rewrites = load_rewrites(args.rewrites_path or None)
    print(f"  Total rewrites: {len(rewrites)}")

    # Load evaluator scores if available
    scores = {}
    if args.scores_path:
        scores = load_evaluator_scores(args.scores_path)
        print(f"  Loaded {len(scores)} evaluator scores")
    else:
        print("  No evaluator scores provided, using quality_level as proxy")
        # Use quality_level as proxy score
        quality_to_score = {"low": 1, "medium": 3, "high": 5}
        for r in rewrites:
            h = hashlib.md5(
                (r["source_text"] + r["rewrite_text"]).encode("utf-8")
            ).hexdigest()
            scores[h] = quality_to_score.get(r.get("quality_level", "medium"), 3)

    import hashlib

    # Need to recompute scores with correct hash
    scores = {}
    quality_to_score = {"low": 1, "medium": 3, "high": 5}
    for r in rewrites:
        h = hashlib.md5(
            (r["source_text"] + r["rewrite_text"]).encode("utf-8")
        ).hexdigest()
        scores[h] = quality_to_score.get(r.get("quality_level", "medium"), 3)

    strategies = {}

    if args.strategy in ["all", "random"]:
        strategies["random_2000"] = filter_random(rewrites, 2000)

    if args.strategy in ["all", "bleu"]:
        strategies["bleu_filtered"] = filter_by_bleu_range(rewrites)

    if args.strategy in ["all", "top_k"]:
        strategies[f"top_{args.k}"] = filter_top_k(rewrites, scores, args.k)

    if args.strategy in ["all", "threshold"]:
        strategies[f"threshold_{args.threshold}"] = filter_by_threshold(rewrites, scores, args.threshold)

    # Save filtered datasets
    for name, filtered in strategies.items():
        print(f"\nStrategy: {name}")
        print(f"  Selected: {len(filtered)} rewrites")

        sft_data = format_for_sft(filtered)
        out_path = output_dir / f"sft_{name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved to: {out_path}")

        # Save raw filtered data too (for analysis)
        raw_path = output_dir / f"filtered_{name}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
