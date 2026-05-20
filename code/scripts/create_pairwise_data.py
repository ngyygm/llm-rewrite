"""
Create pairwise training data for LoRA evaluator.

Constructs two datasets:
1. generated_train/eval: Same-source pairs from scored_rewrites.json
   - Pairs rewrites of different prompt_types (quality levels) for the same source
   - 200 sources for train, 100 for eval

2. cross_source_train: Cross-source pairs from human_eval/train.json
   - Pairs samples from different sources where |avg_score_A - avg_score_B| >= 0.5
   - Max 10 pairs per sample, prioritizing larger score differences
   - Balanced labels (~50% each)
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
SEED = 42

SYSTEM_PROMPT = (
    "你是一个专业的文本改写质量评估专家。"
    "请根据原文和两篇改写，判断哪篇改写的质量更高。"
)

# Label to response mapping
LABEL_RESPONSES = {
    1: "改写A的质量更高。",   # A preferred
    0: "改写B的质量更高。",   # B preferred
    0.5: "两篇改写质量相当。",  # tie (unlikely for generated data)
}


def make_pairwise_sample(source_text, rewrite_a, rewrite_b, label):
    """Create a single pairwise training sample in LoRA format."""
    user_content = (
        f"原文：{source_text}\n\n"
        f"改写A：{rewrite_a}\n\n"
        f"改写B：{rewrite_b}\n\n"
        f"请问哪篇改写的质量更高？"
        f"如果改写A更好请回答A，如果改写B更好请回答B，如果两者差不多请回答平局。"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": LABEL_RESPONSES[label]},
        ]
    }


def build_generated_pairs(scored_rewrites, rng):
    """
    Build same-source pairwise data from scored rewrites.

    Each source has 3 rewrites (prompt_type 0=high, 1=mid, 2=low).
    Creates all ordered pairs of different prompt_types:
      (0,1): label=1, (1,0): label=0
      (0,2): label=1, (2,0): label=0
      (1,2): label=1, (2,1): label=0

    Returns list of (source_hash, prompt_type_a, prompt_type_b, label) tuples
    organized by source_hash.
    """
    # Group rewrites by source_hash
    by_source = defaultdict(dict)
    for item in scored_rewrites:
        by_source[item["source_hash"]][item["prompt_type"]] = item

    # Define pairwise comparisons: (type_a, type_b, label)
    # label=1 means A is better (type_a has lower number = higher quality)
    comparisons = [
        (0, 1, 1), (1, 0, 0),
        (0, 2, 1), (2, 0, 0),
        (1, 2, 1), (2, 1, 0),
    ]

    source_pairs = []
    for source_hash, rewrites in sorted(by_source.items()):
        # Ensure all 3 prompt types exist
        if not all(pt in rewrites for pt in [0, 1, 2]):
            continue
        for pt_a, pt_b, label in comparisons:
            source_pairs.append({
                "source_hash": source_hash,
                "prompt_type_a": pt_a,
                "prompt_type_b": pt_b,
                "label": label,
                "source_text": rewrites[pt_a]["source_text"],
                "rewrite_a": rewrites[pt_a]["rewrite_text"],
                "rewrite_b": rewrites[pt_b]["rewrite_text"],
            })

    return source_pairs


def build_cross_source_pairs(train_data, rng, max_pairs_per_sample=10):
    """
    Build cross-source pairwise data from human annotations.

    For each sample, find up to max_pairs_per_sample partners where
    |avg_score_A - avg_score_B| >= 0.5. Prioritize larger score differences
    but include a mix of difficulty levels (not just the most extreme).

    Strategy:
    1. Sort all samples by avg_score.
    2. For each sample, find valid partners using binary search on sorted
       scores (score difference >= 0.5).
    3. Sample partners weighted towards larger differences but capped per sample.
    4. Collect all pairs, then balance labels by undersampling the majority.
    """
    import bisect

    n = len(train_data)
    scores = [train_data[i]["avg_score"] for i in range(n)]

    # Sort indices by score for efficient range queries
    sorted_indices = sorted(range(n), key=lambda i: scores[i])
    sorted_scores = [scores[i] for i in sorted_indices]

    all_pairs = []  # (idx_a, idx_b, label)
    pair_count_per_sample = defaultdict(int)

    # Process samples in random order
    process_order = list(range(n))
    rng.shuffle(process_order)

    for idx_a in process_order:
        if pair_count_per_sample[idx_a] >= max_pairs_per_sample:
            continue

        avg_a = scores[idx_a]

        # Find valid partner range: scores outside [avg_a - 0.5, avg_a + 0.5]
        lo_threshold = avg_a - 0.5
        hi_threshold = avg_a + 0.5

        # Use binary search on sorted scores to find candidate ranges
        lo_end = bisect.bisect_left(sorted_scores, lo_threshold)
        hi_start = bisect.bisect_right(sorted_scores, hi_threshold)

        # Collect candidate indices with both valid diff and under pair limit
        candidates = []
        for pos in list(range(lo_end)) + list(range(hi_start, n)):
            idx_b = sorted_indices[pos]
            if idx_b == idx_a:
                continue
            if pair_count_per_sample[idx_b] >= max_pairs_per_sample:
                continue
            avg_b = scores[idx_b]
            diff = abs(avg_a - avg_b)
            if diff < 0.5:
                continue
            label = 1 if avg_a > avg_b else 0
            candidates.append((idx_b, diff, label))

        if not candidates:
            continue

        # Separate into tiers: hard (0.5-1.5), medium (1.5-3.0), easy (3.0+)
        # Sample from each tier to get diverse difficulty
        hard = [c for c in candidates if 0.5 <= c[1] < 1.5]
        medium = [c for c in candidates if 1.5 <= c[1] < 3.0]
        easy = [c for c in candidates if c[1] >= 3.0]

        # Allocate slots: 40% hard, 30% medium, 30% easy (rounded)
        remaining = max_pairs_per_sample - pair_count_per_sample[idx_a]
        n_hard = min(len(hard), max(1, int(remaining * 0.4)))
        n_medium = min(len(medium), max(1, int(remaining * 0.3)))
        n_easy = min(len(easy), max(1, int(remaining * 0.3)))

        # If a tier is empty, redistribute
        if not hard:
            extra = n_hard
            n_hard = 0
            n_medium = min(len(medium), n_medium + extra // 2)
            n_easy = min(len(easy), n_easy + extra - extra // 2)
        if not medium:
            extra = n_medium
            n_medium = 0
            n_hard = min(len(hard), n_hard + extra // 2)
            n_easy = min(len(easy), n_easy + extra - extra // 2)
        if not easy:
            extra = n_easy
            n_easy = 0
            n_hard = min(len(hard), n_hard + extra // 2)
            n_medium = min(len(medium), n_medium + extra - extra // 2)

        # Randomly sample from each tier
        rng.shuffle(hard)
        rng.shuffle(medium)
        rng.shuffle(easy)

        selected = hard[:n_hard] + medium[:n_medium] + easy[:n_easy]

        for idx_b, diff, label in selected:
            all_pairs.append((idx_a, idx_b, label))
            pair_count_per_sample[idx_a] += 1
            pair_count_per_sample[idx_b] += 1

    print(f"  Total pairs before balancing: {len(all_pairs)}")

    # Balance labels
    label_1 = [(a, b, l) for a, b, l in all_pairs if l == 1]
    label_0 = [(a, b, l) for a, b, l in all_pairs if l == 0]

    print(f"  Label 1 (A preferred): {len(label_1)}, Label 0 (B preferred): {len(label_0)}")

    target_count = min(len(label_1), len(label_0))

    if len(label_1) > target_count:
        rng.shuffle(label_1)
        label_1 = label_1[:target_count]
    if len(label_0) > target_count:
        rng.shuffle(label_0)
        label_0 = label_0[:target_count]

    balanced_pairs = label_1 + label_0
    rng.shuffle(balanced_pairs)

    return balanced_pairs


def main():
    rng = np.random.RandomState(SEED)
    out_dir = BASE_DIR / "data" / "pairwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}

    # =========================================================================
    # Dataset 1: Generated rewrites (same-source pairs)
    # =========================================================================
    print("=" * 60)
    print("DATASET 1: Generated rewrites (same-source pairs)")
    print("=" * 60)

    scored_path = BASE_DIR / "data" / "generated_rewrites" / "scored_rewrites.json"
    with open(scored_path, "r", encoding="utf-8") as f:
        scored_rewrites = json.load(f)
    print(f"Loaded {len(scored_rewrites)} scored rewrites")

    # Build pairs
    all_pairs = build_generated_pairs(scored_rewrites, rng)
    print(f"Created {len(all_pairs)} pairwise comparisons")

    # Get unique source hashes and split 200 train / 100 eval
    unique_hashes = sorted(set(p["source_hash"] for p in all_pairs))
    rng.shuffle(unique_hashes)
    train_hashes = set(unique_hashes[:200])
    eval_hashes = set(unique_hashes[200:])

    train_pairs = [p for p in all_pairs if p["source_hash"] in train_hashes]
    eval_pairs = [p for p in all_pairs if p["source_hash"] in eval_hashes]

    print(f"Train sources: {len(train_hashes)}, Eval sources: {len(eval_hashes)}")
    print(f"Train pairs: {len(train_pairs)}, Eval pairs: {len(eval_pairs)}")

    # Convert to LoRA format
    train_samples = [
        make_pairwise_sample(p["source_text"], p["rewrite_a"], p["rewrite_b"], p["label"])
        for p in train_pairs
    ]
    eval_samples = [
        make_pairwise_sample(p["source_text"], p["rewrite_a"], p["rewrite_b"], p["label"])
        for p in eval_pairs
    ]

    # Save
    train_path = out_dir / "generated_train.json"
    eval_path = out_dir / "generated_eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {train_path}: {len(train_samples)} samples")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {eval_path}: {len(eval_samples)} samples")

    # Label distribution for generated data
    gen_train_labels = Counter(p["label"] for p in train_pairs)
    gen_eval_labels = Counter(p["label"] for p in eval_pairs)

    # Prompt type pair distribution
    gen_pt_pairs = Counter(
        (p["prompt_type_a"], p["prompt_type_b"]) for p in all_pairs
    )

    metadata["generated"] = {
        "train": {
            "total_pairs": len(train_pairs),
            "train_sources": len(train_hashes),
            "label_distribution": dict(gen_train_labels),
            "prompt_type_pair_distribution": dict(
                (f"({a},{b})", c)
                for (a, b), c in gen_pt_pairs.items()
            ),
        },
        "eval": {
            "total_pairs": len(eval_pairs),
            "eval_sources": len(eval_hashes),
            "label_distribution": dict(gen_eval_labels),
        },
    }

    print(f"\nLabel distribution (train): {dict(gen_train_labels)}")
    print(f"Label distribution (eval):  {dict(gen_eval_labels)}")
    print(f"Prompt type pairs: {dict(gen_pt_pairs)}")

    # =========================================================================
    # Dataset 2: Cross-source pairs from human annotations
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATASET 2: Cross-source pairs (human annotations)")
    print("=" * 60)

    train_raw_path = BASE_DIR / "data" / "human_eval" / "train.json"
    with open(train_raw_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    print(f"Loaded {len(train_raw)} training samples")

    # Score distribution
    score_counts = Counter(int(s["avg_score"]) for s in train_raw)
    print(f"Score distribution: {dict(sorted(score_counts.items()))}")

    # Build cross-source pairs
    balanced_pairs = build_cross_source_pairs(train_raw, rng)
    print(f"Created {len(balanced_pairs)} balanced cross-source pairs")

    # Label distribution
    cs_labels = Counter(l for _, _, l in balanced_pairs)
    print(f"Label distribution: {dict(cs_labels)}")

    # Score difference distribution
    score_diffs = [
        abs(train_raw[a]["avg_score"] - train_raw[b]["avg_score"])
        for a, b, _ in balanced_pairs
    ]
    print(f"Score difference stats: min={min(score_diffs):.3f}, "
          f"max={max(score_diffs):.3f}, "
          f"mean={np.mean(score_diffs):.3f}, "
          f"median={np.median(score_diffs):.3f}")

    # Convert to LoRA format
    # For cross-source pairs, we present both source texts and their rewrites.
    # Format: show both source+rewrite pairs so the model can compare.
    cross_source_samples = []
    for idx_a, idx_b, label in balanced_pairs:
        item_a = train_raw[idx_a]
        item_b = train_raw[idx_b]

        # For cross-source, include both source texts since they differ
        user_content = (
            f"原文A：{item_a['input']}\n\n"
            f"改写A：{item_a['output']}\n\n"
            f"原文B：{item_b['input']}\n\n"
            f"改写B：{item_b['output']}\n\n"
            f"请问哪篇改写的质量更高？"
            f"如果改写A更好请回答A，如果改写B更好请回答B，如果两者差不多请回答平局。"
        )
        cross_source_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": LABEL_RESPONSES[label]},
            ]
        })

    # Save
    cs_train_path = out_dir / "cross_source_train.json"
    with open(cs_train_path, "w", encoding="utf-8") as f:
        json.dump(cross_source_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {cs_train_path}: {len(cross_source_samples)} samples")

    # Per-sample pair count distribution
    pair_counts = Counter()
    for idx_a, idx_b, _ in balanced_pairs:
        pair_counts[idx_a] += 1
        pair_counts[idx_b] += 1
    pair_count_values = list(pair_counts.values())
    print(f"Pairs per sample: min={min(pair_count_values)}, "
          f"max={max(pair_count_values)}, "
          f"mean={np.mean(pair_count_values):.1f}, "
          f"median={np.median(pair_count_values):.1f}")

    metadata["cross_source"] = {
        "total_pairs": len(balanced_pairs),
        "source_samples_used": len(pair_counts),
        "label_distribution": dict(cs_labels),
        "score_diff_stats": {
            "min": round(min(score_diffs), 3),
            "max": round(max(score_diffs), 3),
            "mean": round(float(np.mean(score_diffs)), 3),
            "median": round(float(np.median(score_diffs)), 3),
        },
        "pairs_per_sample_stats": {
            "min": min(pair_count_values),
            "max": max(pair_count_values),
            "mean": round(float(np.mean(pair_count_values)), 1),
            "median": round(float(np.median(pair_count_values)), 1),
        },
        "original_score_distribution": dict(sorted(score_counts.items())),
    }

    # =========================================================================
    # Save metadata
    # =========================================================================
    meta_path = out_dir / "pairwise_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metadata: {meta_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated train: {len(train_samples)} pairs (200 sources x 6 pairs)")
    print(f"Generated eval:  {len(eval_samples)} pairs (100 sources x 6 pairs)")
    print(f"Cross-source:    {len(cross_source_samples)} pairs")
    print(f"All files saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
