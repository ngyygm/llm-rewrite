"""
Create score-balanced training data for LoRA evaluator.

The original data is heavily skewed (43% scores 0-1), causing the model
to predict score 1 for 82% of samples. This script creates balanced
versions where each score level is equally represented via oversampling.

Also adds reasoning phrases before the score to encourage analysis.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "human_eval"
SEED = 42

# Reasoning templates per score level to encourage analytical thinking
REASONING_TEMPLATES = {
    0: [
        "该改写存在严重问题：",
        "经过分析，该改写质量极差：",
        "该改写完全不符合要求：",
    ],
    1: [
        "该改写质量较差，分析如下：",
        "该改写存在较多问题：",
        "该改写水平较低，原因如下：",
    ],
    2: [
        "该改写质量一般偏下：",
        "该改写有一定问题但尚可：",
        "该改写部分达标：",
    ],
    3: [
        "该改写质量尚可：",
        "该改写基本符合要求：",
        "该改写水平中等偏上：",
    ],
    4: [
        "该改写质量较好：",
        "该改写表现出色：",
        "该改写大部分符合要求：",
    ],
    5: [
        "该改写质量优秀：",
        "该改写表现极佳：",
        "该改写完全符合所有要求：",
    ],
}

SYSTEM_PROMPT_SCORE_BALANCED = """你是一个专业的中文文本改写质量评估专家。请根据以下维度对中文文本改写质量进行评分（0-5分）：

评分维度：
1. 语义一致性：改写是否保留了原文的核心语义
2. 句式重构：改写是否进行了句法结构的改变
3. 词汇变化：改写是否使用了不同的词汇表达
4. 风格保持：改写是否保持了原文的风格和长度

评分标准：
- 0分：改写完全失败（严重语义扭曲或毫无改写）
- 1分：改写质量很差
- 2分：改写质量较差
- 3分：改写质量一般
- 4分：改写质量较好
- 5分：改写质量优秀

请先简要分析改写的优缺点，然后给出最终综合评分（0-5分的整数）。"""


def make_reasoning_sample(input_text, output_text, score, rng):
    """Create training sample with reasoning before score."""
    user_content = (
        f"原文：\n{input_text}\n\n改写：\n{output_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )
    # Pick a random reasoning template
    template = rng.choice(REASONING_TEMPLATES[score])
    assistant_content = f"{template}综合评分为{score}分。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SCORE_BALANCED},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def make_simple_sample(input_text, output_text, score):
    """Create simple score-only training sample (original format)."""
    user_content = (
        f"原文：\n{input_text}\n\n改写：\n{output_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )
    assistant_content = f"该改写的综合评分为{score}分。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SCORE_BALANCED},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def balance_dataset(train_raw, rng):
    """Oversample minority classes to create balanced dataset."""
    # Group by score
    by_score = {s: [] for s in range(6)}
    for item in train_raw:
        by_score[item["consensus_score"]].append(item)

    print("Original distribution:")
    for s in range(6):
        print(f"  Score {s}: {len(by_score[s])} samples")

    # Target: equal samples per class, matching the majority class
    max_count = max(len(v) for v in by_score.values())
    print(f"\nBalancing to {max_count} samples per class...")

    balanced = []
    for s in range(6):
        items = by_score[s]
        n_needed = max_count - len(items)
        if n_needed > 0:
            # Oversample with random selection (with replacement)
            oversampled = rng.choice(len(items), size=n_needed, replace=True).tolist()
            items = items + [items[i] for i in oversampled]
        rng.shuffle(items)
        balanced.extend(items)

    rng.shuffle(balanced)

    print("\nBalanced distribution:")
    score_counts = Counter(d["consensus_score"] for d in balanced)
    for s in range(6):
        print(f"  Score {s}: {score_counts[s]} samples")
    print(f"  Total: {len(balanced)}")

    return balanced


def main():
    rng = np.random.RandomState(SEED)

    # Load training data
    print("Loading training data...")
    with open(OUT_DIR / "train.json", "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    print(f"  Loaded {len(train_raw)} training samples")

    # Create balanced version
    balanced = balance_dataset(train_raw, rng)

    # Format 1: Balanced + reasoning (primary)
    balanced_reasoning = [
        make_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng)
        for d in balanced
    ]

    # Format 2: Balanced + simple (no reasoning prefix)
    balanced_simple = [
        make_simple_sample(d["input"], d["output"], d["consensus_score"])
        for d in balanced
    ]

    # Format 3: Original (unbalanced) + reasoning
    original_reasoning = [
        make_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng)
        for d in train_raw
    ]

    # Save
    with open(OUT_DIR / "train_score_only_balanced.json", "w", encoding="utf-8") as f:
        json.dump(balanced_simple, f, ensure_ascii=False, indent=2)
    print(f"\nSaved train_score_only_balanced.json: {len(balanced_simple)} samples")

    with open(OUT_DIR / "train_score_reasoning.json", "w", encoding="utf-8") as f:
        json.dump(original_reasoning, f, ensure_ascii=False, indent=2)
    print(f"Saved train_score_reasoning.json: {len(original_reasoning)} samples")

    with open(OUT_DIR / "train_score_balanced_reasoning.json", "w", encoding="utf-8") as f:
        json.dump(balanced_reasoning, f, ensure_ascii=False, indent=2)
    print(f"Saved train_score_balanced_reasoning.json: {len(balanced_reasoning)} samples")

    # Also create balanced subsets for learning curves
    subsets = [50, 100, 200, 400]
    for size in subsets:
        rng_sub = np.random.RandomState(SEED + size + 1000)  # Different seed to avoid overlap with original subsets
        # Sample from balanced dataset, ensuring stratification
        by_score = {s: [] for s in range(6)}
        for item in balanced:
            by_score[item["consensus_score"]].append(item)

        per_class = max(1, size // 6)
        remainder = size - per_class * 6
        subset = []
        for s in range(6):
            rng_sub.shuffle(by_score[s])
            n = per_class + (1 if s < remainder else 0)
            subset.extend(by_score[s][:n])

        rng_sub.shuffle(subset)

        sub_data = [
            make_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng_sub)
            for d in subset
        ]

        with open(OUT_DIR / f"train_balanced_reasoning_{size}.json", "w", encoding="utf-8") as f:
            json.dump(sub_data, f, ensure_ascii=False, indent=2)
        print(f"Saved train_balanced_reasoning_{size}.json: {len(sub_data)} samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
