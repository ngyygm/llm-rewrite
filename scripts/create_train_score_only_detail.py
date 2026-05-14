#!/usr/bin/env python3
"""
Create detail score-only training variants from train.json.

Input:
  data/human_eval/train.json

Outputs:
  data/human_eval/train_score_only_detail.json
  data/human_eval/train_score_only_detail_balanced.json
  data/human_eval/train_score_detail_reasoning.json
  data/human_eval/train_score_detail_balanced_reasoning.json
  data/human_eval/train_score_only_detail_balanced_{size}.json
  data/human_eval/train_detail_balanced_reasoning_{size}.json
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "human_eval"
INPUT_PATH = DATA_DIR / "train.json"
OUTPUT_PATH = DATA_DIR / "train_score_only_detail.json"
BALANCED_OUTPUT_PATH = DATA_DIR / "train_score_only_detail_balanced.json"
REASONING_OUTPUT_PATH = DATA_DIR / "train_score_detail_reasoning.json"
BALANCED_REASONING_OUTPUT_PATH = DATA_DIR / "train_score_detail_balanced_reasoning.json"
SEED = 42

SYSTEM_PROMPT_SCORE_DETAIL = """请帮我判断输入文本与输出文本是否满足以下要求:
要求 1. 在保持原文含义不变的前提下，用原文不同的文字进行表达，不额外新增、删减会严重影响语义的内容。
要求 2. 在保证逻辑正确、语序通顺的前提下，变换实体、短语、概念出现的前后顺序、结构关系；顺序变换幅度越小，分数越低。
要求 3. 在保证特殊实体、引用、解释、特定语句不变的前提下，使用同义词替换；同义词替换越少，分数越低；
要求 4. 保持与原文语言风格一致，字数长度变化合理；语言风格差距越大，分数越低。
要求 5. 改写综合评分要非常严格，同时考虑上述 4 个要求；在保证语义一致（要求 1）的前提下，（要求 2,3）变化越大越好，字数越接近越好（要求 4）；如果存在大量长字段完全复制，则给 0 分。
评分准则，每个要求最低 0 分，满分均为 5 分，评分要尽可能严格。"""

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


def make_score_only_detail_sample(input_text: str, output_text: str, score: int) -> Dict:
    user_content = (
        f"原文：\n{input_text}\n\n"
        f"改写：\n{output_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )
    assistant_content = f"该改写的综合评分为{score}分。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SCORE_DETAIL},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def make_score_only_reasoning_sample(
    input_text: str, output_text: str, score: int, rng: np.random.RandomState
) -> Dict:
    user_content = (
        f"原文：\n{input_text}\n\n"
        f"改写：\n{output_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )
    template = rng.choice(REASONING_TEMPLATES[score])
    assistant_content = f"{template}综合评分为{score}分。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SCORE_DETAIL},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def balance_dataset(train_raw: List[Dict], rng: np.random.RandomState) -> List[Dict]:
    by_score = {s: [] for s in range(6)}
    for item in train_raw:
        by_score[item["consensus_score"]].append(item)

    print("Original distribution:")
    for s in range(6):
        print(f"  Score {s}: {len(by_score[s])} samples")

    max_count = max(len(v) for v in by_score.values())
    print(f"\nBalancing to {max_count} samples per class...")

    balanced: List[Dict] = []
    for s in range(6):
        items = by_score[s]
        n_needed = max_count - len(items)
        if n_needed > 0:
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


def save_json(path: Path, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    rng = np.random.RandomState(SEED)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} training samples from: {INPUT_PATH}")

    output_data = []
    for i, item in enumerate(train_data):
        try:
            input_text = item["input"]
            output_text = item["output"]
            score = item["consensus_score"]
        except KeyError as e:
            raise KeyError(f"Missing key {e} in sample index {i}") from e

        output_data.append(
            make_score_only_detail_sample(
                input_text=input_text,
                output_text=output_text,
                score=score,
            )
        )

    balanced = balance_dataset(train_data, rng)

    reasoning_data = [
        make_score_only_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng)
        for d in train_data
    ]
    balanced_detail_data = [
        make_score_only_detail_sample(d["input"], d["output"], d["consensus_score"])
        for d in balanced
    ]
    balanced_reasoning_data = [
        make_score_only_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng)
        for d in balanced
    ]

    save_json(OUTPUT_PATH, output_data)
    save_json(BALANCED_OUTPUT_PATH, balanced_detail_data)
    save_json(REASONING_OUTPUT_PATH, reasoning_data)
    save_json(BALANCED_REASONING_OUTPUT_PATH, balanced_reasoning_data)

    print(f"\nSaved {len(output_data)} samples to: {OUTPUT_PATH}")
    print(f"Saved {len(balanced_detail_data)} samples to: {BALANCED_OUTPUT_PATH}")
    print(f"Saved {len(reasoning_data)} samples to: {REASONING_OUTPUT_PATH}")
    print(f"Saved {len(balanced_reasoning_data)} samples to: {BALANCED_REASONING_OUTPUT_PATH}")

    subsets = [50, 100, 200, 400]
    for size in subsets:
        rng_sub = np.random.RandomState(SEED + size + 1000)

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

        subset_detail = [
            make_score_only_detail_sample(d["input"], d["output"], d["consensus_score"])
            for d in subset
        ]
        subset_reasoning = [
            make_score_only_reasoning_sample(d["input"], d["output"], d["consensus_score"], rng_sub)
            for d in subset
        ]

        subset_detail_path = DATA_DIR / f"train_score_only_detail_balanced_{size}.json"
        subset_reasoning_path = DATA_DIR / f"train_detail_balanced_reasoning_{size}.json"
        save_json(subset_detail_path, subset_detail)
        save_json(subset_reasoning_path, subset_reasoning)
        print(f"Saved {len(subset_detail)} samples to: {subset_detail_path}")
        print(f"Saved {len(subset_reasoning)} samples to: {subset_reasoning_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
