"""
Convert human annotation data (human_rank1/2/3.json) to unified training format.

Human annotations only have 要求5 (overall score 0-5).
Dimensions 1-4 are only available from LLM evaluators (QWQ-32B multi-reason).

Input: 3 annotator JSON files with 730 samples each
Output: full.json, train.json, eval.json + LoRA training formats
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path("/home/linkco/exa/llm-rewrite/old_rewrite_exp/data/output/human_Rewrite/human_rank")
OUT_DIR = BASE_DIR / "data" / "human_eval"

SEED = 42

SYSTEM_PROMPT_SCORE = """你是一个专业的中文文本改写质量评估专家。请根据以下维度对中文文本改写质量进行评分（0-5分）：

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

请先简要分析，然后给出最终综合评分（0-5分的整数）。"""

SYSTEM_PROMPT_MULTI = """你是一个专业的中文文本改写质量评估专家。请根据以下5个维度对中文文本改写质量进行评分（0-5分）：

1. 语义一致性（要求1）：改写是否保留了原文的核心语义，没有添加、删除或扭曲重要信息。
2. 句式重构（要求2）：改写是否对原文进行了足够的句法结构改变，而非简单替换。
3. 词汇变化（要求3）：改写是否使用了不同的词汇和表达方式。
4. 风格保持（要求4）：改写是否保持了原文的风格特征和合理长度。
5. 综合评分（要求5）：综合以上维度的总体评价。

评分标准：0-5分整数。

请以JSON格式返回：[{{"要求1": "理由", "score": X}}, ..., {{"要求5": "综合理由", "score": Y}}]"""


def load_annotations():
    """Load all 3 annotator files."""
    annotators = []
    for i in range(1, 4):
        path = SRC_DIR / f"human_rank{i}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 730, f"Expected 730 samples in human_rank{i}, got {len(data)}"
        annotators.append(data)
    return annotators


def get_overall_score(sample):
    """Extract the overall score (要求5) from a human annotation sample."""
    for item in sample["score"]:
        if "要求5" in item:
            return item["score"]
    # Fallback: take the first score entry
    return sample["score"][0]["score"]


def make_score_only_sample(input_text, output_text, score):
    """Create score-only training format (primary training format)."""
    user_content = f"原文：\n{input_text}\n\n改写：\n{output_text}\n\n请对该改写进行综合评分（0-5分）。"
    assistant_content = f"该改写的综合评分为{score}分。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SCORE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def make_multi_score_sample(input_text, output_text, score):
    """Create multi-dimension training format (uses same score for all dims since humans only annotated overall)."""
    user_content = (
        f"原文：\n{input_text}\n\n改写：\n{output_text}\n\n"
        f"请按照5个维度（语义一致性、句式重构、词汇变化、风格保持、综合评分）评分（0-5分）。"
    )
    # Since humans only gave overall score, we train the model to produce all dims
    # using the overall score as the target for 要求5 and leave others for the model to learn
    assistant_content = (
        f'[{{"要求1": "分析理由", "score": {score}}}, '
        f'{{"要求2": "分析理由", "score": {score}}}, '
        f'{{"要求3": "分析理由", "score": {score}}}, '
        f'{{"要求4": "分析理由", "score": {score}}}, '
        f'{{"要求5": "综合评价", "score": {score}}}]'
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_MULTI},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def make_eval_format(input_text, output_text, scores_individual, consensus_score, avg_score, std_score):
    """Create evaluation set format."""
    return {
        "input": input_text,
        "output": output_text,
        "annotator_scores": scores_individual,  # [s1, s2, s3]
        "consensus_score": consensus_score,  # rounded mean
        "avg_score": avg_score,  # precise mean
        "std_score": std_score,  # std
    }


def main():
    print("Loading annotations...")
    annotators = load_annotations()
    assert len(annotators) == 3
    assert all(len(a) == 730 for a in annotators)

    # Build unified dataset
    full_data = []
    for idx in range(730):
        samples = [a[idx] for a in annotators]

        input_text = samples[0]["input"]
        output_text = samples[0]["output"]
        assert all(s["input"] == input_text for s in samples), f"Input mismatch at idx {idx}"
        assert all(s["output"] == output_text for s in samples), f"Output mismatch at idx {idx}"

        individual_scores = [get_overall_score(s) for s in samples]
        avg_score = round(np.mean(individual_scores), 3)
        std_score = round(np.std(individual_scores), 3)
        consensus_score = round(np.mean(individual_scores))

        full_data.append(make_eval_format(
            input_text, output_text, individual_scores,
            consensus_score, avg_score, std_score
        ))

    # Compute inter-annotator agreement
    all_s1 = [d["annotator_scores"][0] for d in full_data]
    all_s2 = [d["annotator_scores"][1] for d in full_data]
    all_s3 = [d["annotator_scores"][2] for d in full_data]

    pairs = (all_s1 + all_s2, all_s2 + all_s3)
    r_spearman, _ = stats.spearmanr(all_s1, all_s2)
    print(f"  Annotator 1 vs 2 Spearman: {r_spearman:.4f}")
    r_spearman, _ = stats.spearmanr(all_s1, all_s3)
    print(f"  Annotator 1 vs 3 Spearman: {r_spearman:.4f}")
    r_spearman, _ = stats.spearmanr(all_s2, all_s3)
    print(f"  Annotator 2 vs 3 Spearman: {r_spearman:.4f}")

    # Average pairwise Spearman (reported as "Human Upper Bound" in the paper)
    avg_spearman = np.mean([
        stats.spearmanr(all_s1, all_s2)[0],
        stats.spearmanr(all_s1, all_s3)[0],
        stats.spearmanr(all_s2, all_s3)[0],
    ])
    print(f"  Average pairwise Spearman: {avg_spearman:.4f}")

    # Train/eval split with stratification on consensus score
    rng = np.random.RandomState(SEED)
    consensus_scores = np.array([d["consensus_score"] for d in full_data])

    train_indices = []
    eval_indices = []
    for score_val in range(6):
        idxs = np.where(consensus_scores == score_val)[0]
        rng.shuffle(idxs)
        n_eval = max(1, round(len(idxs) * 130 / 730))
        eval_indices.extend(idxs[:n_eval].tolist())
        train_indices.extend(idxs[n_eval:].tolist())

    # Adjust sizes
    while len(eval_indices) > 130:
        train_indices.append(eval_indices.pop())
    all_idx_set = set(range(730))
    remaining = list(all_idx_set - set(eval_indices) - set(train_indices))
    while len(eval_indices) < 130 and remaining:
        eval_indices.append(remaining.pop(0))
    while len(train_indices) > 600:
        train_indices.pop()
    while len(train_indices) < 600 and remaining:
        train_indices.append(remaining.pop(0))

    print(f"\nSplit: train={len(train_indices)}, eval={len(eval_indices)}")

    train_raw = [full_data[i] for i in train_indices]
    eval_raw = [full_data[i] for i in eval_indices]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "full.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_raw, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_raw, f, ensure_ascii=False, indent=2)

    # Create LoRA training formats (score-only is the primary format)
    train_score_only = [
        make_score_only_sample(d["input"], d["output"], d["consensus_score"])
        for d in train_raw
    ]
    train_multi_score = [
        make_multi_score_sample(d["input"], d["output"], d["consensus_score"])
        for d in train_raw
    ]

    with open(OUT_DIR / "train_score_only.json", "w", encoding="utf-8") as f:
        json.dump(train_score_only, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "train_multi_score.json", "w", encoding="utf-8") as f:
        json.dump(train_multi_score, f, ensure_ascii=False, indent=2)

    # Create subset files for learning curves
    subsets = [50, 100, 200, 400]
    for size in subsets:
        rng_sub = np.random.RandomState(SEED + size)
        idxs = rng_sub.choice(len(train_raw), size=size, replace=False).tolist()
        sub_score = [train_score_only[i] for i in idxs]
        sub_multi = [train_multi_score[i] for i in idxs]
        with open(OUT_DIR / f"train_score_only_{size}.json", "w", encoding="utf-8") as f:
            json.dump(sub_score, f, ensure_ascii=False, indent=2)
        with open(OUT_DIR / f"train_multi_score_{size}.json", "w", encoding="utf-8") as f:
            json.dump(sub_multi, f, ensure_ascii=False, indent=2)

    # Print score distribution
    print("\nScore distribution (consensus):")
    for s in range(6):
        count = sum(1 for d in full_data if d["consensus_score"] == s)
        print(f"  {s}: {count} ({count/730*100:.1f}%)")

    print("\nDone! Files saved to:", OUT_DIR)
    print(f"  full.json: {len(full_data)} samples")
    print(f"  train.json: {len(train_raw)} samples")
    print(f"  eval.json: {len(eval_raw)} samples")
    print(f"  train_score_only.json: {len(train_score_only)} samples")
    print(f"  train_multi_score.json: {len(train_multi_score)} samples")
    for size in subsets:
        print(f"  train_score_only_{size}.json: {size} samples")


if __name__ == "__main__":
    main()
