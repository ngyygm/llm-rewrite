#!/usr/bin/env python3
"""
Run Qwen2.5-32B-Instruct (AWQ) as a zero-shot prompt-based evaluator.

Uses a multi-step reasoning approach similar to the original ICLR submission.

EMNLP 2026
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging():
    logger = logging.getLogger("qwen32b_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


SYSTEM_PROMPT = """你是一个专业的中文文本改写质量评估专家。请根据以下维度对中文文本改写质量进行评分（0-5分）：

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


def build_messages(source_text: str, rewrite_text: str) -> list[dict]:
    user_content = (
        f"原文：\n{source_text}\n\n改写：\n{rewrite_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_score(response_text: str) -> int | None:
    # Strategy 1: explicit score mention
    match = re.search(r"评分为\s*([0-5])\s*分", response_text)
    if match:
        return int(match.group(1))
    # Strategy 2: standalone 0-5
    numbers = re.findall(r"(?<![0-9.])([0-5])(?![0-9.])", response_text)
    if numbers:
        return int(numbers[-1])
    return None


def run_inference(model, tokenizer, eval_data, logger, max_new_tokens=512, temperature=0.1):
    model.eval()
    results = []
    parse_failures = 0

    for idx, sample in enumerate(tqdm(eval_data, desc="Inference", leave=False)):
        messages = build_messages(sample["input"], sample["output"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096 - max_new_tokens).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_score = parse_score(response_text)

        if predicted_score is None:
            parse_failures += 1

        results.append({
            "index": idx,
            "response": response_text,
            "predicted_score": predicted_score,
            "consensus_score": sample["consensus_score"],
            "avg_score": sample["avg_score"],
        })

    logger.info(f"Parse failures: {parse_failures}/{len(eval_data)}")
    return results


def compute_metrics(predictions, references):
    from scipy import stats
    preds = np.array(predictions, dtype=float)
    refs = np.array(references, dtype=float)
    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() < 2:
        return {"spearman_rho": 0.0, "pearson_r": 0.0, "kendall_tau": 0.0,
                "mae": float("nan"), "rmse": float("nan"),
                "exact_agreement": 0.0, "within_1": 0.0, "within_2": 0.0,
                "num_valid": 0, "num_total": len(predictions),
                "parse_failures": int(np.isnan(preds).sum())}
    vp, vr = preds[valid_mask], refs[valid_mask]
    spearman, _ = stats.spearmanr(vp, vr)
    pearson, _ = stats.pearsonr(vp, vr)
    kendall, _ = stats.kendalltau(vp, vr)
    abs_err = np.abs(vp - vr)
    return {
        "spearman_rho": round(float(spearman), 4),
        "pearson_r": round(float(pearson), 4),
        "kendall_tau": round(float(kendall), 4),
        "mae": round(float(np.mean(abs_err)), 4),
        "rmse": round(float(np.sqrt(np.mean(abs_err**2))), 4),
        "exact_agreement": round(float(np.mean(vp == vr)), 4),
        "within_1_pct": round(float(np.mean(abs_err <= 1)), 4),
        "within_2_pct": round(float(np.mean(abs_err <= 2)), 4),
        "num_valid": int(valid_mask.sum()),
        "num_total": len(predictions),
        "parse_failures": int((~valid_mask).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/linkco/exa/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--eval_data_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"))
    parser.add_argument("--results_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "baselines" / "results_qwen14b_zeroshot.json"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Qwen2.5-32B Zero-shot Evaluation")
    logger.info("=" * 60)

    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} eval samples")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {args.model_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"  GPU {i}: {mem:.1f} GB total, {alloc:.1f} GB allocated")

    logger.info("Running inference...")
    start = time.time()
    results = run_inference(model, tokenizer, eval_data, logger,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature)
    elapsed = time.time() - start
    logger.info(f"Inference: {len(eval_data)} samples in {elapsed:.1f}s ({elapsed/len(eval_data):.2f}s/sample)")

    predictions = [r["predicted_score"] if r["predicted_score"] is not None else float("nan") for r in results]
    references = [r["avg_score"] for r in results]
    consensus_refs = [r["consensus_score"] for r in results]

    metrics_avg = compute_metrics(predictions, references)
    metrics_consensus = compute_metrics(predictions, consensus_refs)

    logger.info("")
    logger.info("-" * 50)
    logger.info("Results (reference = avg_score):")
    for k, v in metrics_avg.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info("")
    logger.info("Results (reference = consensus_score):")
    for k, v in metrics_consensus.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info("-" * 50)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model_path,
        "metrics_vs_avg_score": metrics_avg,
        "metrics_vs_consensus_score": metrics_consensus,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_path}")

    if args.save_predictions:
        preds_path = results_path.parent / f"{results_path.stem}_predictions.json"
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to: {preds_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
