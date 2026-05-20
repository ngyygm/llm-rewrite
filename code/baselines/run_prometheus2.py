#!/usr/bin/env python3
"""
Run Prometheus 2 (prometheus-7b-v2.0) as a baseline evaluator on the Chinese
rewriting quality benchmark.

Uses absolute grading without reference answer, with a custom rubric
adapted for Chinese text rewriting evaluation.

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
    logger = logging.getLogger("prometheus2_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


REWRITE_RUBRIC = """[Chinese Text Rewriting Quality]
Score 1: The rewrite completely fails. Severe semantic distortion, no structural change, or essentially identical to the original text.
Score 2: The rewrite has poor quality. Significant loss of original meaning, minimal structural modification, or excessive word-for-word copying.
Score 3: The rewrite has acceptable quality. Retains most of the original meaning but lacks sufficient structural or lexical variation.
Score 4: The rewrite has good quality. Effectively preserves the original meaning while introducing meaningful structural and lexical changes.
Score 5: The rewrite has excellent quality. Fully preserves the original meaning with creative and natural structural reformulation and diverse vocabulary."""

ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, "
    "objective feedback based on specific criteria, ensuring each "
    "assessment reflects the absolute standards set for performance."
)

ABS_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.

1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.

2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.

3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"

4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback:"""


def build_prometheus_messages(source_text: str, rewrite_text: str) -> list[dict]:
    """Build messages for Prometheus 2 evaluation."""
    instruction = (
        f"请对以下中文文本进行改写，要求保留原文核心语义，同时进行句式和词汇的变化。\n\n"
        f"原文：{source_text}"
    )
    response = rewrite_text

    user_content = ABS_PROMPT_WO_REF.format(
        instruction=instruction,
        response=response,
        rubric=REWRITE_RUBRIC,
    )

    return [
        {"role": "system", "content": ABS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_prometheus_score(response_text: str) -> int | None:
    """Parse score from Prometheus 2 output format: '... [RESULT] N'."""
    # Strategy 1: Look for [RESULT] pattern
    match = re.search(r"\[RESULT\]\s*([1-5])", response_text)
    if match:
        return int(match.group(1))

    # Strategy 2: Find standalone integer 1-5 (last occurrence)
    numbers = re.findall(r"(?<![0-9.])([1-5])(?![0-9.])", response_text)
    if numbers:
        return int(numbers[-1])

    return None


def run_inference(model, tokenizer, eval_data, logger, max_new_tokens=512, temperature=0.1):
    """Run Prometheus 2 inference on eval data."""
    model.eval()
    results = []
    parse_failures = 0

    for idx, sample in enumerate(tqdm(eval_data, desc="Inference", leave=False)):
        source_text = sample["input"]
        rewrite_text = sample["output"]

        messages = build_prometheus_messages(source_text, rewrite_text)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted_score = parse_prometheus_score(response_text)

        if predicted_score is None:
            parse_failures += 1
            logger.debug(f"  [Sample {idx}] Parse failure: {response_text[:200]}")

        results.append({
            "index": idx,
            "response": response_text,
            "predicted_score": predicted_score,
            "consensus_score": sample["consensus_score"],
            "avg_score": sample["avg_score"],
            "annotator_scores": sample["annotator_scores"],
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

    vp = preds[valid_mask]
    vr = refs[valid_mask]

    spearman_rho, _ = stats.spearmanr(vp, vr)
    pearson_r, _ = stats.pearsonr(vp, vr)
    kendall_tau, _ = stats.kendalltau(vp, vr)

    abs_errors = np.abs(vp - vr)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(abs_errors ** 2)))
    exact = float(np.mean(vp == vr))
    within_1 = float(np.mean(abs_errors <= 1))
    within_2 = float(np.mean(abs_errors <= 2))

    return {
        "spearman_rho": round(float(spearman_rho), 4),
        "pearson_r": round(float(pearson_r), 4),
        "kendall_tau": round(float(kendall_tau), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "exact_agreement": round(exact, 4),
        "within_1_pct": round(within_1, 4),
        "within_2_pct": round(within_2, 4),
        "num_valid": int(valid_mask.sum()),
        "num_total": len(predictions),
        "parse_failures": int((~valid_mask).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Prometheus 2 baseline")
    parser.add_argument("--model_path", type=str,
                        default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--eval_data_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"))
    parser.add_argument("--results_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "baselines" / "results_prometheus2.json"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Prometheus 2 Baseline Evaluation")
    logger.info("=" * 60)

    # Load eval data
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} eval samples")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU: {gpu_mem:.1f} GB total, {allocated:.1f} GB allocated")

    # Run inference
    logger.info("Running inference...")
    start = time.time()
    results = run_inference(model, tokenizer, eval_data, logger,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature)
    elapsed = time.time() - start
    logger.info(f"Inference: {len(eval_data)} samples in {elapsed:.1f}s ({elapsed/len(eval_data):.2f}s/sample)")

    # Compute metrics
    predictions = [r["predicted_score"] if r["predicted_score"] is not None else float("nan")
                   for r in results]
    references = [r["avg_score"] for r in results]
    consensus_refs = [r["consensus_score"] for r in results]

    metrics_avg = compute_metrics(predictions, references)
    metrics_consensus = compute_metrics(predictions, consensus_refs)

    logger.info("")
    logger.info("-" * 50)
    logger.info("Results (reference = avg_score):")
    logger.info(f"  Spearman rho     : {metrics_avg['spearman_rho']:.4f}")
    logger.info(f"  Pearson r        : {metrics_avg['pearson_r']:.4f}")
    logger.info(f"  Kendall tau      : {metrics_avg['kendall_tau']:.4f}")
    logger.info(f"  MAE              : {metrics_avg['mae']:.4f}")
    logger.info(f"  RMSE             : {metrics_avg['rmse']:.4f}")
    logger.info(f"  Exact agreement  : {metrics_avg['exact_agreement']:.2%}")
    logger.info(f"  Within +/-1      : {metrics_avg['within_1_pct']:.2%}")
    logger.info(f"  Within +/-2      : {metrics_avg['within_2_pct']:.2%}")
    logger.info(f"  Valid predictions: {metrics_avg['num_valid']}/{metrics_avg['num_total']}")
    logger.info(f"  Parse failures   : {metrics_avg['parse_failures']}")
    logger.info("")
    logger.info("Results (reference = consensus_score):")
    logger.info(f"  Spearman rho     : {metrics_consensus['spearman_rho']:.4f}")
    logger.info(f"  Pearson r        : {metrics_consensus['pearson_r']:.4f}")
    logger.info(f"  Kendall tau      : {metrics_consensus['kendall_tau']:.4f}")
    logger.info(f"  MAE              : {metrics_consensus['mae']:.4f}")
    logger.info(f"  RMSE             : {metrics_consensus['rmse']:.4f}")
    logger.info(f"  Exact agreement  : {metrics_consensus['exact_agreement']:.2%}")
    logger.info(f"  Within +/-1      : {metrics_consensus['within_1_pct']:.2%}")
    logger.info(f"  Within +/-2      : {metrics_consensus['within_2_pct']:.2%}")
    logger.info("-" * 50)

    # Save results
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "prometheus-eval/prometheus-7b-v2.0",
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
