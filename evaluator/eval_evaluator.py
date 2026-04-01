#!/usr/bin/env python3
"""
Evaluation Script for LoRA Fine-tuned Chinese Rewriting Quality Evaluator.

Loads a trained LoRA adapter on top of Qwen2.5-7B-Instruct, runs inference
on the held-out evaluation set (129 samples), parses predicted scores from
model outputs, and computes comprehensive correlation and agreement metrics.

Computed metrics:
    - Spearman rho (rank correlation)
    - Pearson r (linear correlation)
    - Kendall tau (rank correlation)
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Exact agreement (% of predictions matching exactly)
    - +/-1 agreement (% of predictions within 1 point)
    - +/-2 agreement (% of predictions within 2 points)

Usage:
    python eval_evaluator.py \
        --model_path evaluator/checkpoints/score_only_full \
        --eval_data_path data/human_eval/eval.json \
        --mode score_only

    # With custom base model path
    python eval_evaluator.py \
        --model_path evaluator/checkpoints/multi_score_full \
        --eval_data_path data/human_eval/eval.json \
        --base_model /path/to/local/Qwen2.5-7B-Instruct \
        --mode multi_score

EMNLP 2026
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.prompts import build_eval_messages, parse_score_from_response


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to console."""
    logger = logging.getLogger("eval_evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list[int], references: list[float]) -> dict:
    """Compute all evaluation metrics.

    Args:
        predictions: List of predicted integer scores from the model.
        references: List of reference scores (consensus_score or avg_score).

    Returns:
        Dictionary of metric name -> value.
    """
    from scipy import stats

    preds = np.array(predictions, dtype=float)
    refs = np.array(references, dtype=float)

    # Filter out None predictions
    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() == 0:
        return {
            "spearman_rho": 0.0,
            "pearson_r": 0.0,
            "kendall_tau": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "exact_agreement": 0.0,
            "within_1": 0.0,
            "within_2": 0.0,
            "num_valid": 0,
            "num_total": len(predictions),
            "parse_failures": int(np.isnan(preds).sum()),
        }

    valid_preds = preds[valid_mask]
    valid_refs = refs[valid_mask]

    # Correlation metrics
    if len(valid_preds) >= 2:
        spearman_rho, _ = stats.spearmanr(valid_preds, valid_refs)
        pearson_r, _ = stats.pearsonr(valid_preds, valid_refs)
        kendall_tau, _ = stats.kendalltau(valid_preds, valid_refs)
    else:
        spearman_rho = 0.0
        pearson_r = 0.0
        kendall_tau = 0.0

    # Error metrics
    abs_errors = np.abs(valid_preds - valid_refs)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(abs_errors ** 2)))

    # Agreement metrics
    exact = float(np.mean(valid_preds == valid_refs))
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    tokenizer,
    eval_data: list[dict],
    mode: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    logger: logging.Logger,
) -> list[dict]:
    """Run inference on evaluation data.

    Args:
        model: The PEFT model (LoRA on base).
        tokenizer: The tokenizer.
        eval_data: List of eval samples (input/output/consensus_score/...).
        mode: "score_only" or "multi_score".
        temperature: Generation temperature.
        max_new_tokens: Maximum tokens to generate.
        top_p: Top-p (nucleus) sampling.
        repetition_penalty: Repetition penalty.
        logger: Logger instance.

    Returns:
        List of result dicts with prediction info.
    """
    model.eval()

    results = []
    parse_failures = 0

    for idx, sample in enumerate(tqdm(eval_data, desc="Inference", leave=False)):
        source_text = sample["input"]
        rewrite_text = sample["output"]
        consensus_score = sample["consensus_score"]
        avg_score = sample["avg_score"]

        # Build messages
        messages = build_eval_messages(source_text, rewrite_text, mode=mode)

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip input tokens)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse score
        predicted_score = parse_score_from_response(response_text, mode=mode)

        if predicted_score is None:
            parse_failures += 1
            logger.debug(
                f"  [Sample {idx}] Parse failure. Response: {response_text[:200]}"
            )

        results.append({
            "index": idx,
            "source_text": source_text[:200],  # Truncate for storage
            "rewrite_text": rewrite_text[:200],
            "response": response_text,
            "predicted_score": predicted_score,
            "consensus_score": consensus_score,
            "avg_score": avg_score,
            "annotator_scores": sample["annotator_scores"],
        })

    logger.info(f"Parse failures: {parse_failures}/{len(eval_data)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA fine-tuned Chinese rewriting quality evaluator"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--eval_data_path", type=str,
        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"),
        help="Path to eval.json.",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--mode", type=str, default="score_only",
        choices=["score_only", "multi_score"],
        help="Evaluation mode: score_only or multi_score.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature (0.1 = near-deterministic).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling top_p.",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Repetition penalty for generation.",
    )
    parser.add_argument(
        "--results_path", type=str,
        default=str(PROJECT_ROOT / "data" / "baselines" / "results_lora_evaluator.json"),
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save individual predictions to a separate file.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Evaluation: LoRA Fine-tuned Rewriting Quality Evaluator")
    logger.info("=" * 70)
    logger.info(f"Base model      : {args.base_model}")
    logger.info(f"LoRA adapter    : {args.model_path}")
    logger.info(f"Eval data       : {args.eval_data_path}")
    logger.info(f"Mode            : {args.mode}")
    logger.info(f"Temperature     : {args.temperature}")
    logger.info(f"Max new tokens  : {args.max_new_tokens}")
    logger.info(f"Results path    : {args.results_path}")

    # ------------------------------------------------------------------
    # Load eval data
    # ------------------------------------------------------------------
    logger.info("Loading evaluation data...")
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"  Loaded {len(eval_data)} evaluation samples")

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    logger.info("Importing transformers, peft...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # Log GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"  GPU: {gpu_mem:.1f} GB total, {allocated:.1f} GB allocated")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    logger.info("Running inference...")
    start_time = time.time()

    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        logger=logger,
    )

    elapsed = time.time() - start_time
    n_samples = len(eval_data)
    logger.info(f"Inference completed: {n_samples} samples in {elapsed:.1f}s "
                f"({elapsed / n_samples:.2f}s/sample)")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")

    # Extract predictions and references
    predictions = []
    references = []  # Use avg_score as the continuous reference
    consensus_refs = []

    for r in results:
        if r["predicted_score"] is not None:
            predictions.append(r["predicted_score"])
        else:
            predictions.append(float("nan"))
        references.append(r["avg_score"])
        consensus_refs.append(r["consensus_score"])

    # Primary metrics: against avg_score
    metrics_avg = compute_metrics(predictions, references)

    # Secondary metrics: against consensus_score (integer)
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

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": args.base_model,
        "adapter_path": str(args.model_path),
        "mode": args.mode,
        "inference_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "num_samples": n_samples,
        "metrics_vs_avg_score": metrics_avg,
        "metrics_vs_consensus_score": metrics_consensus,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_path}")

    # Optionally save individual predictions
    if args.save_predictions:
        predictions_path = Path(args.model_path) / "eval_predictions.json"
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to: {predictions_path}")

    logger.info("Done.")

    # Return non-zero exit code if parse failure rate is too high
    if metrics_avg["parse_failures"] > n_samples * 0.1:
        logger.error(
            f"WARNING: {metrics_avg['parse_failures']} parse failures "
            f"({metrics_avg['parse_failures']/n_samples:.1%}). "
            f"Score parsing may need tuning."
        )


if __name__ == "__main__":
    main()
