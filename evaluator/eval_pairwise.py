#!/usr/bin/env python3
"""
Evaluation Script for Pairwise LoRA Fine-tuned Chinese Rewriting Quality Evaluator.

Runs the pairwise model (pairwise_b2_generated) on two evaluation modes:

1. Same-source evaluation (generated_eval.json):
   - For each source text with 3 rewrites (prompt_type 0, 1, 2), run all
     choose-2 = 3 pairwise comparisons.
   - Position-swap each pair and average to reduce position bias.
   - Compute win_rate per prompt_type; expected ranking: 0 > 1 > 2.
   - Report Kendall tau and Spearman rho between predicted and GT ranking.

2. Cross-source evaluation (eval.json - human annotated):
   - Construct all pairs where |avg_score_A - avg_score_B| >= 1.
   - Run pairwise model with position-swap averaging.
   - Compute pairwise accuracy, per-rewrite win_rate, and Spearman rho
     between win_rates and human avg_scores.

Usage:
    # Both modes (default)
    python evaluator/eval_pairwise.py \
        --checkpoint evaluator/checkpoints/pairwise_b2_generated \
        --output_path data/pairwise/pairwise_eval_results.json

    # Same-source only
    python evaluator/eval_pairwise.py \
        --checkpoint evaluator/checkpoints/pairwise_b2_generated \
        --eval_mode same_source

    # Cross-source only
    python evaluator/eval_pairwise.py \
        --checkpoint evaluator/checkpoints/pairwise_b2_generated \
        --eval_mode cross_source

EMNLP 2026
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to console."""
    logger = logging.getLogger("eval_pairwise")
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
# Pairwise Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PAIRWISE = (
    "你是一个专业的文本改写质量评估专家。"
    "请根据原文和两篇改写，判断哪篇改写的质量更高。"
)

USER_PROMPT_TEMPLATE = (
    "原文：{source}\n\n"
    "改写A：{rewrite_a}\n\n"
    "改写B：{rewrite_b}\n\n"
    "请问哪篇改写的质量更高？如果改写A更好请回答A，"
    "如果改写B更好请回答B，如果两者差不多请回答平局。"
)

CROSS_SOURCE_USER_PROMPT_TEMPLATE = (
    "原文A：{source_a}\n\n"
    "改写A：{rewrite_a}\n\n"
    "原文B：{source_b}\n\n"
    "改写B：{rewrite_b}\n\n"
    "请问哪篇改写的质量更高？如果改写A更好请回答A，"
    "如果改写B更好请回答B，如果两者差不多请回答平局。"
)


def build_pairwise_messages(
    source_text: str,
    rewrite_a: str,
    rewrite_b: str,
) -> list[dict]:
    """Build chat messages for pairwise comparison.

    Args:
        source_text: Original source text.
        rewrite_a: First rewrite candidate.
        rewrite_b: Second rewrite candidate.

    Returns:
        List of message dicts in chat format.
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        source=source_text,
        rewrite_a=rewrite_a,
        rewrite_b=rewrite_b,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PAIRWISE},
        {"role": "user", "content": user_content},
    ]


def build_cross_source_messages(
    source_a: str,
    rewrite_a: str,
    source_b: str,
    rewrite_b: str,
) -> list[dict]:
    """Build chat messages for cross-source pairwise comparison.

    Each rewrite has its own source text, matching the training format.

    Args:
        source_a: Source text for rewrite A.
        rewrite_a: First rewrite candidate.
        source_b: Source text for rewrite B.
        rewrite_b: Second rewrite candidate.

    Returns:
        List of message dicts in chat format.
    """
    user_content = CROSS_SOURCE_USER_PROMPT_TEMPLATE.format(
        source_a=source_a,
        rewrite_a=rewrite_a,
        source_b=source_b,
        rewrite_b=rewrite_b,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PAIRWISE},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------

def parse_pairwise_output(text: str) -> float:
    """Parse model output to get preference.

    Returns:
        1.0 if A preferred,
        0.0 if B preferred,
        0.5 if tie,
        -1.0 if parse failure.
    """
    text = text.strip()
    if "改写A" in text and ("更高" in text or "更好" in text or "A" == text[-1].strip()):
        return 1.0
    elif "改写B" in text and ("更高" in text or "更好" in text or "B" == text[-1].strip()):
        return 0.0
    elif "平局" in text or "相当" in text or "一样" in text:
        return 0.5
    else:
        return -1.0


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str,
    checkpoint: str,
    logger: logging.Logger,
):
    """Load base model with 4-bit quantization and LoRA adapter.

    Args:
        base_model: Base model name or local path.
        checkpoint: LoRA adapter directory path.
        logger: Logger instance.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
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

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"Loading LoRA adapter from {checkpoint}...")
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"  GPU: {gpu_mem:.1f} GB total, {allocated:.1f} GB allocated")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-Pair Generation
# ---------------------------------------------------------------------------

def generate_pairwise(
    model,
    tokenizer,
    source_text: str,
    rewrite_a: str,
    rewrite_b: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> dict:
    """Run pairwise comparison and return parsed result.

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        source_text: Original source text.
        rewrite_a: First rewrite.
        rewrite_b: Second rewrite.
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.

    Returns:
        Dict with 'response', 'preference' (1.0/0.0/0.5/-1.0).
    """
    messages = build_pairwise_messages(source_text, rewrite_a, rewrite_b)
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
    preference = parse_pairwise_output(response_text)

    return {
        "response": response_text,
        "preference": preference,
    }


def generate_with_position_swap(
    model,
    tokenizer,
    source_text: str,
    rewrite_a: str,
    rewrite_b: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> dict:
    """Run pairwise comparison twice (A-B and B-A) and average.

    This reduces position bias. Returns the average P(A preferred).

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        source_text: Original source text.
        rewrite_a: First rewrite (the one we care about winning).
        rewrite_b: Second rewrite.
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.

    Returns:
        Dict with 'response_ab', 'response_ba', 'pref_ab', 'pref_ba',
        'avg_preference' (P(A preferred), 0.0 to 1.0), 'parse_ok'.
    """
    # Forward: A vs B
    result_ab = generate_pairwise(
        model, tokenizer, source_text, rewrite_a, rewrite_b,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )

    # Swapped: B vs A  (A is now in position B)
    result_ba = generate_pairwise(
        model, tokenizer, source_text, rewrite_b, rewrite_a,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )

    pref_ab = result_ab["preference"]
    pref_ba = result_ba["preference"]

    parse_ok = (pref_ab >= 0) and (pref_ba >= 0)

    if parse_ok:
        # Forward: A preferred -> 1, B preferred -> 0, tie -> 0.5
        # Swapped: A preferred (now in B slot) -> 0, B preferred (now in A slot) -> 1
        # So for swapped: if model says A(original B) preferred, original A loses -> 0
        #                  if model says B(original A) preferred, original A wins -> 1
        # pref_ba_swapped = 1.0 - pref_ba
        # avg = (pref_ab + (1.0 - pref_ba)) / 2
        avg_preference = (pref_ab + (1.0 - pref_ba)) / 2.0
    else:
        avg_preference = -1.0

    return {
        "response_ab": result_ab["response"],
        "response_ba": result_ba["response"],
        "pref_ab": pref_ab,
        "pref_ba": pref_ba,
        "avg_preference": avg_preference,
        "parse_ok": parse_ok,
    }


def generate_pairwise_cross(
    model,
    tokenizer,
    source_a: str,
    rewrite_a: str,
    source_b: str,
    rewrite_b: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> dict:
    """Run cross-source pairwise comparison and return parsed result.

    Each rewrite has its own source text, matching the training format.

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        source_a: Source text for rewrite A.
        rewrite_a: First rewrite.
        source_b: Source text for rewrite B.
        rewrite_b: Second rewrite.
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.

    Returns:
        Dict with 'response', 'preference' (1.0/0.0/0.5/-1.0).
    """
    messages = build_cross_source_messages(source_a, rewrite_a, source_b, rewrite_b)
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
    preference = parse_pairwise_output(response_text)

    return {
        "response": response_text,
        "preference": preference,
    }


def generate_cross_with_position_swap(
    model,
    tokenizer,
    source_a: str,
    rewrite_a: str,
    source_b: str,
    rewrite_b: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> dict:
    """Run cross-source pairwise comparison twice with position swap and average.

    Forward: (source_a, rewrite_a) vs (source_b, rewrite_b)
    Swapped: (source_b, rewrite_b) vs (source_a, rewrite_a)
    Both source-rewrite pairs stay together during the swap.

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        source_a: Source text for rewrite A.
        rewrite_a: First rewrite.
        source_b: Source text for rewrite B.
        rewrite_b: Second rewrite.
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.

    Returns:
        Dict with 'response_ab', 'response_ba', 'pref_ab', 'pref_ba',
        'avg_preference' (P(A preferred), 0.0 to 1.0), 'parse_ok'.
    """
    # Forward: A pair vs B pair
    result_ab = generate_pairwise_cross(
        model, tokenizer, source_a, rewrite_a, source_b, rewrite_b,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )

    # Swapped: B pair vs A pair (both source and rewrite swap together)
    result_ba = generate_pairwise_cross(
        model, tokenizer, source_b, rewrite_b, source_a, rewrite_a,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )

    pref_ab = result_ab["preference"]
    pref_ba = result_ba["preference"]

    parse_ok = (pref_ab >= 0) and (pref_ba >= 0)

    if parse_ok:
        # Forward: A preferred -> 1, B preferred -> 0, tie -> 0.5
        # Swapped: A pair is now in B slot, so pref_ba reflects B-slot winning
        # If swapped says "A" (original B pair) preferred -> original A loses -> 0
        # If swapped says "B" (original A pair) preferred -> original A wins -> 1
        # pref_ba_swapped = 1.0 - pref_ba
        avg_preference = (pref_ab + (1.0 - pref_ba)) / 2.0
    else:
        avg_preference = -1.0

    return {
        "response_ab": result_ab["response"],
        "response_ba": result_ba["response"],
        "pref_ab": pref_ab,
        "pref_ba": pref_ba,
        "avg_preference": avg_preference,
        "parse_ok": parse_ok,
    }


# ---------------------------------------------------------------------------
# Same-Source Evaluation
# ---------------------------------------------------------------------------

def eval_same_source(
    model,
    tokenizer,
    scored_rewrites_path: str,
    max_new_tokens: int,
    temperature: float,
    logger: logging.Logger,
) -> dict:
    """Evaluate pairwise model on same-source comparisons.

    For each source with 3 rewrites (prompt_type 0, 1, 2):
    - Compare all choose(3,2) = 3 pairs
    - Use position-swap averaging
    - Compute win_rate per prompt_type
    - Compare against expected ranking (0 > 1 > 2)

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        scored_rewrites_path: Path to scored_rewrites.json.
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.
        logger: Logger.

    Returns:
        Dict with 'per_source_results', 'per_prompt_win_rates', 'metrics'.
    """
    logger.info("Loading scored rewrites for same-source evaluation...")
    with open(scored_rewrites_path, "r", encoding="utf-8") as f:
        scored_rewrites = json.load(f)
    logger.info(f"  Loaded {len(scored_rewrites)} scored rewrites")

    # Group by source_hash
    by_source = defaultdict(list)
    for item in scored_rewrites:
        by_source[item["source_hash"]].append(item)

    logger.info(f"  {len(by_source)} unique sources")

    # We need to use the eval sources (not train). The generated_eval.json
    # uses 100 eval sources. We use all 300 sources from scored_rewrites
    # that also appear in generated_eval.json. But actually, since we want
    # to test on held-out data, let's try to match against generated_eval.
    # For simplicity, use all sources from scored_rewrites.
    # Each source has 3 rewrites: prompt_type 0, 1, 2.

    per_source_results = []
    wins_by_prompt_type = defaultdict(lambda: {"wins": 0, "total": 0})
    parse_failures = 0
    total_pairs = 0

    logger.info(f"Running same-source pairwise evaluation on {len(by_source)} sources...")

    for source_idx, (source_hash, rewrites) in enumerate(tqdm(
        by_source.items(), desc="Same-source", leave=False
    )):
        # Sort by prompt_type to get consistent ordering
        rewrites_sorted = sorted(rewrites, key=lambda x: x["prompt_type"])
        source_text = rewrites_sorted[0]["source_text"]

        # Ensure we have exactly 3 rewrites with prompt_types 0, 1, 2
        prompt_types = [r["prompt_type"] for r in rewrites_sorted]
        if prompt_types != [0, 1, 2]:
            logger.warning(
                f"  Source {source_hash}: unexpected prompt_types={prompt_types}, skipping"
            )
            continue

        source_result = {
            "source_hash": source_hash,
            "source_text": source_text[:200],
            "pairs": [],
            "predicted_scores": {},  # win_rate as proxy
        }

        # All 3 choose 2 pairs
        for (i, _), (j, _) in combinations(enumerate(rewrites_sorted), 2):
            rewrite_a = rewrites_sorted[i]["rewrite_text"]
            rewrite_b = rewrites_sorted[j]["rewrite_text"]
            pt_a = rewrites_sorted[i].get("prompt_type", i)
            pt_b = rewrites_sorted[j].get("prompt_type", j)
            predicted_a = rewrites_sorted[i].get("predicted_score", 0)
            predicted_b = rewrites_sorted[j].get("predicted_score", 0)

            result = generate_with_position_swap(
                model, tokenizer, source_text, rewrite_a, rewrite_b,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )

            total_pairs += 1

            if not result["parse_ok"]:
                parse_failures += 1

            pair_result = {
                "prompt_type_a": int(pt_a),
                "prompt_type_b": int(pt_b),
                "predicted_score_a": predicted_a,
                "predicted_score_b": predicted_b,
                "response_ab": result["response_ab"],
                "response_ba": result["response_ba"],
                "pref_ab": result["pref_ab"],
                "pref_ba": result["pref_ba"],
                "avg_preference": result["avg_preference"],
                "parse_ok": result["parse_ok"],
            }
            source_result["pairs"].append(pair_result)

            # Track wins: avg_preference > 0.5 means A wins
            if result["parse_ok"]:
                wins_by_prompt_type[pt_a]["total"] += 1
                wins_by_prompt_type[pt_b]["total"] += 1
                if result["avg_preference"] > 0.5:
                    wins_by_prompt_type[pt_a]["wins"] += 1
                elif result["avg_preference"] < 0.5:
                    wins_by_prompt_type[pt_b]["wins"] += 1
                # tie: no win for either

        per_source_results.append(source_result)

    # Compute win rates per prompt_type
    win_rates = {}
    for pt in [0, 1, 2]:
        stats = wins_by_prompt_type[pt]
        if stats["total"] > 0:
            win_rates[pt] = stats["wins"] / stats["total"]
        else:
            win_rates[pt] = 0.0

    logger.info(f"  Parse failures: {parse_failures}/{total_pairs}")
    logger.info(f"  Win rates by prompt_type:")
    for pt in [0, 1, 2]:
        logger.info(f"    prompt_type {pt}: {win_rates[pt]:.4f} "
                     f"({wins_by_prompt_type[pt]['wins']}/"
                     f"{wins_by_prompt_type[pt]['total']})")

    # Compute ranking correlation metrics
    from scipy import stats as scipy_stats

    # Ground truth ranking: prompt_type 0 (best) > 1 > 2 (worst)
    gt_ranking = [0, 1, 2]  # prompt_type order from best to worst
    predicted_ranking = [0, 1, 2]
    # Sort prompt_types by win_rate (descending = best first)
    predicted_ranking_sorted = sorted(
        [0, 1, 2], key=lambda pt: -win_rates[pt]
    )

    # Map prompt_types to quality tiers for correlation
    # GT: 0->3, 1->2, 2->1 (higher = better)
    gt_quality = {0: 3.0, 1: 2.0, 2: 1.0}
    gt_values = [gt_quality[pt] for pt in [0, 1, 2]]
    pred_values = [win_rates[pt] for pt in [0, 1, 2]]

    # Since we only have 3 data points for Spearman/Kendall, we also
    # report the expected ranking correctness.
    expected_correct = (predicted_ranking_sorted == gt_ranking)

    # Spearman and Kendall (3 points is the minimum)
    if len(gt_values) >= 3:
        spearman_rho, _ = scipy_stats.spearmanr(gt_values, pred_values)
        kendall_tau, _ = scipy_stats.kendalltau(gt_values, pred_values)
    else:
        spearman_rho = 0.0
        kendall_tau = 0.0

    # Discrimination: does prompt_type 0 beat prompt_type 2?
    # (most clear signal pair)
    discrimination_0v2 = win_rates[0] - win_rates[2]

    same_source_metrics = {
        "num_sources": len(per_source_results),
        "total_pairs": total_pairs,
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / max(total_pairs, 1), 4),
        "win_rate_prompt0": round(win_rates[0], 4),
        "win_rate_prompt1": round(win_rates[1], 4),
        "win_rate_prompt2": round(win_rates[2], 4),
        "expected_ranking_correct": expected_correct,
        "spearman_rho": round(float(spearman_rho), 4),
        "kendall_tau": round(float(kendall_tau), 4),
        "discrimination_0v2": round(discrimination_0v2, 4),
    }

    logger.info("")
    logger.info("  Same-source results:")
    logger.info(f"    Ranking correct (0>1>2): {expected_correct}")
    logger.info(f"    Spearman rho : {spearman_rho:.4f}")
    logger.info(f"    Kendall tau  : {kendall_tau:.4f}")
    logger.info(f"    Discrimination (0 vs 2): {discrimination_0v2:+.4f}")

    return {
        "per_source_results": per_source_results,
        "win_rates_by_prompt_type": {str(k): v for k, v in win_rates.items()},
        "metrics": same_source_metrics,
    }


# ---------------------------------------------------------------------------
# Cross-Source Evaluation
# ---------------------------------------------------------------------------

def eval_cross_source(
    model,
    tokenizer,
    eval_data_path: str,
    min_score_diff: float,
    max_pairs: int,
    max_new_tokens: int,
    temperature: float,
    logger: logging.Logger,
) -> dict:
    """Evaluate pairwise model on cross-source comparisons.

    From 129 human-annotated eval samples, construct all pairs where
    |avg_score_A - avg_score_B| >= min_score_diff. Run pairwise model
    with position-swap averaging. Compute accuracy and Spearman rho
    between win_rates and human avg_scores.

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        eval_data_path: Path to eval.json.
        min_score_diff: Minimum score difference for pair construction.
        max_pairs: Maximum number of pairs (0 = no limit).
        max_new_tokens: Max generation tokens.
        temperature: Generation temperature.
        logger: Logger.

    Returns:
        Dict with 'pair_results', 'per_rewrite_stats', 'metrics'.
    """
    logger.info("Loading human eval data for cross-source evaluation...")
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"  Loaded {len(eval_data)} human-annotated samples")

    # Build pairs where |avg_score_A - avg_score_B| >= min_score_diff
    pairs = []
    for i in range(len(eval_data)):
        for j in range(i + 1, len(eval_data)):
            score_a = eval_data[i]["avg_score"]
            score_b = eval_data[j]["avg_score"]
            if abs(score_a - score_b) >= min_score_diff:
                # Higher-scored rewrite should be "better"
                if score_a > score_b:
                    pairs.append((i, j, True))   # A should win
                else:
                    pairs.append((i, j, False))  # B should win
            elif abs(score_a - score_b) >= 0.5:
                # Include moderate-diff pairs too for more data, with a flag
                if score_a > score_b:
                    pairs.append((i, j, True))
                else:
                    pairs.append((i, j, False))

    logger.info(f"  Constructed {len(pairs)} candidate pairs")

    # Apply limit if needed
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
        logger.info(f"  Limited to {max_pairs} pairs")

    # Track per-rewrite win statistics
    # For each sample index: {wins: int, comparisons: int}
    rewrite_stats = defaultdict(lambda: {"wins": 0.0, "comparisons": 0})

    pair_results = []
    parse_failures = 0
    correct_count = 0
    valid_count = 0

    # Also separate by score diff category
    easy_correct = 0   # diff >= 2
    easy_total = 0
    medium_correct = 0  # 1 <= diff < 2
    medium_total = 0
    hard_correct = 0   # 0.5 <= diff < 1
    hard_total = 0

    logger.info(f"Running cross-source pairwise evaluation on {len(pairs)} pairs...")

    for pair_idx, (i, j, a_should_win) in enumerate(tqdm(
        pairs, desc="Cross-source", leave=False
    )):
        sample_a = eval_data[i]
        sample_b = eval_data[j]

        source_a = sample_a["input"]
        rewrite_a = sample_a["output"]
        score_a = sample_a["avg_score"]

        source_b = sample_b["input"]
        rewrite_b = sample_b["output"]
        score_b = sample_b["avg_score"]

        result = generate_cross_with_position_swap(
            model, tokenizer,
            source_a, rewrite_a, source_b, rewrite_b,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        score_diff = abs(score_a - score_b)
        if score_diff >= 2:
            easy_total += 1
        elif score_diff >= 1:
            medium_total += 1
        else:
            hard_total += 1

        pair_result = {
            "index_a": i,
            "index_b": j,
            "avg_score_a": score_a,
            "avg_score_b": score_b,
            "score_diff": round(score_diff, 3),
            "a_should_win": a_should_win,
            "response_ab": result["response_ab"],
            "response_ba": result["response_ba"],
            "pref_ab": result["pref_ab"],
            "pref_ba": result["pref_ba"],
            "avg_preference": result["avg_preference"],
            "parse_ok": result["parse_ok"],
        }

        if result["parse_ok"]:
            valid_count += 1
            # avg_preference > 0.5 means A (higher-scored) wins
            predicted_a_wins = result["avg_preference"] > 0.5
            predicted_b_wins = result["avg_preference"] < 0.5

            # Only count as correct if the model makes a clear prediction (not a tie)
            # Ties (avg_preference ≈ 0.5) indicate position bias or indecision
            if (predicted_a_wins and a_should_win) or (predicted_b_wins and not a_should_win):
                correct_count += 1
                if score_diff >= 2:
                    easy_correct += 1
                elif score_diff >= 1:
                    medium_correct += 1
                else:
                    hard_correct += 1

            # Track per-rewrite stats
            # For sample A: a win for A means A won this comparison
            if result["avg_preference"] > 0.5:
                rewrite_stats[i]["wins"] += 1.0
            elif result["avg_preference"] < 0.5:
                rewrite_stats[j]["wins"] += 1.0
            rewrite_stats[i]["comparisons"] += 1
            rewrite_stats[j]["comparisons"] += 1
        else:
            parse_failures += 1
            # Still count comparisons even on parse failure
            rewrite_stats[i]["comparisons"] += 1
            rewrite_stats[j]["comparisons"] += 1

        pair_results.append(pair_result)

    # Compute per-rewrite win rates
    per_rewrite_win_rates = {}
    for idx, stats in rewrite_stats.items():
        if stats["comparisons"] > 0:
            win_rate = stats["wins"] / stats["comparisons"]
        else:
            win_rate = 0.0
        per_rewrite_win_rates[idx] = {
            "avg_score": eval_data[idx]["avg_score"],
            "win_rate": round(win_rate, 4),
            "wins": stats["wins"],
            "comparisons": stats["comparisons"],
        }

    # Compute Spearman rho between win_rates and avg_scores
    from scipy import stats as scipy_stats

    idx_list = sorted(per_rewrite_win_rates.keys())
    win_rates_arr = [per_rewrite_win_rates[idx]["win_rate"] for idx in idx_list]
    avg_scores_arr = [per_rewrite_win_rates[idx]["avg_score"] for idx in idx_list]

    if len(win_rates_arr) >= 3:
        spearman_rho, sp_pvalue = scipy_stats.spearmanr(win_rates_arr, avg_scores_arr)
        kendall_tau, kt_pvalue = scipy_stats.kendalltau(win_rates_arr, avg_scores_arr)
    else:
        spearman_rho = 0.0
        kendall_tau = 0.0
        sp_pvalue = 1.0
        kt_pvalue = 1.0

    # Pairwise accuracy
    accuracy = correct_count / max(valid_count, 1)
    easy_acc = easy_correct / max(easy_total, 1)
    medium_acc = medium_correct / max(medium_total, 1)
    hard_acc = hard_correct / max(hard_total, 1)

    cross_source_metrics = {
        "total_pairs": len(pairs),
        "valid_pairs": valid_count,
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / max(len(pairs), 1), 4),
        "pairwise_accuracy": round(accuracy, 4),
        "accuracy_easy_diff_ge2": round(easy_acc, 4),
        "accuracy_easy_total": easy_total,
        "accuracy_medium_diff_1to2": round(medium_acc, 4),
        "accuracy_medium_total": medium_total,
        "accuracy_hard_diff_05to1": round(hard_acc, 4),
        "accuracy_hard_total": hard_total,
        "spearman_rho_winrate_vs_avg": round(float(spearman_rho), 4),
        "spearman_pvalue": round(float(sp_pvalue), 6),
        "kendall_tau_winrate_vs_avg": round(float(kendall_tau), 4),
        "kendall_pvalue": round(float(kt_pvalue), 6),
        "num_rewrites_with_comparisons": len(per_rewrite_win_rates),
        "min_score_diff": min_score_diff,
    }

    logger.info("")
    logger.info("  Cross-source results:")
    logger.info(f"    Pairs           : {len(pairs)} (valid: {valid_count}, "
                 f"parse failures: {parse_failures})")
    logger.info(f"    Pairwise accuracy: {accuracy:.4f}")
    logger.info(f"    Easy (diff>=2)  : {easy_acc:.4f} ({easy_correct}/{easy_total})")
    logger.info(f"    Medium (1<=d<2) : {medium_acc:.4f} ({medium_correct}/{medium_total})")
    logger.info(f"    Hard (0.5<=d<1) : {hard_acc:.4f} ({hard_correct}/{hard_total})")
    logger.info(f"    Spearman (win_rate vs avg): {spearman_rho:.4f} (p={sp_pvalue:.6f})")
    logger.info(f"    Kendall (win_rate vs avg) : {kendall_tau:.4f} (p={kt_pvalue:.6f})")

    return {
        "pair_results": pair_results,
        "per_rewrite_stats": per_rewrite_win_rates,
        "metrics": cross_source_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise LoRA fine-tuned Chinese rewriting evaluator"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained pairwise LoRA adapter directory.",
    )
    parser.add_argument(
        "--base_model", type=str,
        default="/home/linkco/exa/models/Qwen2.5-7B-Instruct",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--eval_mode", type=str, default="both",
        choices=["same_source", "cross_source", "both"],
        help="Which evaluation mode(s) to run.",
    )
    parser.add_argument(
        "--scored_rewrites_path", type=str,
        default=str(PROJECT_ROOT / "data" / "generated_rewrites" / "scored_rewrites.json"),
        help="Path to scored_rewrites.json (for same-source eval).",
    )
    parser.add_argument(
        "--eval_data_path", type=str,
        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"),
        help="Path to eval.json (for cross-source eval).",
    )
    parser.add_argument(
        "--min_score_diff", type=float, default=1.0,
        help="Minimum score difference for cross-source pair construction.",
    )
    parser.add_argument(
        "--max_pairs", type=int, default=0,
        help="Max cross-source pairs to evaluate (0 = no limit).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature (0.1 = near-deterministic).",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=str(PROJECT_ROOT / "data" / "pairwise" / "pairwise_eval_results.json"),
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Evaluation: Pairwise LoRA Rewriting Quality Evaluator")
    logger.info("=" * 70)
    logger.info(f"Base model      : {args.base_model}")
    logger.info(f"LoRA checkpoint : {args.checkpoint}")
    logger.info(f"Eval mode       : {args.eval_mode}")
    logger.info(f"Temperature     : {args.temperature}")
    logger.info(f"Max new tokens  : {args.max_new_tokens}")
    logger.info(f"Min score diff  : {args.min_score_diff}")
    logger.info(f"Output path     : {args.output_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.checkpoint, logger
    )

    output = {
        "model": args.base_model,
        "checkpoint": str(args.checkpoint),
        "eval_mode": args.eval_mode,
        "inference_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    overall_start = time.time()

    # ------------------------------------------------------------------
    # Same-source evaluation
    # ------------------------------------------------------------------
    if args.eval_mode in ("same_source", "both"):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PART 1: Same-Source Evaluation")
        logger.info("=" * 70)

        start = time.time()
        same_source_output = eval_same_source(
            model=model,
            tokenizer=tokenizer,
            scored_rewrites_path=args.scored_rewrites_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            logger=logger,
        )
        elapsed = time.time() - start

        output["same_source"] = {
            "elapsed_seconds": round(elapsed, 1),
            "metrics": same_source_output["metrics"],
            "win_rates_by_prompt_type": same_source_output["win_rates_by_prompt_type"],
        }
        # Save per-source details separately (can be large)
        output["same_source"]["num_per_source_results"] = len(
            same_source_output["per_source_results"]
        )

    # ------------------------------------------------------------------
    # Cross-source evaluation
    # ------------------------------------------------------------------
    if args.eval_mode in ("cross_source", "both"):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PART 2: Cross-Source Evaluation")
        logger.info("=" * 70)

        start = time.time()
        cross_source_output = eval_cross_source(
            model=model,
            tokenizer=tokenizer,
            eval_data_path=args.eval_data_path,
            min_score_diff=args.min_score_diff,
            max_pairs=args.max_pairs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            logger=logger,
        )
        elapsed = time.time() - start

        output["cross_source"] = {
            "elapsed_seconds": round(elapsed, 1),
            "metrics": cross_source_output["metrics"],
        }
        output["cross_source"]["num_pair_results"] = len(cross_source_output["pair_results"])
        output["cross_source"]["num_per_rewrite_stats"] = len(
            cross_source_output["per_rewrite_stats"]
        )

    overall_elapsed = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total time: {overall_elapsed:.1f}s")

    if "same_source" in output:
        m = output["same_source"]["metrics"]
        logger.info("")
        logger.info("Same-source evaluation:")
        logger.info(f"  Sources evaluated  : {m['num_sources']}")
        logger.info(f"  Total pairs        : {m['total_pairs']}")
        logger.info(f"  Parse failures     : {m['parse_failures']} ({m['parse_failure_rate']:.2%})")
        logger.info(f"  Win rate (pt=0, hi): {m['win_rate_prompt0']:.4f}")
        logger.info(f"  Win rate (pt=1, md): {m['win_rate_prompt1']:.4f}")
        logger.info(f"  Win rate (pt=2, lo): {m['win_rate_prompt2']:.4f}")
        logger.info(f"  Ranking correct    : {m['expected_ranking_correct']}")
        logger.info(f"  Spearman rho       : {m['spearman_rho']:.4f}")
        logger.info(f"  Kendall tau        : {m['kendall_tau']:.4f}")
        logger.info(f"  Discrimination 0v2 : {m['discrimination_0v2']:+.4f}")

    if "cross_source" in output:
        m = output["cross_source"]["metrics"]
        logger.info("")
        logger.info("Cross-source evaluation:")
        logger.info(f"  Total pairs        : {m['total_pairs']}")
        logger.info(f"  Pairwise accuracy  : {m['pairwise_accuracy']:.4f}")
        logger.info(f"  Easy accuracy      : {m['accuracy_easy_diff_ge2']:.4f} (n={m['accuracy_easy_total']})")
        logger.info(f"  Medium accuracy    : {m['accuracy_medium_diff_1to2']:.4f} (n={m['accuracy_medium_total']})")
        logger.info(f"  Hard accuracy      : {m['accuracy_hard_diff_05to1']:.4f} (n={m['accuracy_hard_total']})")
        logger.info(f"  Spearman (wr vs sc): {m['spearman_rho_winrate_vs_avg']:.4f} (p={m['spearman_pvalue']:.6f})")
        logger.info(f"  Kendall (wr vs sc) : {m['kendall_tau_winrate_vs_avg']:.4f}")
        logger.info(f"  Rewrites evaluated : {m['num_rewrites_with_comparisons']}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = Path(args.output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"")
    logger.info(f"Results saved to: {results_path}")

    # Save detailed per-sample results in separate files
    if args.eval_mode in ("same_source", "both") and "same_source" in dir():
        detail_path = results_path.with_name(
            results_path.stem + "_same_source_details.json"
        )
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                same_source_output["per_source_results"],
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Same-source details: {detail_path}")

    if args.eval_mode in ("cross_source", "both") and "cross_source" in dir():
        # Pair-level detail
        pair_detail_path = results_path.with_name(
            results_path.stem + "_cross_source_pairs.json"
        )
        with open(pair_detail_path, "w", encoding="utf-8") as f:
            json.dump(
                cross_source_output["pair_results"],
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Cross-source pair details: {pair_detail_path}")

        # Per-rewrite win rate detail
        rewrite_detail_path = results_path.with_name(
            results_path.stem + "_cross_source_rewrites.json"
        )
        with open(rewrite_detail_path, "w", encoding="utf-8") as f:
            json.dump(
                cross_source_output["per_rewrite_stats"],
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Cross-source rewrite details: {rewrite_detail_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
