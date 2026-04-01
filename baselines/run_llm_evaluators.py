"""
LLM-based evaluator baselines for Chinese text rewriting evaluation.

Implements:
- G-Eval style evaluation with Qwen2.5-7B (chain-of-thought scoring)
- Zero-shot Qwen2.5-7B with direct prompting

Uses local inference via HuggingFace transformers with 4-bit quantization.

Usage:
    python baselines/run_llm_evaluators.py --model Qwen/Qwen2.5-7B-Instruct
    python baselines/run_llm_evaluators.py --skip-geval  # only zero-shot
"""

import json
import re
import sys
import torch
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from baselines.correlation_utils import (
    load_eval_data,
    compute_correlations,
    per_score_level_analysis,
    print_correlation_table,
    print_level_analysis,
    save_results,
)

EVAL_PATH = BASE_DIR / "data" / "human_eval" / "eval.json"
RESULTS_DIR = BASE_DIR / "data" / "baselines"


# ============================================================
# Prompt templates
# ============================================================

SYSTEM_PROMPT_EVAL = """你是一个专业的中文文本改写质量评估专家。请根据以下维度对中文文本改写质量进行评分（0-5分）：

评分维度：
1. 语义一致性：改写是否保留了原文的核心语义，没有添加、删除或扭曲重要信息。
2. 句式重构：改写是否对原文进行了足够的句法结构改变，而非简单替换。
3. 词汇变化：改写是否使用了不同的词汇和表达方式。
4. 风格保持：改写是否保持了原文的风格特征和合理长度。

评分标准：
- 0分：改写完全失败（严重语义扭曲或毫无改写）
- 1分：改写质量很差
- 2分：改写质量较差
- 3分：改写质量一般
- 4分：改写质量较好
- 5分：改写质量优秀"""


# G-Eval step-by-step evaluation steps prompt
GEVAL_STEPS_PROMPT = """请按照以下步骤评估这段改写的质量：

步骤1：分析改写文本与原文的语义一致性。检查是否有重要信息的丢失、添加或扭曲。
步骤2：评估句式重构程度。改写是否改变了句法结构，还是仅仅做了简单的词语替换。
步骤3：检查词汇变化。改写是否使用了不同的词汇和表达方式。
步骤4：评估风格保持。改写是否保持了原文的风格特征和合理的文本长度。
步骤5：综合以上分析，给出最终评分（0-5分的整数）。

请严格按照以下格式输出：
[步骤1分析内容]
[步骤2分析内容]
[步骤3分析内容]
[步骤4分析内容]
最终评分：X"""


USER_PROMPT_TEMPLATE = """原文：
{input_text}

改写：
{output_text}

请对该改写进行综合评分（0-5分的整数）。"""


# Zero-shot direct prompt (simpler, no chain-of-thought)
ZERO_SHOT_USER_PROMPT = """原文：
{input_text}

改写：
{output_text}

请直接给出该改写的综合评分（0-5分的整数），只需输出一个数字，不要有任何其他文字。"""


# ============================================================
# Score extraction
# ============================================================

def extract_score_from_response(response: str) -> Optional[float]:
    """Extract a score (0-5) from the model's response.

    Looks for patterns like "最终评分：X", "评分：X", "X分", or a standalone digit.
    """
    if not response:
        return None

    # Remove whitespace
    text = response.strip()

    # Pattern 1: "最终评分：X" or "最终评分: X" or "最终评分为X"
    patterns = [
        r'最终评分[：:为]?\s*(\d)',
        r'综合评分[：:为]?\s*(\d)',
        r'评分[：:为]?\s*(\d)',
        r'[评分]是?\s*(\d)\s*分',
        r'[最终综合]?[评分][：:为]?\s*(\d)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 5:
                return float(score)

    # Pattern 2: Last standalone digit in the response
    # Look for "X分" at the end
    match = re.search(r'(\d)\s*分?\s*$', text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 5:
            return float(score)

    # Pattern 3: Find any digit 0-5 that appears near scoring keywords
    match = re.search(r'(\d)\s*(?:[/／]?\s*5)?(?:分)?', text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 5:
            return float(score)

    # Pattern 4: If the response is just a number
    text_clean = text.strip().strip('.')
    try:
        score = int(float(text_clean))
        if 0 <= score <= 5:
            return float(score)
    except (ValueError, TypeError):
        pass

    return None


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> Tuple:
    """Load a HuggingFace model with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier or local path.
        load_in_4bit: Use 4-bit quantization for memory efficiency.
        device_map: Device mapping strategy.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading model: {model_name} ...")
    print(f"  4-bit quantization: {load_in_4bit}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model.eval()
    print(f"  Model loaded successfully.")

    return model, tokenizer


# ============================================================
# G-Eval style evaluation
# ============================================================

def run_geval(
    model,
    tokenizer,
    input_text: str,
    output_text: str,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
) -> Tuple[Optional[float], str]:
    """Run G-Eval style chain-of-thought evaluation.

    First generates evaluation steps, then extracts the score.
    Uses lower temperature for more deterministic scoring.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        input_text: Original Chinese text.
        output_text: Rewritten Chinese text.
        temperature: Sampling temperature for generation.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (extracted_score, raw_response).
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        input_text=input_text,
        output_text=output_text,
    )

    # Build chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVAL + "\n\n" + GEVAL_STEPS_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        text = f"系统：{SYSTEM_PROMPT_EVAL}\n\n{GEVAL_STEPS_PROMPT}\n\n用户：{user_content}\n\n助手："

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    score = extract_score_from_response(response)
    return score, response


# ============================================================
# Zero-shot evaluation
# ============================================================

def run_zero_shot(
    model,
    tokenizer,
    input_text: str,
    output_text: str,
    temperature: float = 0.1,
    max_new_tokens: int = 128,
) -> Tuple[Optional[float], str]:
    """Run zero-shot direct scoring evaluation.

    Asks the model to output only a score (0-5) without explanation.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        input_text: Original Chinese text.
        output_text: Rewritten Chinese text.
        temperature: Sampling temperature.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (extracted_score, raw_response).
    """
    user_content = ZERO_SHOT_USER_PROMPT.format(
        input_text=input_text,
        output_text=output_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVAL},
        {"role": "user", "content": user_content},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = f"系统：{SYSTEM_PROMPT_EVAL}\n\n用户：{user_content}\n\n助手："

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    score = extract_score_from_response(response)
    return score, response


# ============================================================
# Batch evaluation
# ============================================================

def evaluate_dataset(
    model,
    tokenizer,
    data: List[Dict],
    eval_type: str = "zero_shot",
    temperature: float = 0.1,
    batch_size: int = 1,
) -> Dict:
    """Run evaluation on the full dataset.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        data: Evaluation dataset.
        eval_type: "geval" or "zero_shot".
        temperature: Generation temperature.
        batch_size: Not used (sequential processing), kept for API consistency.

    Returns:
        Dict with predictions, raw responses, and metadata.
    """
    n = len(data)
    gt_scores = [d["consensus_score"] for d in data]

    predictions = []
    raw_responses = []
    parse_failures = 0

    eval_fn = run_geval if eval_type == "geval" else run_zero_shot

    max_tokens = 512 if eval_type == "geval" else 128

    for i in range(n):
        input_text = data[i]["input"]
        output_text = data[i]["output"]

        try:
            score, response = eval_fn(
                model, tokenizer,
                input_text, output_text,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
        except Exception as e:
            print(f"    ERROR at sample {i}: {e}")
            score = None
            response = ""

        predictions.append(score)
        raw_responses.append(response)

        if score is None:
            parse_failures += 1

        # Progress
        if (i + 1) % 10 == 0 or i == n - 1:
            parsed = sum(1 for p in predictions if p is not None)
            print(
                f"    Progress: {i + 1}/{n}  "
                f"(parsed: {parsed}, failed: {parse_failures})"
            )

    # Fill in None predictions with median for correlation computation
    valid_scores = [s for s in predictions if s is not None]
    if valid_scores:
        fill_value = float(np.median(valid_scores))
    else:
        fill_value = 3.0

    filled_predictions = [
        p if p is not None else fill_value for p in predictions
    ]

    return {
        "predictions": predictions,
        "filled_predictions": filled_predictions,
        "raw_responses": raw_responses,
        "parse_failures": parse_failures,
        "fill_value": fill_value,
    }


# ============================================================
# Main runner
# ============================================================

def run_llm_evaluator(
    model_name: str,
    data: List[Dict],
    eval_type: str = "zero_shot",
    temperature: float = 0.1,
    load_in_4bit: bool = True,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run a single LLM evaluator on the dataset.

    Args:
        model_name: HuggingFace model name.
        data: Evaluation dataset.
        eval_type: "geval" or "zero_shot".
        temperature: Generation temperature.
        load_in_4bit: Use 4-bit quantization.
        max_samples: Limit evaluation to N samples (for debugging).

    Returns:
        Results dict with correlations and per-sample data.
    """
    method_name = f"{eval_type}_{model_name.split('/')[-1]}"

    if max_samples:
        eval_data = data[:max_samples]
    else:
        eval_data = data

    print(f"\n{'='*60}")
    print(f"  Running: {method_name}")
    print(f"  Samples: {len(eval_data)}")
    print(f"  Eval type: {eval_type}")
    print(f"  Temperature: {temperature}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name, load_in_4bit=load_in_4bit
    )

    # Evaluate
    print(f"\n  Running {eval_type} evaluation ...")
    eval_results = evaluate_dataset(
        model, tokenizer, eval_data,
        eval_type=eval_type,
        temperature=temperature,
    )

    # Free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Compute correlations
    gt_scores = [d["consensus_score"] for d in eval_data]
    correlations = compute_correlations(
        eval_results["filled_predictions"],
        gt_scores,
        method_name,
        round_predictions=True,
    )

    print(f"\n  {print_correlation_table(correlations)}")

    # Per-score-level analysis
    level_analysis = per_score_level_analysis(
        eval_results["filled_predictions"],
        gt_scores,
        method_name,
    )
    print(print_level_analysis(level_analysis))

    # Save sample-level results
    sample_results = []
    for i, d in enumerate(eval_data):
        sample_results.append({
            "idx": i,
            "consensus_score": d["consensus_score"],
            "predicted_score": eval_results["predictions"][i],
            "raw_response": eval_results["raw_responses"][i][:300],
        })

    # Full results
    results = {
        "method": method_name,
        "model_name": model_name,
        "eval_type": eval_type,
        "temperature": temperature,
        "n_samples": len(eval_data),
        "parse_failures": eval_results["parse_failures"],
        "correlations": correlations,
        "level_analysis": level_analysis,
        "sample_results": sample_results,
    }

    # Save
    filename = f"llm_{method_name}.json"
    out_path = save_results(results, filename)
    print(f"  Results saved to: {out_path}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluator baselines"
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--eval-path", type=str, default=str(EVAL_PATH),
        help="Path to eval.json"
    )
    parser.add_argument(
        "--eval-types", nargs="+",
        default=["zero_shot", "geval"],
        choices=["zero_shot", "geval"],
        help="Which evaluation types to run"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to N samples (for debugging)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LLM-based Evaluator Baselines")
    print("=" * 60)

    data = load_eval_data(args.eval_path)

    all_results = []
    for eval_type in args.eval_types:
        result = run_llm_evaluator(
            model_name=args.model,
            data=data,
            eval_type=eval_type,
            temperature=args.temperature,
            load_in_4bit=not args.no_4bit,
            max_samples=args.max_samples,
        )
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in all_results:
        corr = r["correlations"]
        print(
            f"  {r['method']:<40s}  "
            f"Spearman={corr.get('spearman_rho', 0):+.4f}  "
            f"MAE={corr.get('mae', 0):.3f}  "
            f"Exact={corr.get('exact_agreement_pct', 0):.1f}%"
        )


if __name__ == "__main__":
    main()
