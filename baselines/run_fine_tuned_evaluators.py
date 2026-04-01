"""
Fine-tuned evaluator baselines for Chinese text rewriting evaluation.

Implements inference with:
- Prometheus 2 (prometheus-eval/prometheus-7b-v2.0)
- M-Prometheus (prometheus-eval/prometheus-7b-v2.0, multilingual mode)

These models are specifically designed for evaluation and use a particular
prompt format with score rubrics and reference answers.

Uses local inference via HuggingFace transformers with 4-bit quantization.

Usage:
    python baselines/run_fine_tuned_evaluators.py
    python baselines/run_fine_tuned_evaluators.py --model prometheus-eval/prometheus-7b-v2.0
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

DEFAULT_MODEL = "prometheus-eval/prometheus-7b-v2.0"


# ============================================================
# Prometheus prompt format
# ============================================================

# The score rubric for Chinese text rewriting quality (0-5 scale)
SCORE_RUBRIC_ZH = """\
改写质量评分标准：
- 0分：改写完全失败。严重语义扭曲或完全未进行改写，改写结果不可用。
- 1分：改写质量很差。存在严重的语义偏差，改写程度极低，基本不可用。
- 2分：改写质量较差。语义基本保留但有明显偏差，改写程度不足，可用性低。
- 3分：改写质量一般。语义基本一致，进行了一定程度的改写，但改进空间较大。
- 4分：改写质量较好。语义一致性好，改写程度较高，词汇和句式有明显变化。
- 5分：改写质量优秀。语义完全一致，改写彻底且自然，词汇丰富，句式多样，风格一致。"""

# Reference answer (instruction for what constitutes good rewriting)
REFERENCE_ANSWER_ZH = """\
优秀的中文文本改写应该：1) 完整保留原文的核心语义和信息要点；2) 在句法结构上做出实质性改变，而非简单的同义词替换；3) 使用不同的词汇和表达方式；4) 保持原文的风格特征和合理的文本长度。"""


def build_prometheus_prompt(
    input_text: str,
    output_text: str,
    rubric: str = SCORE_RUBRIC_ZH,
    reference_answer: str = REFERENCE_ANSWER_ZH,
) -> str:
    """Build the prompt in Prometheus format.

    Prometheus expects a specific format:
    ###Task Description:
    [instruction]

    ###The score rubric:
    [rubric]

    ###Reference Answer:
    [reference]

    ###The text to evaluate:
    [text to score]

    ###The scored text:
    [candidate text]
    """
    prompt = f"""###Task Description:
请评估以下中文文本改写的质量，根据评分标准给出0-5分的整数评分。

###The score rubric:
{rubric}

###Reference Answer:
{reference_answer}

###The text to evaluate:
{input_text}

###The scored text:
{output_text}"""
    return prompt


# ============================================================
# Score extraction
# ============================================================

def extract_prometheus_score(response: str) -> Optional[float]:
    """Extract score from Prometheus model response.

    Prometheus typically outputs in the format:
    "The score is X" or "得分：X" or similar patterns.
    """
    if not response:
        return None

    text = response.strip()

    # Pattern: "score is X" / "Score: X" / "score of X"
    patterns = [
        r'(?:score|Score|SCORE)\s*(?:is|:|：|=|为|of)\s*(\d)',
        r'(?:得分|评分|分数|最终评分)[：:为]?\s*(\d)',
        r'(\d)\s*(?:[/／]\s*5)?\s*(?:分)?\s*\.?\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 5:
                return float(score)

    # Find the last occurrence of a digit 0-5 in the text
    all_matches = re.findall(r'\b([0-5])\b', text)
    if all_matches:
        return float(int(all_matches[-1]))

    return None


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = True,
) -> Tuple:
    """Load Prometheus model with 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier.
        load_in_4bit: Use 4-bit quantization.

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
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model.eval()
    print("  Model loaded successfully.")
    return model, tokenizer


# ============================================================
# Inference
# ============================================================

def evaluate_single(
    model,
    tokenizer,
    input_text: str,
    output_text: str,
    rubric: str = SCORE_RUBRIC_ZH,
    reference_answer: str = REFERENCE_ANSWER_ZH,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> Tuple[Optional[float], str]:
    """Evaluate a single input/output pair with Prometheus.

    Args:
        model: The Prometheus model.
        tokenizer: The tokenizer.
        input_text: Original text.
        output_text: Rewritten text.
        rubric: Score rubric text.
        reference_answer: Reference answer for the evaluation task.
        max_new_tokens: Maximum generation tokens.
        temperature: Generation temperature.

    Returns:
        Tuple of (extracted_score, raw_response).
    """
    prompt = build_prometheus_prompt(
        input_text, output_text, rubric, reference_answer
    )

    # Prometheus uses a simple instruction format, not chat template
    # Format: "###Instruction:\n{prompt}\n###Response:\n"
    full_prompt = f"###Instruction:\n{prompt}\n\n###Response:\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

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

    score = extract_prometheus_score(response)
    return score, response


def evaluate_dataset(
    model,
    tokenizer,
    data: List[Dict],
    rubric: str = SCORE_RUBRIC_ZH,
    reference_answer: str = REFERENCE_ANSWER_ZH,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> Dict:
    """Run Prometheus evaluation on the full dataset.

    Args:
        model: The Prometheus model.
        tokenizer: The tokenizer.
        data: Evaluation dataset.
        rubric: Score rubric.
        reference_answer: Reference answer.
        max_new_tokens: Maximum generation tokens.
        temperature: Generation temperature.

    Returns:
        Dict with predictions, raw responses, and metadata.
    """
    n = len(data)
    predictions = []
    raw_responses = []
    parse_failures = 0

    for i in range(n):
        try:
            score, response = evaluate_single(
                model, tokenizer,
                data[i]["input"],
                data[i]["output"],
                rubric=rubric,
                reference_answer=reference_answer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"    ERROR at sample {i}: {e}")
            score = None
            response = ""

        predictions.append(score)
        raw_responses.append(response)

        if score is None:
            parse_failures += 1

        if (i + 1) % 10 == 0 or i == n - 1:
            parsed = sum(1 for p in predictions if p is not None)
            print(
                f"    Progress: {i + 1}/{n}  "
                f"(parsed: {parsed}, failed: {parse_failures})"
            )

    # Fill None values with median
    valid_scores = [s for s in predictions if s is not None]
    fill_value = float(np.median(valid_scores)) if valid_scores else 3.0

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

def run_prometheus_evaluator(
    model_name: str,
    data: List[Dict],
    method_label: str = "",
    rubric: str = SCORE_RUBRIC_ZH,
    reference_answer: str = REFERENCE_ANSWER_ZH,
    temperature: float = 0.1,
    load_in_4bit: bool = True,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run Prometheus-based evaluation.

    Args:
        model_name: HuggingFace model name.
        data: Evaluation dataset.
        method_label: Custom label for this run.
        rubric: Score rubric text.
        reference_answer: Reference answer.
        temperature: Generation temperature.
        load_in_4bit: Use 4-bit quantization.
        max_samples: Limit to N samples.

    Returns:
        Full results dict.
    """
    if not method_label:
        method_label = model_name.split("/")[-1]

    eval_data = data[:max_samples] if max_samples else data

    print(f"\n{'='*60}")
    print(f"  Running: {method_label}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {len(eval_data)}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name, load_in_4bit=load_in_4bit
    )

    # Evaluate
    print(f"\n  Running evaluation ...")
    eval_results = evaluate_dataset(
        model, tokenizer, eval_data,
        rubric=rubric,
        reference_answer=reference_answer,
        temperature=temperature,
    )

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Correlations
    gt_scores = [d["consensus_score"] for d in eval_data]
    correlations = compute_correlations(
        eval_results["filled_predictions"],
        gt_scores,
        method_label,
        round_predictions=True,
    )

    print(f"\n  {print_correlation_table(correlations)}")

    # Per-level analysis
    level_analysis = per_score_level_analysis(
        eval_results["filled_predictions"],
        gt_scores,
        method_label,
    )
    print(print_level_analysis(level_analysis))

    # Sample results
    sample_results = []
    for i, d in enumerate(eval_data):
        sample_results.append({
            "idx": i,
            "consensus_score": d["consensus_score"],
            "predicted_score": eval_results["predictions"][i],
            "raw_response": eval_results["raw_responses"][i][:300],
        })

    results = {
        "method": method_label,
        "model_name": model_name,
        "temperature": temperature,
        "n_samples": len(eval_data),
        "parse_failures": eval_results["parse_failures"],
        "correlations": correlations,
        "level_analysis": level_analysis,
        "sample_results": sample_results,
    }

    filename = f"finetuned_{method_label}.json"
    out_path = save_results(results, filename)
    print(f"  Results saved to: {out_path}")

    return results


def run_all_fine_tuned(
    data: List[Dict],
    temperature: float = 0.1,
    load_in_4bit: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Run all fine-tuned evaluator baselines.

    Runs both Prometheus 2 and M-Prometheus (multilingual mode).

    Args:
        data: Evaluation dataset.
        temperature: Generation temperature.
        load_in_4bit: Use 4-bit quantization.
        max_samples: Limit to N samples.

    Returns:
        List of result dicts.
    """
    all_results = []

    # Prometheus 2 (English rubric - default mode)
    # Prometheus 2 is designed for English but can evaluate Chinese text
    # when given appropriate rubric
    print("\n" + "=" * 60)
    print("Fine-tuned Evaluator: Prometheus 2")
    print("=" * 60)

    result_prometheus = run_prometheus_evaluator(
        model_name=DEFAULT_MODEL,
        data=data,
        method_label="Prometheus-2-7B",
        rubric=SCORE_RUBRIC_ZH,
        reference_answer=REFERENCE_ANSWER_ZH,
        temperature=temperature,
        load_in_4bit=load_in_4bit,
        max_samples=max_samples,
    )
    all_results.append(result_prometheus)

    # M-Prometheus (multilingual mode)
    # M-Prometheus uses the same model but with a multilingual rubric
    # The key difference is the instruction to respond in Chinese
    print("\n" + "=" * 60)
    print("Fine-tuned Evaluator: M-Prometheus")
    print("=" * 60)

    mprompt_reference = REFERENCE_ANSWER_ZH + "\n\n请用中文进行评估并给出评分。"

    result_mprometheus = run_prometheus_evaluator(
        model_name=DEFAULT_MODEL,
        data=data,
        method_label="M-Prometheus-7B",
        rubric=SCORE_RUBRIC_ZH,
        reference_answer=mprompt_reference,
        temperature=temperature,
        load_in_4bit=load_in_4bit,
        max_samples=max_samples,
    )
    all_results.append(result_mprometheus)

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fine-tuned evaluator baselines (Prometheus 2, M-Prometheus)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--eval-path", type=str, default=str(EVAL_PATH),
        help="Path to eval.json"
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
    parser.add_argument(
        "--single", action="store_true",
        help="Run only the specified model (not both Prometheus and M-Prometheus)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Fine-tuned Evaluator Baselines")
    print("=" * 60)

    data = load_eval_data(args.eval_path)

    if args.single:
        results = [
            run_prometheus_evaluator(
                model_name=args.model,
                data=data,
                temperature=args.temperature,
                load_in_4bit=not args.no_4bit,
                max_samples=args.max_samples,
            )
        ]
    else:
        results = run_all_fine_tuned(
            data,
            temperature=args.temperature,
            load_in_4bit=not args.no_4bit,
            max_samples=args.max_samples,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        corr = r["correlations"]
        print(
            f"  {r['method']:<25s}  "
            f"Spearman={corr.get('spearman_rho', 0):+.4f}  "
            f"Kendall={corr.get('kendall_tau', 0):+.4f}  "
            f"Pearson={corr.get('pearson_r', 0):+.4f}  "
            f"MAE={corr.get('mae', 0):.3f}  "
            f"Exact={corr.get('exact_agreement_pct', 0):.1f}%"
        )


if __name__ == "__main__":
    main()
