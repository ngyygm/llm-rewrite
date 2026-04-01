"""
Downstream evaluation for SFT-trained models.

Evaluates rewrite quality on held-out data using:
- Automatic metrics (BLEU, ROUGE, ParaScore)
- LLM-based evaluation (optional)
- Human evaluation (optional, 100 sample subset)
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent


def generate_rewrite(model, tokenizer, source_text: str, max_new_tokens: int = 256) -> str:
    """Generate a rewrite using the trained model."""
    system_prompt = "你是一个专业的中文文本改写助手。请根据用户提供的原文，生成一段高质量的改写文本。改写应保留原文核心语义，使用不同的词汇和句式表达。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请改写以下文本：\n{source_text}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    rewrite = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return rewrite.strip()


def compute_char_bleu(hypothesis: str, reference: str) -> float:
    """Character-level BLEU for Chinese text."""
    from collections import Counter

    hyp_chars = list(hypothesis)
    ref_chars = list(reference)

    if len(hyp_chars) == 0 or len(ref_chars) == 0:
        return 0.0

    hyp_counts = Counter(hyp_chars)
    ref_counts = Counter(ref_chars)

    # Unigram precision
    clipped = sum(min(hyp_counts[c], ref_counts[c]) for c in hyp_counts)
    precision = clipped / len(hyp_chars)

    # Bigram precision
    hyp_bigrams = Counter(zip(hyp_chars[:-1], hyp_chars[1:]))
    ref_bigrams = Counter(zip(ref_chars[:-1], ref_chars[1:]))
    if hyp_bigrams:
        clipped_bi = sum(min(hyp_bigrams[b], ref_bigrams[b]) for b in hyp_bigrams)
        bi_precision = clipped_bi / len(hyp_bigrams)
    else:
        bi_precision = 0.0

    # Combined score
    score = 0.5 * (precision + bi_precision)

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_chars) / len(hyp_chars)))
    return bp * score


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score for Chinese text (character-level)."""
    hyp_chars = list(hypothesis)
    ref_chars = list(reference)

    if len(hyp_chars) == 0 or len(ref_chars) == 0:
        return 0.0

    # LCS length
    m, n = len(hyp_chars), len(ref_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_chars[i - 1] == ref_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / len(hyp_chars)
    recall = lcs / len(ref_chars)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_distinctness(texts: List[str], n: int = 2) -> float:
    """Compute distinct-n metric (vocabulary diversity)."""
    ngrams = []
    for text in texts:
        chars = list(text)
        for i in range(len(chars) - n + 1):
            ngrams.append(tuple(chars[i:i + n]))

    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)


def load_model(base_model_path: str, lora_path: Optional[str] = None):
    """Load base model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    eval_data: List[Dict],
    output_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run evaluation on a set of source texts with reference rewrites."""
    if max_samples:
        eval_data = eval_data[:max_samples]

    results = []
    bleu_scores = []
    rouge_scores = []
    rewrite_lengths = []
    source_lengths = []

    for i, item in enumerate(eval_data):
        source = item.get("source_text", item.get("input", ""))
        reference = item.get("reference", item.get("gold_rewrite", item.get("output", "")))

        # Generate rewrite
        rewrite = generate_rewrite(model, tokenizer, source)

        # Compute metrics
        bleu = compute_char_bleu(rewrite, reference)
        rouge = compute_rouge_l(rewrite, reference)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        rewrite_lengths.append(len(rewrite))
        source_lengths.append(len(source))

        results.append({
            "source": source,
            "reference": reference,
            "rewrite": rewrite,
            "bleu": bleu,
            "rouge_l": rouge,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(eval_data)}")

    # Compute summary metrics
    summary = {
        "num_samples": len(results),
        "avg_bleu": round(np.mean(bleu_scores), 4),
        "std_bleu": round(np.std(bleu_scores), 4),
        "avg_rouge_l": round(np.mean(rouge_scores), 4),
        "std_rouge_l": round(np.std(rouge_scores), 4),
        "avg_rewrite_length": round(np.mean(rewrite_lengths), 1),
        "avg_source_length": round(np.mean(source_lengths), 1),
        "distinct_2": round(compute_distinctness([r["rewrite"] for r in results], 2), 4),
        "distinct_3": round(compute_distinctness([r["rewrite"] for r in results], 3), 4),
        "detailed_results": results,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation for SFT models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--lora_path", type=str, default="",
                        help="Path to LoRA adapter (empty for base model only)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation data JSON")
    parser.add_argument("--output_path", type=str, default="",
                        help="Path to save results")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA adapter: {args.lora_path}")
    model, tokenizer = load_model(args.model_path, args.lora_path or None)

    print(f"Loading eval data: {args.eval_data}")
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"  Total eval samples: {len(eval_data)}")

    max_samples = args.max_samples if args.max_samples > 0 else None
    summary = evaluate_model(
        model, tokenizer, eval_data,
        output_path=args.output_path or None,
        max_samples=max_samples,
    )

    print(f"\n{'='*60}")
    print(f"Downstream Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples: {summary['num_samples']}")
    print(f"  Avg BLEU: {summary['avg_bleu']} ± {summary['std_bleu']}")
    print(f"  Avg ROUGE-L: {summary['avg_rouge_l']} ± {summary['std_rouge_l']}")
    print(f"  Distinct-2: {summary['distinct_2']}")
    print(f"  Distinct-3: {summary['distinct_3']}")
    print(f"  Avg rewrite length: {summary['avg_rewrite_length']}")
    print(f"  Avg source length: {summary['avg_source_length']}")


if __name__ == "__main__":
    main()
