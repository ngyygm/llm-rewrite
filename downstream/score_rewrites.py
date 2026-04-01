#!/usr/bin/env python3
"""
Score generated rewrites using the LoRA fine-tuned evaluator.

Loads the best LoRA model (balanced_simple) and scores all rewrite pairs.

EMNLP 2026
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated_rewrites"


def main():
    parser = argparse.ArgumentParser(description="Score rewrites with LoRA evaluator")
    parser.add_argument("--rewrites_path", type=str, default=str(GENERATED_DIR / "all_rewrites.json"))
    parser.add_argument("--evaluator_path", type=str,
                        default=str(PROJECT_ROOT / "evaluator" / "checkpoints" / "balanced_simple"))
    parser.add_argument("--base_model", type=str, default="/home/linkco/exa/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_path", type=str, default=str(GENERATED_DIR / "scored_rewrites.json"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)  # Sequential for now
    args = parser.parse_args()

    # Load rewrites
    with open(args.rewrites_path, "r", encoding="utf-8") as f:
        rewrites = json.load(f)
    print(f"Loaded {len(rewrites)} rewrites")

    # Load evaluator
    sys_path = str(PROJECT_ROOT)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from evaluator.prompts import build_eval_messages, parse_score_from_response

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model with 4-bit quantization...")
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

    print(f"Loading LoRA adapter from {args.evaluator_path}...")
    model = PeftModel.from_pretrained(base_model, args.evaluator_path)
    model.eval()

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU: {mem:.1f} GB total, {alloc:.1f} GB allocated")

    # Score each rewrite
    results = []
    parse_failures = 0
    start = time.time()

    for idx, item in enumerate(tqdm(rewrites, desc="Scoring")):
        source_text = item["source_text"]
        rewrite_text = item["rewrite_text"]

        messages = build_eval_messages(source_text, rewrite_text, mode="score_only")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048 - args.max_new_tokens).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_score = parse_score_from_response(response_text, mode="score_only")

        if predicted_score is None:
            parse_failures += 1

        results.append({
            "index": idx,
            "source_text": source_text[:200],
            "rewrite_text": rewrite_text[:200],
            "response": response_text,
            "predicted_score": predicted_score,
            "source_hash": item.get("source_hash", ""),
            "prompt_type": item.get("prompt_type", ""),
        })

    elapsed = time.time() - start
    print(f"\nScored {len(rewrites)} rewrites in {elapsed:.1f}s ({elapsed/len(rewrites):.2f}s/sample)")
    print(f"Parse failures: {parse_failures}/{len(rewrites)}")

    # Score distribution
    valid_scores = [r["predicted_score"] for r in results if r["predicted_score"] is not None]
    if valid_scores:
        from collections import Counter
        dist = Counter(valid_scores)
        print(f"\nScore distribution:")
        for s in range(6):
            print(f"  Score {s}: {dist.get(s, 0)} ({dist.get(s, 0)/len(valid_scores)*100:.1f}%)")
        print(f"  Mean: {np.mean(valid_scores):.2f}, Std: {np.std(valid_scores):.2f}")

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
