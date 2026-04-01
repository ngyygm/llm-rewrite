#!/usr/bin/env python3
"""
Generate rewrites using a batched approach - all variations in one prompt.

Much faster than individual calls: ~3s per source vs ~50s per source.

EMNLP 2026
"""

import argparse
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated_rewrites"

REWRITE_BATCH_PROMPT = """请对以下文本进行3次不同质量的改写，用"===改写1==="、"===改写2==="、"===改写3==="分隔。

改写1要求：高质量改写，完整保留原文语义，使用完全不同的词汇和句式。
改写2要求：中等质量改写，保留大部分原意，有一些词汇和句式变化。
改写3要求：低质量改写，仅做少量词语替换，基本保持原句结构。

原文：{text}

请直接输出3个改写结果，用分隔符隔开，不要输出其他内容。"""


def generate_rewrites_batched(model, tokenizer, source_texts, output_dir):
    """Generate all rewrites using batched single-call approach."""
    all_rewrites = []

    print(f"Generating rewrites for {len(source_texts)} source texts (batched)...")
    pbar = tqdm(total=len(source_texts), desc="Rewrites")

    for idx, source in enumerate(source_texts):
        source_text = source["text"]

        prompt = REWRITE_BATCH_PROMPT.format(text=source_text)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse the 3 rewrites
        parts = response.split("===改写")
        rewrites_found = []
        for part in parts:
            # Remove header like "1==="
            if "===" in part:
                part = part.split("===", 1)[1] if "===" in part[1:] else part
                part = part.split("===")[0] if "===" in part else part
            part = part.strip()
            if len(part) >= 10:
                rewrites_found.append(part)

        for ri, rewrite in enumerate(rewrites_found[:3]):
            quality = ["high", "medium", "low"][ri] if ri < 3 else "unknown"
            all_rewrites.append({
                "source_text": source_text,
                "source_hash": source["hash"],
                "rewrite_text": rewrite,
                "prompt_type": ri,
                "quality_level": quality,
            })

        pbar.update(1)

        # Save progress every 50
        if (idx + 1) % 50 == 0:
            progress_path = output_dir / "rewrites_progress.json"
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(all_rewrites, f, ensure_ascii=False, indent=2)
            elapsed = pbar.format_dict.get("elapsed", 0)
            rate = (idx + 1) / max(elapsed, 1) * len(rewrites_found) if elapsed else 0
            print(f"\n  Progress: {len(all_rewrites)} rewrites, ~{rate:.1f} rewrites/min")

    return all_rewrites


def main():
    parser = argparse.ArgumentParser(description="Generate rewrites (batched)")
    parser.add_argument("--source_path", type=str, default=str(GENERATED_DIR / "source_texts.json"))
    parser.add_argument("--output_dir", type=str, default=str(GENERATED_DIR))
    parser.add_argument("--model_path", type=str, default="/home/linkco/exa/models/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load source texts
    with open(args.source_path, "r", encoding="utf-8") as f:
        source_texts = json.load(f)
    print(f"Loaded {len(source_texts)} source texts")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU: {mem:.1f} GB total, {alloc:.1f} GB allocated")

    # Generate
    start = time.time()
    all_rewrites = generate_rewrites_batched(model, tokenizer, source_texts, output_dir)
    elapsed = time.time() - start

    print(f"\nGenerated {len(all_rewrites)} rewrites in {elapsed:.1f}s")
    print(f"Rate: {len(all_rewrites)/elapsed*60:.1f} rewrites/min")

    # Save
    rewrites_path = output_dir / "all_rewrites.json"
    with open(rewrites_path, "w", encoding="utf-8") as f:
        json.dump(all_rewrites, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {rewrites_path}")

    # Summary
    from collections import Counter
    quality_counts = Counter(r["quality_level"] for r in all_rewrites)
    print(f"\nQuality distribution: {dict(quality_counts)}")


if __name__ == "__main__":
    main()
