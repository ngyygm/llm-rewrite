#!/usr/bin/env python3
"""
Generate SFT rewrite data using local Qwen2.5-7B-Instruct.

Steps:
1. Generate diverse Chinese source texts
2. Generate multiple rewrites per source with varying prompts
3. Save for downstream filtering experiments

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
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated_rewrites"
EVAL_DIR = DATA_DIR / "human_eval"


def load_existing_hashes():
    """Load hashes from eval/train sets to avoid overlap."""
    hashes = set()
    for fname in ["train.json", "eval.json"]:
        fpath = EVAL_DIR / fname
        if fpath.exists():
            data = json.load(open(fpath, "r", encoding="utf-8"))
            for item in data:
                h = hashlib.md5(item["input"].encode("utf-8")).hexdigest()
                hashes.add(h)
    return hashes


SOURCE_PROMPTS = [
    "请写一段50-100字的中文新闻短讯，包含具体事件和细节。",
    "请写一段50-100字的中文百科知识介绍，内容客观准确。",
    "请写一段30-80字的中文社交媒体帖子，语气自然口语化。",
    "请写一段50-100字的中文科技摘要，描述某项研究成果。",
    "请写一段50-100字的中文文学描写，语言优美有表现力。",
    "请写一段50-100字的中文商业评论，分析某个市场趋势。",
    "请写一段50-100字的中文教育类文本，关于学习方法或教育理念。",
    "请写一段50-100字的中文健康科普短文。",
    "请写一段50-100字的中文历史事件叙述。",
    "请写一段50-100字的中文环保主题短文。",
]

REWRITE_PROMPTS = [
    # High quality
    "请对以下文本进行高质量改写。要求：完整保留原文语义，使用完全不同的词汇和句式表达，保持原文风格和长度。只输出改写结果，不要解释。\n\n原文：{text}",
    # Medium quality
    "请改写以下文本，保留大致意思但改变部分词汇。\n\n原文：{text}",
    # Low quality (minimal change)
    "请用相近的词语替换原文中的一些词，保持句子结构不变。\n\n原文：{text}",
    # Another variation
    "请将以下文本改写为更口语化的表达方式。\n\n原文：{text}",
    # Formal variation
    "请将以下文本改写为更正式的书面语风格。\n\n原文：{text}",
]


def generate_source_texts(model, tokenizer, n_texts, existing_hashes, batch_size=4):
    """Generate diverse Chinese source texts."""
    all_texts = []
    seen_hashes = set()

    rng = np.random.RandomState(42)

    print(f"Generating {n_texts} source texts...")
    pbar = tqdm(total=n_texts, desc="Source texts")

    while len(all_texts) < n_texts:
        # Generate batch
        batch_prompts = []
        for _ in range(batch_size):
            prompt = rng.choice(SOURCE_PROMPTS)
            batch_prompts.append(prompt)

        messages_list = [[{"role": "user", "content": p}] for p in batch_prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i in range(len(batch_prompts)):
            generated = outputs[i][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            if len(text) < 20:
                continue

            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash not in seen_hashes and text_hash not in existing_hashes:
                seen_hashes.add(text_hash)
                all_texts.append({"text": text, "hash": text_hash})
                pbar.update(1)
                if len(all_texts) >= n_texts:
                    break

    pbar.close()
    print(f"Generated {len(all_texts)} source texts")
    return all_texts


def generate_rewrites(model, tokenizer, source_texts, n_variations=3, batch_size=4):
    """Generate multiple rewrites per source text."""
    all_rewrites = []

    print(f"Generating rewrites for {len(source_texts)} source texts...")
    pbar = tqdm(total=len(source_texts), desc="Rewrites")

    for src_idx, source in enumerate(source_texts):
        source_text = source["text"]
        # Select n_variations different rewrite prompts
        rng = np.random.RandomState(src_idx)
        prompt_indices = rng.choice(len(REWRITE_PROMPTS), size=min(n_variations, len(REWRITE_PROMPTS)), replace=False)

        for prompt_idx in prompt_indices:
            prompt = REWRITE_PROMPTS[prompt_idx].format(text=source_text)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            rewrite = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            if len(rewrite) < 10:
                continue

            all_rewrites.append({
                "source_text": source_text,
                "source_hash": source["hash"],
                "rewrite_text": rewrite,
                "prompt_type": prompt_idx,
            })

        pbar.update(1)

        # Save progress every 50
        if (src_idx + 1) % 50 == 0:
            progress_path = GENERATED_DIR / "rewrites_progress.json"
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(all_rewrites, f, ensure_ascii=False, indent=2)

    pbar.close()
    return all_rewrites


def main():
    parser = argparse.ArgumentParser(description="Generate SFT rewrite data")
    parser.add_argument("--n_sources", type=int, default=300, help="Number of source texts")
    parser.add_argument("--n_variations", type=int, default=3, help="Rewrites per source")
    parser.add_argument("--model_path", type=str, default="/home/linkco/exa/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default=str(GENERATED_DIR))
    parser.add_argument("--skip_source", action="store_true", help="Skip source generation")
    parser.add_argument("--skip_rewrite", action="store_true", help="Skip rewrite generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    existing_hashes = load_existing_hashes()
    print(f"Existing hashes: {len(existing_hashes)}")

    # Step 1: Generate source texts
    source_path = output_dir / "source_texts.json"
    if args.skip_source and source_path.exists():
        print(f"Loading existing source texts from {source_path}")
        source_texts = json.load(open(source_path, "r", encoding="utf-8"))
    else:
        source_texts = generate_source_texts(model, tokenizer, args.n_sources, existing_hashes)
        with open(source_path, "w", encoding="utf-8") as f:
            json.dump(source_texts, f, ensure_ascii=False, indent=2)

    print(f"Source texts: {len(source_texts)}")

    # Step 2: Generate rewrites
    rewrites_path = output_dir / "all_rewrites.json"
    if args.skip_rewrite and rewrites_path.exists():
        print(f"Loading existing rewrites from {rewrites_path}")
        all_rewrites = json.load(open(rewrites_path, "r", encoding="utf-8"))
    else:
        start = time.time()
        all_rewrites = generate_rewrites(model, tokenizer, source_texts, args.n_variations)
        elapsed = time.time() - start
        print(f"Generated {len(all_rewrites)} rewrites in {elapsed:.1f}s")
        with open(rewrites_path, "w", encoding="utf-8") as f:
            json.dump(all_rewrites, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(source_texts)} sources, {len(all_rewrites)} rewrites")
    print(f"Expected: {len(source_texts)} x {args.n_variations} = {len(source_texts) * args.n_variations}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
