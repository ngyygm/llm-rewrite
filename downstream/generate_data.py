"""
Generate SFT training data for downstream validation experiments.

Step 1: Generate 2000 diverse Chinese source texts
Step 2: Generate 3 rewrites per source (low/medium/high quality) = 6000 pairs
Step 3: Source texts must NOT overlap with 730 evaluation set

Uses QWQ-32B or other local LLMs for generation.
"""
import json
import os
import random
import hashlib
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated_rewrites"
EVAL_DIR = DATA_DIR / "human_eval"

SEED = 42

# Source text categories for diversity
SOURCE_CATEGORIES = {
    "news": {
        "description": "新闻报道类文本",
        "prompt": "请生成一段中文新闻报道文本，包含具体事件、人物、地点等细节。长度在50-150字之间。内容可以是社会、科技、体育、国际等新闻类型。",
        "count": 500,
    },
    "wikipedia": {
        "description": "百科知识类文本",
        "prompt": "请生成一段中文百科全书风格的文本，介绍一个概念、历史事件或科学原理。长度在50-150字之间。内容准确客观。",
        "count": 400,
    },
    "social_media": {
        "description": "社交媒体类文本",
        "prompt": "请生成一段中文社交媒体风格的文本，如微博、知乎回答或论坛帖子。长度在30-120字之间。语气自然口语化。",
        "count": 400,
    },
    "scientific": {
        "description": "科技论文摘要类文本",
        "prompt": "请生成一段中文科技论文摘要风格的文本，描述某项研究成果。长度在80-200字之间。使用学术化但清晰的中文。",
        "count": 300,
    },
    "literature": {
        "description": "文学散文类文本",
        "prompt": "请生成一段中文文学散文风格的文本，可以是描写、叙事或议论。长度在50-150字之间。语言优美有表现力。",
        "count": 200,
    },
    "business": {
        "description": "商业财经类文本",
        "prompt": "请生成一段中文商业财经类文本，如公司公告、市场分析或财经新闻。长度在50-150字之间。专业但易懂。",
        "count": 200,
    },
}


def load_eval_source_texts():
    """Load source texts from evaluation set to ensure no overlap."""
    eval_data = json.load(open(EVAL_DIR / "eval.json", "r", encoding="utf-8"))
    train_data = json.load(open(EVAL_DIR / "train.json", "r", encoding="utf-8"))
    all_data = eval_data + train_data
    return set(
        hashlib.md5(d["input"].encode("utf-8")).hexdigest()
        for d in all_data
    )


def check_overlap(text: str, existing_hashes: set) -> bool:
    """Check if text overlaps with existing data."""
    text_hash = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
    return text_hash in existing_hashes


def generate_source_texts_via_api(api_url: str, total_target: int = 2000, existing_hashes: Optional[set] = None):
    """Generate source texts using a local LLM API.

    Uses vLLM or OpenAI-compatible API endpoint.
    """
    import requests

    if existing_hashes is None:
        existing_hashes = set()

    all_texts = []
    seen_hashes = set()

    for category, config in SOURCE_CATEGORIES.items():
        count = config["count"]
        print(f"  Generating {count} texts for category: {category} ({config['description']})")

        generated = 0
        batch_size = 10  # Generate in batches for efficiency

        while generated < count:
            # Generate batch prompt
            batch_prompt = f"请一次性生成{min(batch_size, count - generated)}段独立的中文文本。每段文本之间用'---'分隔。\n\n{config['prompt']}\n\n注意：每段文本必须是完全不同的内容。"

            try:
                response = requests.post(
                    f"{api_url}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": batch_prompt}],
                        "temperature": 0.9,
                        "max_tokens": 2048,
                    },
                    timeout=120,
                )
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Split by separator and process each text
                texts = [t.strip() for t in content.split("---") if t.strip()]

                for text in texts:
                    if len(text) < 20:  # Skip very short texts
                        continue
                    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                    if text_hash not in seen_hashes and text_hash not in existing_hashes:
                        seen_hashes.add(text_hash)
                        all_texts.append({
                            "text": text,
                            "category": category,
                            "source_hash": text_hash,
                        })
                        generated += 1
                        if generated >= count:
                            break

            except Exception as e:
                print(f"    Error generating for {category}: {e}")
                continue

        print(f"    Generated {generated}/{count} texts for {category}")

    print(f"\nTotal source texts generated: {len(all_texts)}")
    return all_texts


def generate_rewrites_via_api(source_texts: list, api_url: str, output_dir: Path):
    """Generate 3 rewrites per source text (low/medium/high quality)."""
    import requests

    quality_prompts = {
        "low": "请对以下中文文本进行改写。要求：改写质量较低，可以丢失部分原意，词汇替换不准确，句式变化生硬。\n\n原文：{text}",
        "medium": "请对以下中文文本进行改写。要求：改写质量中等，保留大部分原意，有一定的词汇替换和句式变化。\n\n原文：{text}",
        "high": "请对以下中文文本进行高质量改写。要求：完整保留原文语义，使用不同的词汇和句式表达，保持原文风格。\n\n原文：{text}",
    }

    all_rewrites = []

    for i, source in enumerate(source_texts):
        print(f"  Processing {i + 1}/{len(source_texts)}: {source['category']}")
        source_text = source["text"]
        source_hash = source["source_hash"]

        for quality, prompt_template in quality_prompts.items():
            prompt = prompt_template.format(text=source_text)

            try:
                response = requests.post(
                    f"{api_url}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8 if quality != "high" else 0.6,
                        "max_tokens": 512,
                    },
                    timeout=60,
                )
                result = response.json()
                rewrite = result["choices"][0]["message"]["content"].strip()

                all_rewrites.append({
                    "source_text": source_text,
                    "source_category": source["category"],
                    "source_hash": source_hash,
                    "rewrite_text": rewrite,
                    "quality_level": quality,
                })

            except Exception as e:
                print(f"    Error: {e}")

        # Save progress every 50 sources
        if (i + 1) % 50 == 0:
            save_path = output_dir / "rewrites_progress.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_rewrites, f, ensure_ascii=False, indent=2)
            print(f"    Saved progress: {len(all_rewrites)} rewrites")

    return all_rewrites


def generate_source_texts_from_file(text_file: str) -> list:
    """
    Alternative: Load source texts from a text file (one per line or paragraph).
    Useful when pre-downloaded Chinese text corpora are available.
    """
    texts = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if len(text) >= 30:
                texts.append({
                    "text": text,
                    "category": "file_import",
                    "source_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                })
    return texts


def generate_rewrites_local_model(source_texts, model_path, tokenizer_path, output_dir):
    """
    Generate rewrites using a local model (no API needed).
    Uses HuggingFace transformers directly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    quality_prompts = {
        "low": "请对以下中文文本进行改写。要求：改写质量较低，可以丢失部分原意，词汇替换不准确，句式变化生硬。\n\n原文：{text}",
        "medium": "请对以下中文文本进行改写。要求：改写质量中等，保留大部分原意，有一定的词汇替换和句式变化。\n\n原文：{text}",
        "high": "请对以下中文文本进行高质量改写。要求：完整保留原文语义，使用不同的词汇和句式表达，保持原文风格。\n\n原文：{text}",
    }

    all_rewrites = []

    for i, source in enumerate(source_texts):
        print(f"  Processing {i + 1}/{len(source_texts)}: {source['category']}")
        source_text = source["text"]

        for quality, prompt_template in quality_prompts.items():
            prompt = prompt_template.format(text=source_text)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8 if quality != "high" else 0.6,
                    top_p=0.9,
                    do_sample=True,
                )

            rewrite = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            all_rewrites.append({
                "source_text": source_text,
                "source_category": source["category"],
                "source_hash": source["source_hash"],
                "rewrite_text": rewrite,
                "quality_level": quality,
            })

        if (i + 1) % 50 == 0:
            with open(output_dir / "rewrites_progress.json", "w", encoding="utf-8") as f:
                json.dump(all_rewrites, f, ensure_ascii=False, indent=2)

    return all_rewrites


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--mode", choices=["api", "local", "file"], default="api",
                        help="Generation mode: api (vLLM), local (transformers), file (import from file)")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                        help="vLLM API URL for API mode")
    parser.add_argument("--model_path", type=str, default="",
                        help="Model path for local mode")
    parser.add_argument("--text_file", type=str, default="",
                        help="Source text file for file mode")
    parser.add_argument("--output_dir", type=str, default=str(GENERATED_DIR),
                        help="Output directory")
    parser.add_argument("--skip_source_gen", action="store_true",
                        help="Skip source text generation (use existing)")
    parser.add_argument("--skip_rewrite_gen", action="store_true",
                        help="Skip rewrite generation (use existing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval set hashes to ensure no overlap
    print("Loading eval set hashes...")
    eval_hashes = load_eval_source_texts()
    print(f"  {len(eval_hashes)} unique source texts in eval+train set")

    # Step 1: Generate source texts
    source_texts_path = output_dir / "source_texts.json"
    if args.skip_source_gen and source_texts_path.exists():
        print(f"Loading existing source texts from {source_texts_path}")
        source_texts = json.load(open(source_texts_path, "r", encoding="utf-8"))
    elif args.mode == "file":
        print(f"Loading source texts from file: {args.text_file}")
        source_texts = generate_source_texts_from_file(args.text_file)
        # Remove any that overlap with eval set
        source_texts = [t for t in source_texts if t["source_hash"] not in eval_hashes]
        with open(source_texts_path, "w", encoding="utf-8") as f:
            json.dump(source_texts, f, ensure_ascii=False, indent=2)
    else:
        print("Generating source texts...")
        if args.mode == "api":
            source_texts = generate_source_texts_via_api(args.api_url, 2000, eval_hashes)
        elif args.mode == "local":
            print("Local mode: please provide --text_file for source texts, or use API mode")
            return
        with open(source_texts_path, "w", encoding="utf-8") as f:
            json.dump(source_texts, f, ensure_ascii=False, indent=2)

    print(f"\nTotal source texts: {len(source_texts)}")

    # Step 2: Generate rewrites
    rewrites_path = output_dir / "all_rewrites.json"
    if args.skip_rewrite_gen and rewrites_path.exists():
        print(f"Loading existing rewrites from {rewrites_path}")
        all_rewrites = json.load(open(rewrites_path, "r", encoding="utf-8"))
    else:
        print(f"\nGenerating rewrites for {len(source_texts)} source texts...")
        if args.mode == "api":
            all_rewrites = generate_rewrites_via_api(source_texts, args.api_url, output_dir)
        elif args.mode == "local" and args.model_path:
            all_rewrites = generate_rewrites_local_model(
                source_texts, args.model_path, args.model_path, output_dir
            )
        else:
            print("Error: need --api_url for API mode or --model_path for local mode")
            return
        with open(rewrites_path, "w", encoding="utf-8") as f:
            json.dump(all_rewrites, f, ensure_ascii=False, indent=2)

    print(f"\nTotal rewrites: {len(all_rewrites)}")
    print(f"Expected: {len(source_texts)} × 3 = {len(source_texts) * 3}")

    # Summary
    quality_counts = {}
    category_counts = {}
    for r in all_rewrites:
        q = r["quality_level"]
        quality_counts[q] = quality_counts.get(q, 0) + 1
        c = r["source_category"]
        category_counts[c] = category_counts.get(c, 0) + 1

    print("\nQuality distribution:")
    for q in ["low", "medium", "high"]:
        print(f"  {q}: {quality_counts.get(q, 0)}")

    print("\nCategory distribution:")
    for c, count in sorted(category_counts.items()):
        print(f"  {c}: {count}")

    print(f"\nAll data saved to: {output_dir}")


if __name__ == "__main__":
    main()
