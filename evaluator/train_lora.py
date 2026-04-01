#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Chinese Rewriting Quality Evaluator.

Fine-tunes Qwen2.5-7B-Instruct with LoRA adapters for Chinese text
rewriting quality evaluation. Supports both score-only and multi-score
training modes, as well as subset training for learning curve analysis.

Target hardware: Single NVIDIA RTX 3090 (~24 GB VRAM).
Uses 4-bit quantization (bitsandbytes NF4) + gradient checkpointing.

Usage:
    # Full training, score-only mode
    python train_lora.py \
        --data_path data/human_eval/train_score_only.json \
        --output_dir evaluator/checkpoints/score_only_full

    # Multi-score mode
    python train_lora.py \
        --data_path data/human_eval/train_multi_score.json \
        --output_dir evaluator/checkpoints/multi_score_full \
        --mode multi_score

    # Subset training for learning curves
    python train_lora.py \
        --data_path data/human_eval/train_score_only_200.json \
        --output_dir evaluator/checkpoints/score_only_200 \
        --subset_size 200

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

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.prompts import SYSTEM_PROMPT_SCORE_ONLY, SYSTEM_PROMPT_MULTI_SCORE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
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
    # Ensure deterministic behavior in CuDNN (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_training_data(data_path: str) -> list[dict]:
    """Load training data in messages format.

    Expected format:
    [
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        },
        ...
    ]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_eval_data_for_training(eval_path: str, mode: str = "score_only") -> list[dict]:
    """Load eval.json and convert to messages format for validation during training.

    This converts the eval format (input/output/annotator_scores) into the
    messages format expected by SFTTrainer.
    """
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    system_prompt = SYSTEM_PROMPT_SCORE_ONLY if mode == "score_only" else SYSTEM_PROMPT_MULTI_SCORE

    converted = []
    for item in eval_data:
        source = item["input"]
        rewrite = item["output"]
        score = item["consensus_score"]

        if mode == "score_only":
            user_content = (
                f"原文：\n{source}\n\n改写：\n{rewrite}\n\n"
                f"请对该改写进行综合评分（0-5分）。"
            )
            assistant_content = f"该改写的综合评分为{score}分。"
        else:
            user_content = (
                f"原文：\n{source}\n\n改写：\n{rewrite}\n\n"
                f"请按照5个维度（语义一致性、句式重构、词汇变化、风格保持、综合评分）评分（0-5分）。"
            )
            assistant_content = (
                f'[{{"要求1": "分析理由", "score": {score}}}, '
                f'{{"要求2": "分析理由", "score": {score}}}, '
                f'{{"要求3": "分析理由", "score": {score}}}, '
                f'{{"要求4": "分析理由", "score": {score}}}, '
                f'{{"要求5": "综合评价", "score": {score}}}]'
            )

        converted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    return converted


# ---------------------------------------------------------------------------
# Model & Tokenizer Setup
# ---------------------------------------------------------------------------

def get_bnb_config() -> dict:
    """Return bitsandbytes 4-bit quantization config dict."""
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }


def get_lora_config() -> dict:
    """Return LoRA configuration dict."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Chinese rewriting quality evaluator"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to training data JSON (messages format).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save LoRA adapter and training logs.",
    )
    parser.add_argument(
        "--eval_data_path", type=str,
        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"),
        help="Path to eval.json for validation during training.",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--mode", type=str, default="score_only", choices=["score_only", "multi_score"],
        help="Training mode: score_only or multi_score.",
    )
    parser.add_argument(
        "--subset_size", type=int, default=None,
        help="Subset size for learning curve (e.g. 50, 100, 200, 400).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)
    logger = setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("LoRA Fine-tuning: Chinese Rewriting Quality Evaluator")
    logger.info("=" * 70)
    logger.info(f"Base model       : {args.base_model}")
    logger.info(f"Mode             : {args.mode}")
    logger.info(f"Subset size      : {args.subset_size or 'full'}")
    logger.info(f"Data path        : {args.data_path}")
    logger.info(f"Eval data path   : {args.eval_data_path}")
    logger.info(f"Output dir       : {args.output_dir}")
    logger.info(f"Epochs           : {args.epochs}")
    logger.info(f"Learning rate    : {args.lr}")
    logger.info(f"Batch size       : {args.batch_size} x {args.grad_accum} = "
                f"{args.batch_size * args.grad_accum} (effective)")
    logger.info(f"Max seq length   : {args.max_seq_length}")
    logger.info(f"LoRA r/alpha/drop: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    logger.info(f"Seed             : {args.seed}")

    # Save args
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------
    logger.info("Loading training data...")
    train_data = load_training_data(args.data_path)
    logger.info(f"  Training samples: {len(train_data)}")

    logger.info("Loading and converting eval data...")
    eval_data = load_eval_data_for_training(args.eval_data_path, args.mode)
    logger.info(f"  Eval samples: {len(eval_data)}")

    # ------------------------------------------------------------------
    # Lazy imports (after seed/logging are set up, and to avoid slow
    # import overhead when just checking --help)
    # ------------------------------------------------------------------
    logger.info("Importing transformers, peft, trl...")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig

    # ------------------------------------------------------------------
    # Quantization & Tokenizer
    # ------------------------------------------------------------------
    logger.info("Initializing 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    # Ensure pad token exists (Qwen uses </s> for padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  Set pad_token = eos_token")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.pretraining_tp = 1

    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Enable gradient checkpointing explicitly
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    logger.info("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"  Trainable parameters: {trainable:,} / {total:,} "
                f"({100.0 * trainable / total:.2f}%)")

    # ------------------------------------------------------------------
    # SFT Config & Trainer
    # ------------------------------------------------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if early_stopping is available
    try:
        from transformers import EarlyStoppingCallback
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        logger.info("Early stopping enabled (patience=5)")
    except ImportError:
        callbacks = []
        logger.warning("EarlyStoppingCallback not available, skipping early stopping")

    sft_config = SFTConfig(
        # Data
        dataset_text_field="messages",
        max_seq_length=args.max_seq_length,
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Precision
        bf16=True,
        fp16=False,
        # Optimizer
        optim="paged_adamw_8bit",
        # Gradient checkpointing is already enabled on model
        gradient_checkpointing=False,
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        # Saving & Eval
        output_dir=str(output_path),
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Do not remove columns since we use the "messages" field directly
        remove_unused_columns=False,
        # Seed
        seed=args.seed,
        # Dataset kwargs for chat template
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    logger.info("Initializing SFTTrainer...")
    # Convert lists to HuggingFace Datasets (trl requires Dataset objects)
    from datasets import Dataset

    # Pre-format messages into chat template strings for TRL compatibility
    def format_messages(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = Dataset.from_list(train_data).map(format_messages, remove_columns=["messages"])
    eval_dataset = Dataset.from_list(eval_data).map(format_messages, remove_columns=["messages"])
    logger.info(f"  Formatted train dataset: {len(train_dataset)} samples")
    logger.info(f"  Formatted eval dataset: {len(eval_dataset)} samples")
    logger.info(f"  Example text length: {len(train_dataset[0]['text'])} chars")

    sft_config.dataset_text_field = "text"
    sft_config.remove_unused_columns = True

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    start_time = time.time()

    # Check for flash_attention_2 availability and warn if not installed
    try:
        import flash_attn  # noqa: F401
        logger.info("Flash Attention 2 is available.")
    except ImportError:
        logger.warning(
            "flash_attn not installed. Model will use sdpa attention. "
            "Install with: pip install flash-attn --no-build-isolation"
        )

    train_result = trainer.train()

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("Saving LoRA adapter...")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save training state
    trainer.save_state()

    # Log final eval metrics if available
    try:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    except Exception as e:
        logger.warning(f"Could not run final evaluation: {e}")

    logger.info("=" * 70)
    logger.info(f"All outputs saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
