"""
SFT training for downstream validation.

Train Qwen2.5-7B-Instruct on filtered rewrite pairs using LoRA.
This tests whether evaluator-guided data selection improves downstream quality.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_sft_data(data_path: str) -> list:
    """Load SFT training data in messages format."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_model_and_tokenizer(base_model: str, use_4bit: bool = True):
    """Load model and tokenizer with optional 4-bit quantization."""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    """Configure and apply LoRA adapter."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_sft(args):
    """Main SFT training function."""
    print(f"Loading data from: {args.data_path}")
    train_data = load_sft_data(args.data_path)
    print(f"  Training samples: {len(train_data)}")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(train_data)

    # Setup model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = setup_model_and_tokenizer(args.base_model, use_4bit=not args.no_4bit)

    # Setup LoRA
    model = setup_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        data_collator=None,
        packing=False,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting SFT training...")
    trainer.train()

    # Save
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = {
        "train_samples": len(train_data),
        "epochs": args.epochs,
        "final_train_loss": trainer.state.log_history[-1].get("loss", "N/A") if trainer.state.log_history else "N/A",
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete! Model saved to: {output_dir}")
    print(f"Final train loss: {metrics['final_train_loss']}")


def main():
    parser = argparse.ArgumentParser(description="SFT training for downstream validation")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to SFT training data JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model path")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_sft(args)


if __name__ == "__main__":
    main()
