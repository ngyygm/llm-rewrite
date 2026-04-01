#!/bin/bash
# Run all remaining pairwise experiments sequentially on GPU 1
# Each experiment takes ~30-40 min with pairwise data (longer sequences)

set -euo pipefail

PROJECT_ROOT="/home/linkco/exa/llm-rewrite/emnlp2026"
BASE_MODEL="/home/linkco/exa/models/Qwen2.5-7B-Instruct"

run_training() {
    local name=$1
    local data=$2
    local lora_r=$3
    local lora_alpha=$4

    echo "===== Training: $name (r=$lora_r) ====="

    CUDA_VISIBLE_DEVICES=1 python "$PROJECT_ROOT/evaluator/train_lora.py" \
        --data_path "$data" \
        --output_dir "$PROJECT_ROOT/evaluator/checkpoints/$name" \
        --base_model "$BASE_MODEL" \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --epochs 3 \
        --lr 2e-4 \
        --batch_size 4 \
        --grad_accum 4 \
        --seed 42

    # If training was interrupted, copy from last checkpoint
    if [ ! -f "$PROJECT_ROOT/evaluator/checkpoints/$name/adapter_model.safetensors" ]; then
        echo "  Final adapter missing, copying from last checkpoint..."
        LAST_CK=$(ls -d "$PROJECT_ROOT/evaluator/checkpoints/$name/checkpoint-"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LAST_CK" ]; then
            cp "$LAST_CK/adapter_model.safetensors" "$PROJECT_ROOT/evaluator/checkpoints/$name/"
            cp "$LAST_CK/adapter_config.json" "$PROJECT_ROOT/evaluator/checkpoints/$name/"
            echo "  Copied from $LAST_CK"
        else
            echo "  ERROR: No checkpoint found!"
            return 1
        fi
    fi

    echo "  Training complete: $(ls -la "$PROJECT_ROOT/evaluator/checkpoints/$name/adapter_model.safetensors" | awk '{print $5}') bytes"
}

run_eval() {
    local name=$1
    local out=$2

    echo "===== Evaluating: $name ====="
    python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
        --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/$name" \
        --output_path "$PROJECT_ROOT/data/pairwise/$out.json" \
        --eval_mode cross_source

    echo "  Results saved to: $out"
}

# Clean up previous interrupted r=8 run (only 1 epoch)
rm -rf "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_r8"

# Experiments
echo "Starting at $(date)"
echo ""

# 1. r=8 ablation
run_training "pairwise_b1_r8" "$PROJECT_ROOT/data/pairwise/cross_source_train.json" 8 16
run_eval "pairwise_b1_r8" "b1_r8_results.json"

# 2. r=32 ablation
run_training "pairwise_b1_r32" "$PROJECT_ROOT/data/pairwise/cross_source_train.json" 32 64
run_eval "pairwise_b1_r32" "b1_r32_results.json"

# 3. 25% data efficiency
run_training "pairwise_b1_25pct" "$PROJECT_ROOT/data/pairwise/cross_source_train_25pct.json" 16 32
run_eval "pairwise_b1_25pct" "b1_25pct_results.json"

# 4. 50% data efficiency
run_training "pairwise_b1_50pct" "$PROJECT_ROOT/data/pairwise/cross_source_train_50pct.json" 16 32
run_eval "pairwise_b1_50pct" "b1_50pct_results.json"

echo ""
echo "===== All experiments complete at $(date) ====="
