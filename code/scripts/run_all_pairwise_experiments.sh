#!/bin/bash
# Run all remaining pairwise experiments sequentially on GPU 1
# Each experiment takes ~30-40 min with pairwise data (longer sequences)

set -euo pipefail

PROJECT_ROOT="/llm-rewrite"
BASE_MODEL="/Qwen/Qwen3-8B"

MODEL_NAME="qwen3-8b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# TIMESTAMP="20260412_141115"

run_training() {
    local name=$1
    local data=$2
    local lora_r=$3
    local lora_alpha=$4

    echo "===== Training: $name (r=$lora_r) ====="

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python "$PROJECT_ROOT/evaluator/train_lora.py" \
        --data_path "$data" \
        --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}" \
        --base_model "$BASE_MODEL" \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --epochs 3 \
        --lr 2e-4 \
        --batch_size 4 \
        --grad_accum 4 \
        --seed 42

    # If training was interrupted, copy from last checkpoint
    if [ ! -f "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}/adapter_model.safetensors" ]; then
        echo "  Final adapter missing, copying from last checkpoint..."
        LAST_CK=$(ls -d "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LAST_CK" ]; then
            cp "$LAST_CK/adapter_model.safetensors" "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}/"
            cp "$LAST_CK/adapter_config.json" "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}/"
            echo "  Copied from $LAST_CK"
        else
            echo "  ERROR: No checkpoint found!"
            return 1
        fi
    fi

    echo "  Training complete: $(ls -la "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}/adapter_model.safetensors" | awk '{print $5}') bytes"
}

run_eval() {
    local name=$1
    local out=$2

    echo "===== Evaluating: $name ====="
    python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
        --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise/${MODEL_NAME}_${TIMESTAMP}/${name}" \
        --base_model "$BASE_MODEL" \
        --output_path "$PROJECT_ROOT/data/pairwise/${MODEL_NAME}_${TIMESTAMP}/$out" \
        --eval_mode cross_source \
        --batch_size ${EVAL_BATCH_SIZE:-16}

    echo "  Results saved to: ${MODEL_NAME}_${TIMESTAMP}/$out"
}

# Clean up previous interrupted r=8 run (only 1 epoch)
# rm -rf "$PROJECT_ROOT/evaluator/checkpoints/${MODEL_NAME}_${TIMESTAMP}/pairwise_b1_r8"

# Experiments
echo "Starting at $(date)"
echo ""


CUDA_VISIBLE_DEVICES=0 python3 evaluator/train_lora.py \
  --data_path data/pairwise/generated_train.json \
  --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise/same_source/${MODEL_NAME}_${TIMESTAMP}"  \
  --base_model "$BASE_MODEL" \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 4 \
  --grad_accum 4 \
  --lora_r 16 \
  --lora_alpha 32

python3 evaluator/eval_pairwise.py \
  --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise/same_source/${MODEL_NAME}_${TIMESTAMP}" \
  --base_model "$BASE_MODEL" \
  --eval_mode cross_source \
  --output_path "data/pairwise/qwen3-8b/pairwise_same_source_generated_results.json" \
  --batch_size 16

# 1. r=8 ablation
run_training "pairwise_b1_r8" "$PROJECT_ROOT/data/pairwise/cross_source_train.json" 8 16
run_eval "pairwise_b1_r8" "pairwise_b1_r8_results.json"

# #r=16 ablation
run_training "pairwise_b1_r16" "$PROJECT_ROOT/data/pairwise/cross_source_train.json" 16 32
run_eval "pairwise_b1_r16" "pairwise_b1_r16_results.json"

# # 2. r=32 ablation
run_training "pairwise_b1_r32" "$PROJECT_ROOT/data/pairwise/cross_source_train.json" 32 64
run_eval "pairwise_b1_r32" "pairwise_b1_r32_results.json"

3. 25% data efficiency
run_training "pairwise_b1_25pct" "$PROJECT_ROOT/data/pairwise/cross_source_train_25pct.json" 16 32
run_eval "pairwise_b1_25pct" "pairwise_b1_25pct_results.json"

# 4. 50% data efficiency
run_training "pairwise_b1_50pct" "$PROJECT_ROOT/data/pairwise/cross_source_train_50pct.json" 16 32
run_eval "pairwise_b1_50pct" "pairwise_b1_50pct_results.json"

echo ""
echo "===== All experiments complete at $(date) ====="
