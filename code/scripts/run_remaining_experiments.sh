#!/bin/bash
# Run remaining experiments for RewritingBench paper
# GPU experiments: LoRA rank ablation + data efficiency curve
# Requires ~7h total GPU time across 2 GPUs

set -euo pipefail

PROJECT_ROOT="/home/linkco/exa/llm-rewrite/emnlp2026"
BASE_MODEL="/home/linkco/exa/models/Qwen2.5-7B-Instruct"
PAIRWISE_DATA="$PROJECT_ROOT/data/pairwise/cross_source_train.json"

echo "========================================="
echo "RewritingBench: Remaining Experiments"
echo "========================================="
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
echo ""

# ============================================================
# Block 2: LoRA Rank Ablation
# ============================================================
echo "===== Block 2: LoRA Rank Ablation ====="

# r=8 (GPU 0)
echo "[Block 2.1] Training pairwise B1 with r=8 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/evaluator/train_lora.py" \
    --data_path "$PAIRWISE_DATA" \
    --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_r8" \
    --base_model "$BASE_MODEL" \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --seed 42

echo "[Block 2.1] r=8 training complete. Running evaluation..."
python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
    --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_r8" \
    --output_path "$PROJECT_ROOT/data/pairwise/b1_r8_results.json" \
    --eval_mode cross_source

# r=32 (GPU 1)
echo "[Block 2.2] Training pairwise B1 with r=32 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python "$PROJECT_ROOT/evaluator/train_lora.py" \
    --data_path "$PAIRWISE_DATA" \
    --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_r32" \
    --base_model "$BASE_MODEL" \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --seed 42

echo "[Block 2.2] r=32 training complete. Running evaluation..."
python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
    --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_r32" \
    --output_path "$PROJECT_ROOT/data/pairwise/b1_r32_results.json" \
    --eval_mode cross_source

echo "===== Block 2 Complete ====="
echo ""

# ============================================================
# Block 3: Data Efficiency Curve
# ============================================================
echo "===== Block 3: Data Efficiency Curve ====="

# 25% data (GPU 0)
echo "[Block 3.1] Training pairwise B1 with 25% data on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/evaluator/train_lora.py" \
    --data_path "$PROJECT_ROOT/data/pairwise/cross_source_train_25pct.json" \
    --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_25pct" \
    --base_model "$BASE_MODEL" \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --seed 42

echo "[Block 3.1] 25% training complete. Running evaluation..."
python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
    --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_25pct" \
    --output_path "$PROJECT_ROOT/data/pairwise/b1_25pct_results.json" \
    --eval_mode cross_source

# 50% data (GPU 1)
echo "[Block 3.2] Training pairwise B1 with 50% data on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python "$PROJECT_ROOT/evaluator/train_lora.py" \
    --data_path "$PROJECT_ROOT/data/pairwise/cross_source_train_50pct.json" \
    --output_dir "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_50pct" \
    --base_model "$BASE_MODEL" \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --seed 42

echo "[Block 3.2] 50% training complete. Running evaluation..."
python "$PROJECT_ROOT/evaluator/eval_pairwise.py" \
    --checkpoint "$PROJECT_ROOT/evaluator/checkpoints/pairwise_b1_50pct" \
    --output_path "$PROJECT_ROOT/data/pairwise/b1_50pct_results.json" \
    --eval_mode cross_source

echo "===== Block 3 Complete ====="
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================="
echo "All experiments complete!"
echo "========================================="
echo ""
echo "Results files:"
echo "  - data/pairwise/b1_r8_results.json"
echo "  - data/pairwise/b1_r32_results.json"
echo "  - data/pairwise/b1_25pct_results.json"
echo "  - data/pairwise/b1_50pct_results.json"
echo ""
echo "Existing results for comparison:"
echo "  - data/pairwise/b1_cross_source_results.json (r=16, 100% data, Spearman 0.665)"
