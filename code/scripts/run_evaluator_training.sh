#!/bin/bash
# =============================================================================
# LoRA Evaluator Training - Full training + Learning Curves
# EMNLP 2026
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"


BASE_MODEL="${LOCAL_MODEL_PATH:-/Qwen/Qwen3-8B}"
EVAL_DATA="$PROJECT_DIR/data/human_eval/eval.json"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CHECKPOINT_DIR="$PROJECT_DIR/evaluator/checkpoints/${MODEL_NAME}_${TIMESTAMP}"
RESULTS_DIR="$PROJECT_DIR/data/baselines/${MODEL_NAME}_${TIMESTAMP}"
LEARNING_CURVE_DATA="${LEARNING_CURVE_DATA:-$RESULTS_DIR/learning_curves.json}"

echo "============================================"
echo "EMNLP 2026: LoRA Evaluator Training"
echo "============================================"
echo "Base model: $BASE_MODEL"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"

# =============================================================================
# Step 1: 平衡版 score_only 完整训练（论文主模型，1008条）
# =============================================================================
echo "[Step 1/4] Training balanced score_only model (1008 samples)..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_score_only_detail_balanced.json" \
    --output_dir "$CHECKPOINT_DIR/score_only_detail_balanced_full" \
    --base_model "$BASE_MODEL" \
    --eval_data_path "$EVAL_DATA" \
    --prompt_variant detail \
    --mode score_only \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

echo ""

# Step 1 评估
echo "[Step 1 Eval] Evaluating balanced score_only model..."
python3 evaluator/eval_evaluator.py \
    --model_path "$CHECKPOINT_DIR/score_only_detail_balanced_full" \
    --eval_data_path "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --mode score_only \
    --results_path "$RESULTS_DIR/results_lora_score_only_detail_balanced_full.json" \
    --prompt_variant detail \
    --save_predictions

echo ""

# =============================================================================
# Step 2: 未平衡版 score_only 完整训练（对照实验，600条）
# =============================================================================
echo "[Step 2/4] Training unbalanced score_only model (600 samples)..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_score_only_detail.json" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$CHECKPOINT_DIR/score_only_detail_unbalanced_full" \
    --base_model "$BASE_MODEL" \
    --mode score_only \
    --prompt_variant detail \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

# echo ""

# # # Step 2 评估
echo "[Step 2 Eval] Evaluating unbalanced score_only model..."
python3 evaluator/eval_evaluator.py \
    --model_path "$CHECKPOINT_DIR/score_only_detail_unbalanced_full" \
    --eval_data_path "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --prompt_variant detail \
    --mode score_only \
    --results_path "$RESULTS_DIR/results_lora_score_only_detail_unbalanced_full.json" \
    --save_predictions

echo ""

# =============================================================================
# Step 3: 学习曲线训练（平衡子集 50/100/200/400）
# =============================================================================
echo "[Step 3/4] Learning curve training..."

# SUBSETS=(50 100 200 400)
SUBSETS=(200)
LEARNING_CURVE_DATA="$RESULTS_DIR/learning_curves.json"
echo "[" > "$LEARNING_CURVE_DATA"
echo "  {\"subset_size\": 0, \"method\": \"zero_shot_7b\", \"spearman\": 0}" >> "$LEARNING_CURVE_DATA"

for SIZE in "${SUBSETS[@]}"; do
    echo ""
    echo "[Learning Curve] Training on $SIZE samples..."
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python3 evaluator/train_lora.py \
        --data_path "$PROJECT_DIR/data/human_eval/train_score_only_detail_balanced_${SIZE}.json" \
        --eval_data_path "$EVAL_DATA" \
        --output_dir "$CHECKPOINT_DIR/score_only_detail_balanced_${SIZE}" \
        --base_model "$BASE_MODEL" \
        --mode score_only \
        --prompt_variant detail \
        --epochs 3 \
        --lr 2e-4 \
        --batch_size 4 \
        --grad_accum 4 \
        --subset_size $SIZE


            # --model_path "$CHECKPOINT_DIR/score_only_detail_balanced_${SIZE}" \

    echo "[Learning Curve] Evaluating $SIZE subset..."
    python3 evaluator/eval_evaluator.py \
        --model_path "$CHECKPOINT_DIR/score_only_detail_balanced_${SIZE}" \
        --eval_data_path "$EVAL_DATA" \
        --base_model "$BASE_MODEL" \
        --prompt_variant detail \
        --mode score_only \
        --results_path "$RESULTS_DIR/results_lora_score_only_detail_balanced_${SIZE}.json" \
        --save_predictions

    SPEARMAN=$(python3 -c "
import json
d = json.load(open('$CHECKPOINT_DIR/score_only_detail_balanced_${SIZE}/best_spearman_checkpoint/eval_results_spearman.json'))
print(d['metrics_vs_avg_score']['spearman_rho'])
")
    echo ", {\"subset_size\": $SIZE, \"method\": \"lora_7b\", \"spearman\": $SPEARMAN}" >> "$LEARNING_CURVE_DATA"
done

echo "]" >> "$LEARNING_CURVE_DATA"

echo ""

# =============================================================================
# Step 4: Multi Score 完整训练（对照实验）
# =============================================================================
echo "[Step 4/4] Training multi_score model..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_multi_score.json" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$CHECKPOINT_DIR/multi_score_full" \
    --base_model "$BASE_MODEL" \
    --mode multi_score \
    --prompt_variant detail \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

python3 evaluator/eval_evaluator.py \
    --model_path "$CHECKPOINT_DIR/multi_score_full/best_spearman_checkpoint" \
    --eval_data_path "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --prompt_variant detail \
    --mode multi_score \
    --results_path "$RESULTS_DIR/results_lora_multi_score_full.json" \
    --save_predictions

echo ""

echo ""

# =============================================================================
# Step 5: Reasoning Prefix 完整训练（论文消融实验 "+ Reasoning prefix"）
# =============================================================================
echo "[Step 5] Training reasoning prefix model..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_score_detail_reasoning.json" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$CHECKPOINT_DIR/score_detail_reasoning_full" \
    --base_model "$BASE_MODEL" \
    --mode score_only \
    --prompt_variant detail \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

echo "[Step 5 Eval] Evaluating reasoning prefix model..."
python3 evaluator/eval_evaluator.py \
    --model_path "$CHECKPOINT_DIR/score_detail_reasoning_full/best_spearman_checkpoint" \
    --eval_data_path "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --prompt_variant detail \
    --mode score_only \
    --results_path "$RESULTS_DIR/results_lora_score_detail_reasoning_full.json" \
    --save_predictions

echo ""

echo "============================================"
echo "Evaluator training complete!"
echo ""
echo "Checkpoints:"
ls -la "$CHECKPOINT_DIR/"
echo ""
echo "Results:"
ls -la "$RESULTS_DIR"/results_lora_*.json
echo ""
echo "Learning curve data: $LEARNING_CURVE_DATA"
echo "============================================"
