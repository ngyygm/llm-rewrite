#!/bin/bash
# =============================================================================
# LoRA Evaluator Training - Full training + Learning Curves
# EMNLP 2026
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

BASE_MODEL="${LOCAL_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
EVAL_DATA="$PROJECT_DIR/data/human_eval/eval.json"
CHECKPOINT_DIR="$PROJECT_DIR/evaluator/checkpoints"
RESULTS_DIR="$PROJECT_DIR/data/baselines"

echo "============================================"
echo "EMNLP 2026: LoRA Evaluator Training"
echo "============================================"
echo "Base model: $BASE_MODEL"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"

# Step 1: Full training - Score Only mode (primary)
echo "[Step 1/6] Training full model - score_only mode..."
python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_score_only.json" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$CHECKPOINT_DIR/score_only_full" \
    --base_model "$BASE_MODEL" \
    --mode score_only \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

echo ""

# Step 2: Evaluate full score_only model
echo "[Step 2/6] Evaluating score_only_full..."
python3 evaluator/eval_evaluator.py \
    --model_path "$CHECKPOINT_DIR/score_only_full" \
    --eval_data_path "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --mode score_only \
    --results_path "$RESULTS_DIR/results_lora_score_only_full.json" \
    --save_predictions

echo ""

# Step 3: Full training - Multi Score mode
echo "[Step 3/6] Training full model - multi_score mode..."
python3 evaluator/train_lora.py \
    --data_path "$PROJECT_DIR/data/human_eval/train_multi_score.json" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$CHECKPOINT_DIR/multi_score_full" \
    --base_model "$BASE_MODEL" \
    --mode multi_score \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4

echo ""

# Step 4: Learning curve training (subsets)
SUBSETS=(50 100 200 400)
LEARNING_CURVE_DATA="$RESULTS_DIR/learning_curves.json"
echo "[" > "$LEARNING_CURVE_DATA"
echo "  {\"subset_size\": 0, \"method\": \"zero_shot_7b\", \"spearman\": 0}" >> "$LEARNING_CURVE_DATA"

for SIZE in "${SUBSETS[@]}"; do
    echo ""
    echo "[Learning Curve] Training on $SIZE samples..."
    python3 evaluator/train_lora.py \
        --data_path "$PROJECT_DIR/data/human_eval/train_score_only_${SIZE}.json" \
        --eval_data_path "$EVAL_DATA" \
        --output_dir "$CHECKPOINT_DIR/score_only_${SIZE}" \
        --base_model "$BASE_MODEL" \
        --mode score_only \
        --epochs 3 \
        --lr 2e-4 \
        --batch_size 4 \
        --grad_accum 4 \
        --subset_size $SIZE

    echo "[Learning Curve] Evaluating $SIZE subset..."
    python3 evaluator/eval_evaluator.py \
        --model_path "$CHECKPOINT_DIR/score_only_${SIZE}" \
        --eval_data_path "$EVAL_DATA" \
        --base_model "$BASE_MODEL" \
        --mode score_only \
        --results_path "$RESULTS_DIR/results_lora_score_only_${SIZE}.json"

    # Append to learning curve data
    SPEARMAN=$(python3 -c "
import json
d = json.load(open('$RESULTS_DIR/results_lora_score_only_${SIZE}.json'))
print(d['metrics_vs_avg_score']['spearman_rho'])
")
    echo ", {\"subset_size\": $SIZE, \"method\": \"lora_7b\", \"spearman\": $SPEARMON}" >> "$LEARNING_CURVE_DATA"
done

echo "]" >> "$LEARNING_CURVE_DATA"

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
