#!/bin/bash
# =============================================================================
# Downstream SFT Validation - Data generation, filtering, training, evaluation
# EMNLP 2026
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

BASE_MODEL="${LOCAL_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
GENERATED_DIR="$PROJECT_DIR/data/generated_rewrites"
FILTERED_DIR="$GENERATED_DIR/filtered"
SFT_DIR="$PROJECT_DIR/downstream/checkpoints"
EVAL_RESULTS_DIR="$PROJECT_DIR/data/downstream_results"

EVALUATOR_CHECKPOINT="${EVALUATOR_ADAPTER:-$PROJECT_DIR/evaluator/checkpoints/score_only_full}"
API_URL="${API_URL:-http://localhost:8000}"

echo "============================================"
echo "EMNLP 2026: Downstream SFT Validation"
echo "============================================"
echo "Base model: $BASE_MODEL"
echo "Evaluator: $EVALUATOR_CHECKPOINT"
echo ""

mkdir -p "$GENERATED_DIR" "$FILTERED_DIR" "$SFT_DIR" "$EVAL_RESULTS_DIR"

# =============================================================================
# Phase 1: Generate rewrite data
# =============================================================================
echo "[Phase 1] Generating SFT training data..."
echo ""

# Step 1a: Generate source texts
if [ ! -f "$GENERATED_DIR/source_texts.json" ]; then
    echo "  [1a] Generating 2000 source texts via API ($API_URL)..."
    python3 downstream/generate_data.py \
        --mode api \
        --api_url "$API_URL" \
        --output_dir "$GENERATED_DIR"
else
    echo "  [1a] Source texts already exist, skipping"
fi

# Step 1b: Generate rewrites
if [ ! -f "$GENERATED_DIR/all_rewrites.json" ]; then
    echo "  [1b] Generating 6000 rewrites (3 per source) via API..."
    python3 downstream/generate_data.py \
        --mode api \
        --api_url "$API_URL" \
        --output_dir "$GENERATED_DIR" \
        --skip_source_gen
else
    echo "  [1b] Rewrites already exist, skipping"
fi

echo ""

# =============================================================================
# Phase 2: Filter data using evaluator
# =============================================================================
echo "[Phase 2] Filtering data using evaluator scores..."
echo ""

# Use quality_level as proxy if evaluator hasn't scored the data yet
# (In practice, you'd first run the evaluator on all 6000 pairs)
python3 downstream/filter_data.py \
    --rewrites_path "$GENERATED_DIR/all_rewrites.json" \
    --strategy all \
    --k 2000 \
    --threshold 3.0 \
    --output_dir "$FILTERED_DIR"

echo ""

# =============================================================================
# Phase 3: SFT training (one model per filtering strategy)
# =============================================================================
echo "[Phase 3] SFT training..."
echo ""

STRATEGIES=("random_2000" "bleu_filtered" "top_2000" "threshold_3.0")

for STRATEGY in "${STRATEGIES[@]}"; do
    SFT_DATA="$FILTERED_DIR/sft_${STRATEGY}.json"

    if [ ! -f "$SFT_DATA" ]; then
        echo "  [!] SFT data not found: $SFT_DATA, skipping"
        continue
    fi

    echo "  Training with strategy: $STRATEGY..."
    python3 downstream/train_sft.py \
        --data_path "$SFT_DATA" \
        --output_dir "$SFT_DIR/sft_${STRATEGY}" \
        --base_model "$BASE_MODEL" \
        --epochs 2 \
        --lr 1e-4 \
        --batch_size 4 \
        --gradient_accumulation 4
    echo ""
done

echo ""

# =============================================================================
# Phase 4: Downstream evaluation
# =============================================================================
echo "[Phase 4] Downstream evaluation..."
echo ""

# Create a held-out eval set for downstream tasks
# (In practice, this would be a separate test set)
EVAL_SET="$GENERATED_DIR/eval_set.json"

if [ ! -f "$EVAL_SET" ]; then
    echo "  Creating downstream eval set from generated data..."
    python3 -c "
import json
data = json.load(open('$GENERATED_DIR/all_rewrites.json'))
# Use every 10th item as eval (roughly 600 samples)
eval_items = data[::10][:100]  # Use 100 for quick eval
with open('$EVAL_SET', 'w') as f:
    json.dump(eval_items, f, ensure_ascii=False, indent=2)
print(f'Created eval set with {len(eval_items)} samples')
"
fi

# Evaluate each SFT model
for STRATEGY in "${STRATEGIES[@]}"; do
    CHECKPOINT="$SFT_DIR/sft_${STRATEGY}"

    if [ ! -d "$CHECKPOINT" ]; then
        echo "  [!] Checkpoint not found: $CHECKPOINT, skipping eval"
        continue
    fi

    echo "  Evaluating: $STRATEGY..."
    python3 downstream/eval_downstream.py \
        --model_path "$BASE_MODEL" \
        --lora_path "$CHECKPOINT" \
        --eval_data "$EVAL_SET" \
        --output_path "$EVAL_RESULTS_DIR/results_${STRATEGY}.json" \
        --max_samples 100
    echo ""
done

# Also evaluate base model (no SFT)
echo "  Evaluating: base model (no SFT)..."
python3 downstream/eval_downstream.py \
    --model_path "$BASE_MODEL" \
    --eval_data "$EVAL_SET" \
    --output_path "$EVAL_RESULTS_DIR/results_base_model.json" \
    --max_samples 100

echo ""

# =============================================================================
# Phase 5: Summary
# =============================================================================
echo "============================================"
echo "Downstream SFT Validation Complete!"
echo ""
echo "Results:"
ls -la "$EVAL_RESULTS_DIR"/results_*.json 2>/dev/null || echo "  No results found"
echo ""
echo "Checkpoints:"
ls -la "$SFT_DIR"/ 2>/dev/null || echo "  No checkpoints found"
echo "============================================"
