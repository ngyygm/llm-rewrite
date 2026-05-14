#!/bin/bash
# =============================================================================
# Downstream SFT Validation - Data generation, filtering, training, evaluation
# EMNLP 2026
# =============================================================================
set -euo pipefail
export PYTHONPATH=/home/hadoop-ai-search/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
MODEL_NAME="qwen3-8b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_MODEL="${LOCAL_MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B}"
EVALUATOR_CHECKPOINT="${EVALUATOR_ADAPTER:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/baokailin/github.com/ngyygm/llm-rewrite.git/evaluator/checkpoints/qwen3-8b_20260511_154259/score_only_detail_balanced_full}"
API_MODEL="${API_MODEL:-gpt-4.1}"
API_KEY="${API_KEY:-}"
API_URL="${API_URL:-}"
GENERATED_DIR="$PROJECT_DIR/data/generated_rewrites"
FILTERED_DIR="$GENERATED_DIR/${API_MODEL}/filtered"
SFT_DIR="$PROJECT_DIR/downstream/checkpoints/${MODEL_NAME}_20260412_160923"
EVAL_RESULTS_DIR="$PROJECT_DIR/data/downstream_results/${MODEL_NAME}_${TIMESTAMP}"
MAX_WORKERS="${MAX_WORKERS:-10}"

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
SOURCE_COUNT=$(python3 -c "import json; print(len(json.load(open('$GENERATED_DIR/${API_MODEL}/source_texts.json'))))" 2>/dev/null || echo "0")
if [ ! -f "$GENERATED_DIR/${API_MODEL}/source_texts.json" ] || [ "$SOURCE_COUNT" -eq 0 ]; then
    echo "  [1a] Generating 300 source texts via API ($API_URL, model: $API_MODEL)..."
    python3 downstream/generate_data.py \
        --mode api \
        --api_url "$API_URL" \
        --api_model "$API_MODEL" \
        --api_key "$API_KEY" \
        --max_workers "$MAX_WORKERS" \
        --output_dir "$GENERATED_DIR/${API_MODEL}"
else
    echo "  [1a] Source texts already exist, skipping"
fi

# # Step 1b: Generate rewrites
REWRITE_COUNT=$(python3 -c "import json; print(len(json.load(open('$GENERATED_DIR/${API_MODEL}/all_rewrites.json'))))" 2>/dev/null || echo "0")
if [ ! -f "$GENERATED_DIR/${API_MODEL}/all_rewrites.json" ] || [ "$REWRITE_COUNT" -eq 0 ]; then
    echo "  [1b] Generating 900 rewrites (3 per source) via API..."
    python3 downstream/generate_data.py \
        --mode api \
        --api_url "$API_URL" \
        --api_model "$API_MODEL" \
        --api_key "$API_KEY" \
        --max_workers "$MAX_WORKERS" \
        --output_dir "$GENERATED_DIR/${API_MODEL}" \
        --skip_source_gen
else
    echo "  [1b] Rewrites already exist, skipping"
fi

echo ""

# =============================================================================
# Phase 2: Score rewrites with evaluator, then filter
# =============================================================================
echo "[Phase 2] Scoring and filtering data..."
echo ""

# Step 2a: Score all 900 rewrites with RewriteJudge
SCORED_PATH="$GENERATED_DIR/${API_MODEL}/scored_rewrites_new.json"
if [ ! -f "$SCORED_PATH" ]; then
    echo "  [2a] Scoring 900 rewrites with RewriteJudge..."
    python3 downstream/score_rewrites.py \
        --evaluator_path "$EVALUATOR_CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --rewrites_path "$GENERATED_DIR/${API_MODEL}/all_rewrites.json" \
        --prompt_variant detail \
        --output_path "$SCORED_PATH"
else
    echo "  [2a] Scored rewrites already exist, skipping"
fi

Step 2b: Filter using all strategies (k=450 = top 50% of 900)
echo "  [2b] Filtering with all strategies..."
python3 downstream/filter_data.py \
    --rewrites_path "$GENERATED_DIR/${API_MODEL}/all_rewrites.json" \
    --scores_path "$SCORED_PATH" \
    --strategy all \
    --k 450 \
    --threshold 3.0 \
    --output_dir "$FILTERED_DIR"

echo ""

# =============================================================================
# Phase 3: SFT training (one model per filtering strategy)
# =============================================================================
echo "[Phase 3] SFT training..."
echo ""

STRATEGIES=("random_450" "bleu_filtered" "top_450" "threshold_3.0")

for STRATEGY in "${STRATEGIES[@]}"; do
    SFT_DATA="$FILTERED_DIR/sft_${STRATEGY}.json"

    if [ ! -f "$SFT_DATA" ]; then
        echo "  [!] SFT data not found: $SFT_DATA, skipping"
        continue
    fi

    # Skip empty datasets
    N=$(python3 -c "import json; print(len(json.load(open('$SFT_DATA'))))")
    if [ "$N" -eq 0 ]; then
        echo "  [!] SFT data is empty: $SFT_DATA, skipping"
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
STRATEGIES=("random_450" "bleu_filtered" "top_450" "threshold_3.0")

echo "[Phase 4] Downstream evaluation..."
echo ""

# Create a held-out eval set for downstream tasks
# (In practice, this would be a separate test set)
EVAL_SET="$GENERATED_DIR/${API_MODEL}/eval_set.json"

if [ ! -f "$EVAL_SET" ]; then
    echo "  Creating downstream eval set from generated data..."
    python3 -c "
import json
data = json.load(open('$GENERATED_DIR/${API_MODEL}/all_rewrites.json'))
# Use every 10th item as eval (90 samples from 900)
eval_items = data[::10]
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
