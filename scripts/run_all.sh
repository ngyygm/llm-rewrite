#!/bin/bash
# =============================================================================
# EMNLP 2026: Complete Experiment Pipeline
#
# Runs all experiments in order:
#   1. Data preparation (already done by convert_data.py)
#   2. Baseline evaluations
#   3. LoRA evaluator training + evaluation
#   4. Downstream SFT validation
#   5. Analysis & figure generation
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  EMNLP 2026: Complete Experiment Pipeline    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo ""

# Configuration (override with environment variables)
# export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen/Qwen2.5-7B-Instruct}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B}"
export PROMETHEUS_MODEL_PATH="${PROMETHEUS_MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/baokailin/train_model/prometheus-7b-v2.0}"
export API_URL="${API_URL:-http://localhost:8000}"
export EVALUATOR_ADAPTER="${EVALUATOR_ADAPTER:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/baokailin/github.com/ngyygm/llm-rewrite.git/evaluator/checkpoints/balanced_simple/checkpoint-189}"

# Track total time
START_TIME=$(date +%s)

# ==========================================================================
# Step 0: Verify data preparation
# ==========================================================================
echo "[Step 0] Verifying data preparation..."
if [ ! -f "data/human_eval/train_score_only_balanced.json" ] || [ ! -f "data/human_eval/eval.json" ]; then
    echo "  Running data conversion..."
    python3 scripts/convert_data.py
else
    echo "  Data already prepared"
fi
echo ""

# ==========================================================================
# Step 1: Run baselines
# ==========================================================================
echo "[Step 1/5] Running baseline evaluations..."
bash scripts/run_baselines.sh
echo ""

# ==========================================================================
# Step 2: Train LoRA evaluator
# ==========================================================================
echo "[Step 2/5] Training LoRA evaluator..."
bash scripts/run_evaluator_training.sh
echo ""

# ==========================================================================
# Step 3: Downstream SFT validation
# ==========================================================================
echo "[Step 3/5] Running downstream SFT validation..."
bash scripts/run_downstream.sh
echo ""

# ==========================================================================
# Step 4: Analysis & figures
# ==========================================================================
echo "[Step 4/5] Generating analysis and figures..."
mkdir -p analysis/results analysis/figures

python3 analysis/correlation_analysis.py \
    --eval-data data/human_eval/eval.json \
    --all-results data/baselines/all_results.json \
    --metadata data/baselines/method_metadata.json \
    --output-dir analysis/results

python3 analysis/learning_curves.py \
    --all-results data/baselines/all_results.json \
    --metadata data/baselines/method_metadata.json \
    --eval-data data/human_eval/eval.json \
    --output-dir analysis/figures

python3 analysis/bias_analysis.py \
    --eval-data data/human_eval/eval.json \
    --figures-dir analysis/figures \
    --results-dir analysis/results

python3 analysis/generate_figures.py \
    --eval-data data/human_eval/eval.json \
    --all-results data/baselines/all_results.json \
    --metadata data/baselines/method_metadata.json \
    --trad-results data/baselines/all_results_traditional.json \
    --figures-dir analysis/qwen3-8B/figures \
    --tables-dir analysis/qwen3-8B/results
# echo ""

# ==========================================================================
# Step 5: Summary
# ==========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo "╔══════════════════════════════════════════════╗"
echo "║  All Experiments Complete!                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Total time: ${MINUTES} minutes"
echo ""
echo "Output locations:"
echo "  Data:          data/human_eval/"
echo "  Baseline results: data/baselines/"
echo "  Evaluator:     evaluator/checkpoints/"
echo "  SFT models:    downstream/checkpoints/"
echo "  Analysis:      analysis/results/"
echo "  Figures:       analysis/figures/"
echo ""
echo "Key files:"
echo "  data/baselines/all_results.json  - Combined baseline results"
echo "  data/baselines/results_summary.csv - CSV summary table"
echo "  data/baselines/learning_curves.json - Learning curve data"
echo "  analysis/figures/ - All paper figures"
echo "============================================"
