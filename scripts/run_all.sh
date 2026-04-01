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
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export PROMETHEUS_MODEL_PATH="${PROMETHEUS_MODEL_PATH:-prometheus-eval/prometheus-7b-v2.0}"
export API_URL="${API_URL:-http://localhost:8000}"
export EVALUATOR_ADAPTER="${EVALUATOR_ADAPTER:-$PROJECT_DIR/evaluator/checkpoints/score_only_full}"

# Track total time
START_TIME=$(date +%s)

# ==========================================================================
# Step 0: Verify data preparation
# ==========================================================================
echo "[Step 0] Verifying data preparation..."
if [ ! -f "data/human_eval/train_score_only.json" ] || [ ! -f "data/human_eval/eval.json" ]; then
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
    --results_dir data/baselines \
    --output_dir analysis/results

python3 analysis/learning_curves.py \
    --data_path data/baselines/learning_curves.json \
    --output_dir analysis/figures

python3 analysis/bias_analysis.py \
    --eval_data data/human_eval/eval.json \
    --evaluator_results data/baselines/results_lora_score_only_full.json \
    --output_dir analysis/figures

python3 analysis/generate_figures.py \
    --results_dir analysis/results \
    --output_dir analysis/figures
echo ""

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
