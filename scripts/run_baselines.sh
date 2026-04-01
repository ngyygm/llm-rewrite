#!/bin/bash
# =============================================================================
# Run all baselines on the eval set
# EMNLP 2026
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "EMNLP 2026: Running All Baselines"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Create output directories
mkdir -p data/baselines

# Step 1: Traditional metrics (no GPU needed)
echo "[Step 1/4] Running traditional metrics (BLEU, ROUGE, Jaccard, etc.)..."
cd baselines
python3 run_traditional.py \
    --eval_data "$PROJECT_DIR/data/human_eval/eval.json" \
    --output_dir "$PROJECT_DIR/data/baselines"
cd "$PROJECT_DIR"
echo ""

# Step 2: ParaScore (no GPU needed)
echo "[Step 2/4] Running ParaScore..."
cd baselines
python3 run_parascore.py \
    --eval_data "$PROJECT_DIR/data/human_eval/eval.json" \
    --output_dir "$PROJECT_DIR/data/baselines"
cd "$PROJECT_DIR"
echo ""

# Step 3: LLM-based evaluators (requires GPU)
echo "[Step 3/4] Running LLM-based evaluators (G-Eval, zero-shot, prompt-based)..."
echo "  This requires a GPU with ~16GB VRAM"
cd baselines
python3 run_llm_evaluators.py \
    --eval_data "$PROJECT_DIR/data/human_eval/eval.json" \
    --output_dir "$PROJECT_DIR/data/baselines" \
    --model_path "${LOCAL_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
cd "$PROJECT_DIR"
echo ""

# Step 4: Prometheus 2 / M-Prometheus (requires GPU)
echo "[Step 4/4] Running Prometheus 2 / M-Prometheus..."
cd baselines
python3 run_fine_tuned_evaluators.py \
    --eval_data "$PROJECT_DIR/data/human_eval/eval.json" \
    --output_dir "$PROJECT_DIR/data/baselines" \
    --model_path "${PROMETHEUS_MODEL_PATH:-prometheus-eval/prometheus-7b-v2.0}"
cd "$PROJECT_DIR"
echo ""

# Combine all results
echo "[Summary] Combining all results..."
cd baselines
python3 run_all_baselines.py --combine_only
cd "$PROJECT_DIR"

echo ""
echo "============================================"
echo "All baselines completed!"
echo "Results saved to: data/baselines/"
echo "  - all_results.json"
echo "  - results_summary.csv"
echo "============================================"
