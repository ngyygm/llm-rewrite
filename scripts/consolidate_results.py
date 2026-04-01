#!/usr/bin/env python3
"""
Consolidate all experiment results into a unified format for analysis.

Output: data/baselines/all_results.json
  {
    "method_name": [score1, score2, ...],  # -1 for parse failures
    ...
  }

Also generates: data/baselines/method_metadata.json
  {
    "method_name": {"display_name": "...", "category": "...", "size": "...", ...},
    ...
  }
"""

import json
from pathlib import Path
from scipy import stats
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "baselines"
CHECKPOINTS_DIR = PROJECT_ROOT / "evaluator" / "checkpoints"

# Load eval data
with open(PROJECT_ROOT / "data" / "human_eval" / "eval.json") as f:
    eval_data = json.load(f)
n_eval = len(eval_data)

human_consensus = [item["consensus_score"] for item in eval_data]
human_avg = [item["avg_score"] for item in eval_data]

all_scores = {}
metadata = {}


def add_method(name, scores, display_name, category, size, notes=""):
    assert len(scores) == n_eval, f"{name}: {len(scores)} != {n_eval}"
    valid = [s for s in scores if s >= 0]
    preds = np.array([s if s >= 0 else np.nan for s in scores], dtype=float)
    refs = np.array(human_avg, dtype=float)
    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() >= 2:
        spearman, _ = stats.spearmanr(preds[valid_mask], refs[valid_mask])
    else:
        spearman = 0.0
    all_scores[name] = scores
    metadata[name] = {
        "display_name": display_name,
        "category": category,
        "size": size,
        "valid": len(valid),
        "failures": len(scores) - len(valid),
        "spearman_vs_avg": round(float(spearman), 4),
        "notes": notes,
    }
    print(f"  {name:35s} | {display_name:40s} | valid={len(valid):3d} | rho={spearman:+.4f}")


print(f"Eval samples: {n_eval}\n")

# ============================================================
# 1. Our LoRA Models
# ============================================================
print("=== LoRA Fine-tuned Models ===")

lora_variants = [
    ("balanced_simple", "LoRA-balanced-simple (Ours)", "7B", "1008 balanced"),
    ("balanced_reasoning", "LoRA-balanced-reasoning", "7B", "1008 balanced"),
    ("original_reasoning", "LoRA-original-reasoning", "7B", "600 original"),
    ("score_only_full", "LoRA-score-only-full", "7B", "600 original"),
]

for ckpt_name, display, size, data_info in lora_variants:
    pred_path = CHECKPOINTS_DIR / ckpt_name / "eval_predictions.json"
    if pred_path.exists():
        preds = json.load(open(pred_path))
        scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in preds]
        add_method(f"lora_{ckpt_name}", scores, display, "lora", size, data_info)

# Learning curve subsets
for subset_size in [50, 100, 200, 400]:
    pred_path = CHECKPOINTS_DIR / f"balanced_simple_{subset_size}" / "eval_predictions.json"
    if pred_path.exists():
        preds = json.load(open(pred_path))
        scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in preds]
        add_method(
            f"lora_balanced_simple_{subset_size}", scores,
            f"LoRA-balanced-simple-{subset_size}", "lora", "7B",
            f"{subset_size} balanced subset",
        )

# ============================================================
# 2. LLM-based Baselines
# ============================================================
print("\n=== LLM-based Baselines ===")

# G-Eval Qwen2.5-7B
geval_path = RESULTS_DIR / "llm_geval_Qwen2.5-7B-Instruct.json"
if geval_path.exists():
    geval = json.load(open(geval_path))
    sr = geval.get("sample_results", [])
    if sr:
        # Ensure ordered by idx
        sr_sorted = sorted(sr, key=lambda x: x["idx"])
        scores = [item.get("predicted_score", -1) for item in sr_sorted]
        add_method("geval_qwen7b", scores, "G-Eval (Qwen2.5-7B)", "llm", "7B", "CoT evaluation")

# Zero-shot Qwen2.5-7B
zs_path = RESULTS_DIR / "llm_zero_shot_Qwen2.5-7B-Instruct.json"
if zs_path.exists():
    zs = json.load(open(zs_path))
    sr = zs.get("sample_results", [])
    if sr:
        sr_sorted = sorted(sr, key=lambda x: x["idx"])
        scores = [item.get("predicted_score", -1) for item in sr_sorted]
        add_method("zeroshot_qwen7b", scores, "Zero-shot Qwen2.5-7B", "llm", "7B", "Zero-shot prompt")

# Qwen2.5-14B Zero-shot
qwen_path = RESULTS_DIR / "results_qwen14b_zeroshot_predictions.json"
if qwen_path.exists():
    preds = json.load(open(qwen_path))
    scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in preds]
    add_method("zeroshot_qwen14b", scores, "Zero-shot Qwen2.5-14B", "llm", "14B", "Zero-shot prompt")

# Prometheus 2 - v2 run with per-sample predictions saved
prom_v2_pred_path = RESULTS_DIR / "results_prometheus2_v2_predictions.json"
if prom_v2_pred_path.exists():
    prom_preds = json.load(open(prom_v2_pred_path))
    # Ensure ordered by index
    prom_sorted = sorted(prom_preds, key=lambda x: x["index"])
    scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in prom_sorted]
    add_method("prometheus2", scores, "Prometheus 2", "llm", "7B", "Fine-tuned judge, absolute grading")

# ============================================================
# 3. Traditional Metrics (continuous values, not 0-5 scores)
# ============================================================
print("\n=== Traditional Metrics (continuous similarity) ===")

trad_path = RESULTS_DIR / "traditional_metrics.json"
if trad_path.exists():
    trad = json.load(open(trad_path))
    sr = trad.get("sample_results", [])
    if sr:
        # These are continuous similarity scores, not discrete 0-5
        # We store them separately since they need different handling
        trad_scores = {}
        for metric_name in ["jaccard_char", "jaccard_word", "bleu", "rouge_l", "tfidf_cosine", "sbert_cosine", "w2v_cosine"]:
            scores = [item.get(metric_name, -1) for item in sr]
            # Compute Spearman vs human scores
            refs_arr = np.array(human_avg, dtype=float)
            preds_arr = np.array(scores, dtype=float)
            valid_mask = ~(preds_arr < 0)
            if valid_mask.sum() >= 2:
                rho, _ = stats.spearmanr(preds_arr[valid_mask], refs_arr[valid_mask])
            else:
                rho = 0.0
            trad_scores[f"trad_{metric_name}"] = scores
            metadata[f"trad_{metric_name}"] = {
                "display_name": metric_name.replace("_", "-").upper(),
                "category": "traditional",
                "size": "-",
                "valid": int(valid_mask.sum()),
                "failures": 0,
                "spearman_vs_avg": round(float(rho), 4),
                "continuous": True,
                "notes": f"Continuous similarity metric (range varies)",
            }
            print(f"  trad_{metric_name:25s} | {metric_name.replace('_', '-').upper():40s} | rho={rho:+.4f}")

        # Save traditional metrics separately
        trad_out = RESULTS_DIR / "all_results_traditional.json"
        with open(trad_out, "w") as f:
            json.dump(trad_scores, f, indent=2)
        print(f"\n  Traditional metrics saved to: {trad_out}")

# ============================================================
# 4. Save consolidated results
# ============================================================
out_path = RESULTS_DIR / "all_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_scores, f, indent=2, ensure_ascii=False)
print(f"\nScores saved to: {out_path}")

meta_path = RESULTS_DIR / "method_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadata saved to: {meta_path}")

# ============================================================
# 5. Summary table
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: All Evaluator Results (Spearman ρ vs avg_score)")
print("=" * 80)
print(f"{'Method':45s} | {'Size':>4s} | {'Valid':>5s} | {'Spearman':>9s}")
print("-" * 80)

# Sort by Spearman
sorted_methods = sorted(metadata.items(), key=lambda x: x[1].get("spearman_vs_avg", 0), reverse=True)
for name, meta in sorted_methods:
    size = meta.get("size", "?")
    valid = meta.get("valid", "?")
    rho = meta.get("spearman_vs_avg", 0)
    is_ours = "lora_balanced_simple" == name and not name.endswith(("50", "100", "200", "400"))
    marker = " ***" if is_ours else ""
    print(f"{meta['display_name']:45s} | {str(size):>4s} | {str(valid):>5s} | {rho:+.4f}{marker}")

print("=" * 80)
