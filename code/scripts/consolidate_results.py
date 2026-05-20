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

import argparse
import json
import re
from pathlib import Path
from scipy import stats
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "baselines"

# Parse args first so CHECKPOINTS_DIR can be overridden
parser = argparse.ArgumentParser(description="Consolidate experiment results")
parser.add_argument(
    "--checkpoints-dir", type=str,
    default=str(PROJECT_ROOT / "evaluator" / "checkpoints"),
    help="Path to evaluator checkpoints directory (supports timestamped dirs)",
)
parser.add_argument(
    "--results-dir", type=str,
    default=str(RESULTS_DIR),
    help="Path to baselines results directory",
)
parser.add_argument(
    "--lora-results-dir", type=str,
    default=None,
    help="Directory containing LoRA eval result files "
         "(e.g., results_lora_score_only_balanced_full.json). "
         "If not set, uses --results-dir.",
)
parser.add_argument(
    "--extra-lora-results-dir", action="append", default=[],
    help="Additional LoRA results directories to include as separate model families. "
         "Can be specified multiple times.",
)
_args = parser.parse_args()

CHECKPOINTS_DIR = Path(_args.checkpoints_dir)
RESULTS_DIR = Path(_args.results_dir)
LORA_RESULTS_DIR = Path(_args.lora_results_dir) if _args.lora_results_dir else RESULTS_DIR
EXTRA_LORA_RESULTS_DIRS = [Path(p) for p in _args.extra_lora_results_dir]

print(f"Checkpoints dir : {CHECKPOINTS_DIR}")
print(f"Results dir     : {RESULTS_DIR}")
print(f"LoRA results dir: {LORA_RESULTS_DIR}")
if EXTRA_LORA_RESULTS_DIRS:
    print(f"Extra LoRA dirs : {', '.join(str(p) for p in EXTRA_LORA_RESULTS_DIRS)}")

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


def _sanitize_tag(text):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    return s or "extra"


def _resolve_results_file(results_dir, file_name_candidates):
    for name in file_name_candidates:
        p = results_dir / name
        if p.exists():
            return p
    return None


def _resolve_predictions_path(results_file):
    try:
        result_obj = json.load(open(results_file))
    except Exception as e:
        print(f"  [skip] Failed reading {results_file}: {e}")
        return None

    adapter_path = result_obj.get("adapter_path")
    if not adapter_path:
        print(f"  [skip] No adapter_path in {results_file}")
        return None

    adapter_path = Path(adapter_path)
    adapter_candidates = []
    if adapter_path.is_absolute():
        adapter_candidates.append(adapter_path)
    else:
        adapter_candidates.append(PROJECT_ROOT / adapter_path)
        adapter_candidates.append(results_file.parent / adapter_path)

    pred_candidates = []
    for a in adapter_candidates:
        pred_candidates.append(a / "eval_predictions.json")
        pred_candidates.append(a / "best_spearman_checkpoint" / "eval_predictions.json")

    for p in pred_candidates:
        if p.exists():
            return p

    print(f"  [skip] Missing eval_predictions.json for adapter_path={adapter_path} from {results_file}")
    return None


def add_lora_from_results_file(results_dir, file_name_candidates, method_key, display_name, size, data_info):
    """
    Read LoRA predictions from a results file in LORA_RESULTS_DIR by using:
      results_lora_*.json -> adapter_path -> eval_predictions.json
    """
    results_file = _resolve_results_file(results_dir, file_name_candidates)
    if not results_file:
        joined = ", ".join(file_name_candidates)
        print(f"  [skip] Missing LoRA result files in {results_dir}: [{joined}]")
        return

    pred_path = _resolve_predictions_path(results_file)
    if not pred_path:
        return

    try:
        preds = json.load(open(pred_path))
        scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in preds]
        add_method(method_key, scores, display_name, "lora", size, data_info)
    except Exception as e:
        print(f"  [skip] Failed parsing predictions from {pred_path}: {e}")


print(f"Eval samples: {n_eval}\n")

# ============================================================
# 1. Our LoRA Models
# ============================================================
print("=== LoRA Fine-tuned Models ===")

add_lora_from_results_file(
    LORA_RESULTS_DIR,
    [
        "results_lora_score_only_detail_balanced_full.json",
        "results_lora_score_only_balanced_full.json",
        "results_balanced_simple.json",
    ],
    "lora_balanced_simple",
    "RewriteJudge (Ours)",
    "8B",
    "1008 balanced",
)
add_lora_from_results_file(
    LORA_RESULTS_DIR,
    [
        "results_lora_score_only_detail_unbalanced_full.json",
        "results_lora_score_only_unbalanced_full.json",
        "results_unbalanced_simple.json",
    ],
    "lora_score_only_unbalanced",
    "Unbalanced training",
    "8B",
    "600 original",
)
add_lora_from_results_file(
    LORA_RESULTS_DIR,
    ["results_lora_multi_score_full.json", "results_multi_score_full.json"],
    "lora_multi_score",
    "LoRA-multi-score",
    "8B",
    "multi score",
)

# Reasoning prefix: prefer results JSON under LORA_RESULTS_DIR (e.g. qwen3-8b_.../results_lora_score_detail_reasoning_full.json)
# so figures match the timestamped baseline run. Legacy hard-coded checkpoint only if that fails.
add_lora_from_results_file(
    LORA_RESULTS_DIR,
    [
        "results_lora_score_detail_reasoning_full.json",
        "results_lora_score_only_detail_reasoning_full.json",
        "results_lora_score_only_reasoning_full.json",
    ],
    "lora_balanced_reasoning",
    "+ Reasoning prefix",
    "8B",
    "reasoning prefix",
)
if "lora_balanced_reasoning" not in all_scores:
    reasoning_pred_path = PROJECT_ROOT / "evaluator" / "checkpoints" / \
        "qwen3-8b_reasoning_prefix_20260412_235440" / "score_reasoning_full" / \
        "best_spearman_checkpoint" / "eval_predictions.json"
    if reasoning_pred_path.exists():
        preds = json.load(open(reasoning_pred_path))
        scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in preds]
        add_method("lora_balanced_reasoning", scores, "+ Reasoning prefix", "lora", "8B", "reasoning prefix (legacy checkpoint)")

# Learning curve subsets (read from timestamped results dir + adapter_path)
for subset_size in [50, 100, 200, 400, 500]:
    add_lora_from_results_file(
        LORA_RESULTS_DIR,
        [
            f"results_lora_score_only_detail_balanced_{subset_size}.json",
            f"results_lora_score_only_balanced_{subset_size}.json",
            f"results_balanced_simple_{subset_size}.json",
            f"results_lora_score_only_{subset_size}.json",
        ],
        f"lora_balanced_simple_{subset_size}",
        f"RewriteJudge-{subset_size}",
        "8B",
        f"{subset_size} balanced subset",
    )

# Optional: include extra model families (e.g., qwen2.5-7B) as separate keys.
for extra_dir in EXTRA_LORA_RESULTS_DIRS:
    tag = _sanitize_tag(extra_dir.name)
    model_label = extra_dir.name
    print(f"\n=== Extra LoRA family: {model_label} ===")
    add_lora_from_results_file(
        extra_dir,
        [
            "results_lora_score_only_detail_balanced_full.json",
            "results_lora_score_only_balanced_full.json",
            "results_balanced_simple.json",
        ],
        f"lora_balanced_simple_{tag}",
        f"RewriteJudge ({model_label})",
        model_label,
        "balanced full",
    )
    add_lora_from_results_file(
        extra_dir,
        [
            "results_lora_score_only_detail_unbalanced_full.json",
            "results_lora_score_only_unbalanced_full.json",
            "results_unbalanced_simple.json",
        ],
        f"lora_score_only_unbalanced_{tag}",
        f"Unbalanced training ({model_label})",
        model_label,
        "unbalanced full",
    )
    for subset_size in [50, 100, 200, 400, 500]:
        add_lora_from_results_file(
            extra_dir,
            [
                f"results_lora_score_only_detail_balanced_{subset_size}.json",
                f"results_lora_score_only_balanced_{subset_size}.json",
                f"results_balanced_simple_{subset_size}.json",
                f"results_lora_score_only_{subset_size}.json",
            ],
            f"lora_balanced_simple_{subset_size}_{tag}",
            f"RewriteJudge-{subset_size} ({model_label})",
            model_label,
            f"{subset_size} subset",
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

qwen_path = RESULTS_DIR / "llm_zero_shot_Qwen2.5-14B-Instruct.json"
if qwen_path.exists():
    preds = json.load(open(qwen_path))
    sr = preds if isinstance(preds, list) else preds.get("sample_results", [])
    sr_sorted = sorted(sr, key=lambda x: x["idx"]) if sr and "idx" in sr[0] else sr
    scores = [p.get("predicted_score", -1) if p.get("predicted_score") is not None else -1 for p in sr_sorted]
    add_method("zeroshot_qwen14b", scores, "Zero-shot Qwen2.5-14B", "llm", "14B", "Zero-shot prompt")

# G-Eval Qwen3-8B
geval_qwen3_path = RESULTS_DIR / "llm_geval_Qwen3-8B.json"
if geval_qwen3_path.exists():
    d = json.load(open(geval_qwen3_path))
    sr = sorted(d.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p.get("predicted_score", -1) if p.get("predicted_score") is not None else -1 for p in sr]
    add_method("geval_qwen3_8b", scores, "G-Eval (Qwen3-8B)", "llm", "8B", "CoT evaluation")

# Zero-shot Qwen3-8B
qwen3_8b_path = RESULTS_DIR / "llm_zero_shot_Qwen3-8B.json"
if qwen3_8b_path.exists():
    d = json.load(open(qwen3_8b_path))
    sr = sorted(d.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p.get("predicted_score", -1) if p.get("predicted_score") is not None else -1 for p in sr]
    add_method("zeroshot_qwen3_8b", scores, "Zero-shot Qwen3-8B", "llm", "8B", "Zero-shot prompt")

# Zero-shot Qwen3-14B
qwen3_14b_path = RESULTS_DIR / "llm_zero_shot_Qwen3-14B.json"
if qwen3_14b_path.exists():
    d = json.load(open(qwen3_14b_path))
    sr = sorted(d.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p.get("predicted_score", -1) if p.get("predicted_score") is not None else -1 for p in sr]
    add_method("zeroshot_qwen3_14b", scores, "Zero-shot Qwen3-14B", "llm", "14B", "Zero-shot prompt")

# Zero-shot Qwen2.5-7B（保留作对比）
zs_qwen25_7b = RESULTS_DIR / "llm_zero_shot_Qwen2.5-7B-Instruct.json"
if zs_qwen25_7b.exists():
    d = json.load(open(zs_qwen25_7b))
    sr = sorted(d.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p.get("predicted_score", -1) if p.get("predicted_score") is not None else -1 for p in sr]
    add_method("zeroshot_qwen7b", scores, "Zero-shot Qwen2.5-7B", "llm", "7B", "Zero-shot prompt")

# Prometheus 2
prom_path = RESULTS_DIR / "finetuned_Prometheus-2-7B.json"
if prom_path.exists():
    prom_data = json.load(open(prom_path))
    sr = sorted(prom_data.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in sr]
    add_method("prometheus2", scores, "Prometheus 2", "llm", "7B", "Fine-tuned judge, absolute grading")

# M-Prometheus
mprom_path = RESULTS_DIR / "finetuned_M-Prometheus-7B.json"
if mprom_path.exists():
    mprom_data = json.load(open(mprom_path))
    sr = sorted(mprom_data.get("sample_results", []), key=lambda x: x["idx"])
    scores = [p["predicted_score"] if p["predicted_score"] is not None else -1 for p in sr]
    add_method("m_prometheus", scores, "M-Prometheus", "llm", "7B", "Multilingual fine-tuned judge")

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
