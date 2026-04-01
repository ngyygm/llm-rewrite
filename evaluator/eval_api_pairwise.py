#!/usr/bin/env python3
"""
API-based Pairwise Cross-Source Evaluation using SiliconFlow.

Multi-threaded with incremental saving (resume-safe).

Usage:
    # Quick sanity check
    python evaluator/eval_api_pairwise.py \
        --model Qwen/Qwen2.5-72B-Instruct --max_pairs 5

    # Full eval, 16 threads, skip position swap
    python evaluator/eval_api_pairwise.py \
        --model Qwen/Qwen2.5-72B-Instruct --workers 16 --skip_swap

    # Resume interrupted run (auto-detects checkpoint)
    python evaluator/eval_api_pairwise.py \
        --model Qwen/Qwen2.5-72B-Instruct --workers 16 --skip_swap

EMNLP 2026
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.eval_pairwise import (
    SYSTEM_PROMPT_PAIRWISE,
    CROSS_SOURCE_USER_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("eval_api_pairwise")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pairwise_output(text: str) -> float:
    """Parse model output to get preference (extended for API responses)."""
    text = text.strip()

    if len(text) <= 5:
        text_upper = text.upper().strip()
        if text_upper == "A":
            return 1.0
        if text_upper == "B":
            return 0.0

    if text and text[0] in "Aa":
        return 1.0
    if text and text[0] in "Bb":
        return 0.0

    # Explicit "改写X的...更高/更好" patterns
    for pa in ["改写A的质量更高", "改写A的质量更好", "改写A更优", "改写A更胜"]:
        if pa in text:
            return 1.0
    for pb in ["改写B的质量更高", "改写B的质量更好", "改写B更优", "改写B更胜"]:
        if pb in text:
            return 0.0

    # Broader: "改写X" + preference verb within 20 chars
    for m in re.finditer(r"改写([AB])[^。？！\n]{0,20}(更高|更好|更优|更胜|略胜|更强|更佳)", text):
        if m.group(1) == "A":
            return 1.0
        if m.group(1) == "B":
            return 0.0

    # Signal counting
    a_indicators = ["改写A", "A更好", "A更优", "选择A", "选A"]
    b_indicators = ["改写B", "B更好", "B更优", "选择B", "选B"]
    a_signals = sum(1 for x in a_indicators if x in text)
    b_signals = sum(1 for x in b_indicators if x in text)

    prefer_verbs = ["更高", "更好", "更优", "更胜一筹", "更强", "更佳", "略胜"]
    has_prefer = any(v in text for v in prefer_verbs)

    if has_prefer:
        if a_signals > b_signals and a_signals > 0:
            return 1.0
        if b_signals > a_signals and b_signals > 0:
            return 0.0

    tie_indicators = ["平局", "相当", "一样", "不相上下", "无法区分"]
    if any(t in text for t in tie_indicators):
        return 0.5

    return -1.0


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"


def call_api(messages, model, api_key, max_tokens=50, temperature=0.1,
             max_retries=5, retry_delay=2.0):
    """Call SiliconFlow API with retry logic."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                return ""
    return ""


# ---------------------------------------------------------------------------
# Build pairs (deterministic, same as eval_pairwise.py)
# ---------------------------------------------------------------------------

def build_pairs(eval_data, min_score_diff):
    pairs = []
    for i in range(len(eval_data)):
        for j in range(i + 1, len(eval_data)):
            sa, sb = eval_data[i]["avg_score"], eval_data[j]["avg_score"]
            if abs(sa - sb) >= min_score_diff:
                pairs.append((i, j, sa > sb))
            elif abs(sa - sb) >= 0.5:
                pairs.append((i, j, sa > sb))
    return pairs


# ---------------------------------------------------------------------------
# Process one pair (thread-safe)
# ---------------------------------------------------------------------------

def process_pair(args_tuple):
    """Process a single pair. Returns (pair_idx, result_dict)."""
    (pair_idx, i, j, a_should_win, score_a, score_b,
     source_a, rewrite_a, source_b, rewrite_b,
     model, api_key, max_tokens, temperature, skip_swap) = args_tuple

    score_diff = abs(score_a - score_b)

    # Build prompt
    user_content = CROSS_SOURCE_USER_PROMPT_TEMPLATE.format(
        source_a=source_a, rewrite_a=rewrite_a,
        source_b=source_b, rewrite_b=rewrite_b,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PAIRWISE},
        {"role": "user", "content": user_content},
    ]

    # Forward call
    resp_ab = call_api(messages, model, api_key, max_tokens, temperature)
    pref_ab = parse_pairwise_output(resp_ab)

    if skip_swap:
        parse_ok = pref_ab >= 0
        avg_pref = pref_ab
        result = {
            "index_a": i, "index_b": j,
            "avg_score_a": score_a, "avg_score_b": score_b,
            "score_diff": round(score_diff, 3),
            "a_should_win": a_should_win,
            "response_ab": resp_ab, "response_ba": "",
            "pref_ab": pref_ab, "pref_ba": -1.0,
            "avg_preference": avg_pref,
            "parse_ok": parse_ok,
        }
    else:
        # Swapped: B pair vs A pair
        user_swap = CROSS_SOURCE_USER_PROMPT_TEMPLATE.format(
            source_a=source_b, rewrite_a=rewrite_b,
            source_b=source_a, rewrite_b=rewrite_a,
        )
        messages_swap = [
            {"role": "system", "content": SYSTEM_PROMPT_PAIRWISE},
            {"role": "user", "content": user_swap},
        ]
        resp_ba = call_api(messages_swap, model, api_key, max_tokens, temperature)
        pref_ba = parse_pairwise_output(resp_ba)

        parse_ok = (pref_ab >= 0) and (pref_ba >= 0)
        avg_pref = (pref_ab + (1.0 - pref_ba)) / 2.0 if parse_ok else -1.0

        result = {
            "index_a": i, "index_b": j,
            "avg_score_a": score_a, "avg_score_b": score_b,
            "score_diff": round(score_diff, 3),
            "a_should_win": a_should_win,
            "response_ab": resp_ab, "response_ba": resp_ba,
            "pref_ab": pref_ab, "pref_ba": pref_ba,
            "avg_preference": avg_pref,
            "parse_ok": parse_ok,
        }

    return (pair_idx, result)


# ---------------------------------------------------------------------------
# Incremental save + compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(pair_results, eval_data, min_score_diff, skip_swap):
    """Compute all metrics from pair_results list."""
    rewrite_stats = defaultdict(lambda: {"wins": 0.0, "comparisons": 0})
    parse_failures = 0
    correct_count = 0
    valid_count = 0
    easy_c = easy_t = med_c = med_t = hard_c = hard_t = 0

    for pr in pair_results:
        i, j = pr["index_a"], pr["index_b"]
        sa, sb = pr["avg_score_a"], pr["avg_score_b"]
        sd = pr["score_diff"]

        if sd >= 2:
            easy_t += 1
        elif sd >= 1:
            med_t += 1
        else:
            hard_t += 1

        if pr["parse_ok"]:
            valid_count += 1
            avg_pref = pr["avg_preference"]
            predicted_a_wins = avg_pref > 0.5
            predicted_tie = abs(avg_pref - 0.5) < 1e-6

            if predicted_a_wins == pr["a_should_win"] or predicted_tie:
                correct_count += 1
                if sd >= 2: easy_c += 1
                elif sd >= 1: med_c += 1
                else: hard_c += 1

            if avg_pref > 0.5:
                rewrite_stats[i]["wins"] += 1.0
            elif avg_pref < 0.5:
                rewrite_stats[j]["wins"] += 1.0
            rewrite_stats[i]["comparisons"] += 1
            rewrite_stats[j]["comparisons"] += 1
        else:
            parse_failures += 1
            rewrite_stats[i]["comparisons"] += 1
            rewrite_stats[j]["comparisons"] += 1

    per_rewrite_win_rates = {}
    for idx, stats in rewrite_stats.items():
        wr = stats["wins"] / stats["comparisons"] if stats["comparisons"] > 0 else 0.0
        per_rewrite_win_rates[idx] = {
            "avg_score": eval_data[idx]["avg_score"],
            "win_rate": round(wr, 4),
            "wins": stats["wins"],
            "comparisons": stats["comparisons"],
        }

    from scipy import stats as scipy_stats
    idx_list = sorted(per_rewrite_win_rates.keys())
    wr_arr = [per_rewrite_win_rates[idx]["win_rate"] for idx in idx_list]
    sc_arr = [per_rewrite_win_rates[idx]["avg_score"] for idx in idx_list]

    if len(wr_arr) >= 3:
        spearman_rho, sp_p = scipy_stats.spearmanr(wr_arr, sc_arr)
        kendall_tau, kt_p = scipy_stats.kendalltau(wr_arr, sc_arr)
    else:
        spearman_rho = kendall_tau = 0.0
        sp_p = kt_p = 1.0

    n = len(pair_results)
    return {
        "total_pairs": n,
        "valid_pairs": valid_count,
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / max(n, 1), 4),
        "pairwise_accuracy": round(correct_count / max(valid_count, 1), 4),
        "accuracy_easy_diff_ge2": round(easy_c / max(easy_t, 1), 4),
        "accuracy_easy_total": easy_t,
        "accuracy_medium_diff_1to2": round(med_c / max(med_t, 1), 4),
        "accuracy_medium_total": med_t,
        "accuracy_hard_diff_05to1": round(hard_c / max(hard_t, 1), 4),
        "accuracy_hard_total": hard_t,
        "spearman_rho_winrate_vs_avg": round(float(spearman_rho), 4),
        "spearman_pvalue": round(float(sp_p), 6),
        "kendall_tau_winrate_vs_avg": round(float(kendall_tau), 4),
        "kendall_pvalue": round(float(kt_p), 6),
        "num_rewrites_with_comparisons": len(per_rewrite_win_rates),
        "min_score_diff": min_score_diff,
        "position_swap": not skip_swap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "human_eval" / "eval.json"))
    parser.add_argument("--min_score_diff", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--skip_swap", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save checkpoint every N completed pairs")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("Error: API key required.")
        sys.exit(1)

    logger = setup_logging()

    model_short = args.model.replace("/", "_")
    output_path = args.output_path or str(
        PROJECT_ROOT / "data" / "pairwise" / f"api_baseline_{model_short}.json"
    )
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.json")
    pairs_detail_path = Path(output_path).with_name(
        Path(output_path).stem + "_pairs.json"
    )

    logger.info("=" * 70)
    logger.info("API Pairwise Cross-Source Evaluation (Multi-threaded)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Skip swap: {args.skip_swap}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Load eval data
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} samples")

    # Build pairs
    all_pairs = build_pairs(eval_data, args.min_score_diff)
    logger.info(f"Constructed {len(all_pairs)} pairs")
    if args.max_pairs > 0:
        all_pairs = all_pairs[:args.max_pairs]
        logger.info(f"Limited to {args.max_pairs} pairs")

    # Resume from checkpoint
    completed_results = {}  # pair_idx -> result
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        completed_results = {r.get("_pair_idx", r.get("index_a", -1)): r
                              for r in ckpt.get("results", [])}
        start_idx = ckpt.get("next_idx", 0)
        logger.info(f"Resuming from checkpoint: {start_idx}/{len(all_pairs)} pairs done")

    # Prepare work items
    work_items = []
    for pair_idx in range(start_idx, len(all_pairs)):
        i, j, a_should_win = all_pairs[pair_idx]
        sa = eval_data[i]["avg_score"]
        sb = eval_data[j]["avg_score"]
        work_items.append((
            pair_idx, i, j, a_should_win, sa, sb,
            eval_data[i]["input"], eval_data[i]["output"],
            eval_data[j]["input"], eval_data[j]["output"],
            args.model, api_key, args.max_tokens, args.temperature,
            args.skip_swap,
        ))

    if not work_items:
        logger.info("Nothing to do!")
        return

    # Thread-safe result collection
    lock = threading.Lock()
    save_counter = [0]  # mutable counter for save interval
    total_api_calls = [0]
    api_errors = [0]

    # Build ordered results list (for saving)
    all_results_list = []
    for r in completed_results.values():
        r["_pair_idx"] = r.get("_pair_idx", r.get("index_a", -1))
        all_results_list.append(r)

    def on_result(pair_idx, result):
        nonlocal all_results_list
        with lock:
            result["_pair_idx"] = pair_idx  # tag for ordering
            all_results_list.append(result)
            total_api_calls[0] += 1 if args.skip_swap else 2
            save_counter[0] += 1

            # Incremental save
            if save_counter[0] >= args.save_interval:
                save_counter[0] = 0
                _save_checkpoint()

    def _save_checkpoint():
        """Save current progress to checkpoint file."""
        ckpt_data = {
            "model": args.model,
            "next_idx": start_idx + len(all_results_list),
            "total_pairs": len(all_pairs),
            "total_api_calls": total_api_calls[0],
            "results": all_results_list,
        }
        tmp_path = checkpoint_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_data, f, ensure_ascii=False)
        tmp_path.replace(checkpoint_path)

    logger.info(f"Running {len(work_items)} pairs with {args.workers} threads...")
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_pair, item): item[0] for item in work_items}

            with tqdm(total=len(work_items), desc=model_short[:25],
                      initial=len(completed_results), leave=False) as pbar:
                for future in as_completed(futures):
                    try:
                        pair_idx, result = future.result()
                        on_result(pair_idx, result)
                    except Exception as e:
                        with lock:
                            api_errors[0] += 1
                        logger.warning(f"Error on pair {futures[future]}: {e}")
                    pbar.update(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted! Saving progress...")

    # Final save
    _save_checkpoint()

    # Build ordered pair_results (by pair_idx)
    all_results_sorted = sorted(all_results_list, key=lambda r: r["_pair_idx"])
    pair_results = [{k: v for k, v in r.items() if k != "_pair_idx"}
                     for r in all_results_sorted]

    # Compute metrics
    metrics = compute_metrics(pair_results, eval_data, args.min_score_diff, args.skip_swap)
    elapsed = time.time() - start_time

    # Print results
    logger.info("")
    logger.info("Results:")
    logger.info(f"  Pairs: {metrics['total_pairs']} (valid: {metrics['valid_pairs']}, "
                f"parse failures: {metrics['parse_failures']})")
    logger.info(f"  Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
    logger.info(f"  Easy (diff>=2): {metrics['accuracy_easy_diff_ge2']:.4f} "
                f"({metrics['accuracy_easy_total']})")
    logger.info(f"  Medium (1<=d<2): {metrics['accuracy_medium_diff_1to2']:.4f} "
                f"({metrics['accuracy_medium_total']})")
    logger.info(f"  Hard (0.5<=d<1): {metrics['accuracy_hard_diff_05to1']:.4f} "
                f"({metrics['accuracy_hard_total']})")
    logger.info(f"  Spearman: {metrics['spearman_rho_winrate_vs_avg']:.4f} "
                f"(p={metrics['spearman_pvalue']:.6f})")
    logger.info(f"  Kendall: {metrics['kendall_tau_winrate_vs_avg']:.4f}")
    logger.info(f"  API errors: {api_errors[0]}")
    logger.info(f"  Total API calls: {total_api_calls[0]}")
    logger.info(f"  Time: {elapsed:.1f}s")

    # Save final results
    output = {
        "model": args.model,
        "eval_type": "api_cross_source_pairwise",
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "total_api_calls": total_api_calls[0],
        "api_errors": api_errors[0],
    }
    results_path = Path(output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(pairs_detail_path, "w", encoding="utf-8") as f:
        json.dump(pair_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {results_path}")
    logger.info(f"Saved: {pairs_detail_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
