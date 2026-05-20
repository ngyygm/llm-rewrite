"""
Baseline evaluation package for EMNLP 2026 Chinese rewriting evaluation.

Modules:
- correlation_utils: Shared correlation and agreement analysis utilities
- run_traditional: Traditional metrics (BLEU, ROUGE-L, Jaccard, TF-IDF, SBERT, W2V)
- run_llm_evaluators: LLM-based evaluators (G-Eval, zero-shot Qwen2.5-7B)
- run_fine_tuned_evaluators: Fine-tuned evaluators (Prometheus 2, M-Prometheus)
- run_parascore: ParaScore baseline
- run_all_baselines: Master runner for all baselines
"""
