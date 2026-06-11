<h1 align="center">RewriteJudge</h1>
<h3 align="center">From Absolute Scores to Pairwise Preferences: Training LLM Judges for Chinese Text Rewriting Evaluation</h3>
<p align="center"><em>EMNLP 2026</em></p>
<p align="center">
  <a href="https://huggingface.co/datasets/heihei/llm-rewrite"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue" alt="HuggingFace"></a>
  <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  <img src="https://img.shields.io/badge/License-CC--BY--4.0-green" alt="License">
</p>

---

## Introduction

Evaluating Chinese text rewriting is inherently subjective: a good rewrite must preserve meaning while improving expression, and annotators often make relative rather than calibrated absolute judgments.
This raises a central question: *what supervision signal should be used to train LLM-based rewrite judges?*

We study two training paradigms for LLM-based rewrite judges: *absolute-score SFT*, which trains the model to predict a 0–5 score directly, and *pairwise preference training*, which trains the model to compare candidate rewrites and aggregates pairwise preferences into a global ranking.

<p align="center"><img src="paper/figures/ai_generated/framework_overview_v2.png" width="80%"><br><em>Overview of our approach: we collect human-rated Chinese rewrite pairs, construct preference supervision from holistic ratings, and train RewriteJudge via contrastive learning with win-rate aggregation.</em></p>

Using 730 human-rated Chinese source–rewrite pairs from three trained annotators (ICC(2,1) = 0.87), we show that pairwise preference training achieves the highest ranking alignment (ρ = 0.688), outperforming absolute-score SFT (ρ = 0.645) and substantially exceeding zero-shot and general-purpose LLM judges.

---

## Preference-Based Rewrite Judge

### Human Preference Normalization

We define text rewriting quality along four complementary dimensions, scored from 0 to 5:

1. **Meaning preservation** — Preserve the source meaning without factual errors
2. **Expression improvement** — Improve fluency, clarity, and readability
3. **Appropriate transformation** — Structural and phrasing changes, not mechanical substitution
4. **Naturalness and idiomaticity** — Natural, contextually appropriate Chinese expression

| Property | Value |
|----------|-------|
| **Total pairs** | 730 Chinese rewrite pairs |
| **Annotators** | 3 trained annotators, independent |
| **Scale** | 0–5 integer (holistic quality) |
| **Inter-annotator agreement** | Spearman ρ ≈ 0.88, ICC(2,1) = 0.87 |
| **Train / Eval split** | 600 / 130 (stratified by score) |
| **Score distribution** | 0(15.5%), 1(27.9%), 2(24.8%), 3(20.2%), 4(9.3%), 5(2.3%) |
| **Balanced training** | 1,008 samples (168 per class, oversampled) |

### Absolute Scoring and Pairwise Preference Training

**Absolute-score SFT** trains a model to predict a holistic quality score f: S × R → [0, 5]. The training objective minimizes the discrepancy between predicted and human-assigned scores.

**Pairwise preference training** trains the model to compare pairs of rewrites. Given two source–rewrite pairs with human ratings qᵢ > qⱼ, the model learns P(rᵢ ≻ rⱼ). The key difference is that absolute scoring requires global calibration across all quality levels, while pairwise training only requires local relative judgments.

<p align="center"><img src="paper/figures/ai_generated/pairwise_vs_absolute_v2.png" width="90%"><br><em>Absolute scoring requires global score calibration across all quality levels; pairwise training learns relative preference and aggregates comparisons into rankings via win rates.</em></p>

### Win-Rate Ranking

From the 600 training samples, we construct 2,652 balanced pairwise comparisons. At inference, all C(130,2) = 8,385 evaluation pairs are scored; each rewrite's win rate aggregates multiple comparisons into a robust quality estimate.

---

## Results and Analysis

### Main Results: Pairwise Preference vs. Absolute SFT

Cross-source pairwise preference training achieves ρ = 0.688 on Qwen3 8B, outperforming absolute-score SFT (ρ = 0.645) by 0.043. The same pattern holds on Qwen2.5 7B (ρ = 0.665 vs. ρ = 0.580, a gain of 0.085). Both trained variants substantially exceed zero-shot and general-purpose judges.

| Method | Model | Supervision | Spearman ρ |
|--------|:-----:|:-----------:|:----------:|
| **Pairwise (ours)** | **Qwen3 8B** | **Cross-source pref.** | **+0.688** |
| Pairwise (ours) | Qwen2.5 7B | Cross-source pref. | +0.665 |
| Absolute SFT (ours) | Qwen3 8B | Score regression | +0.645 |
| Absolute SFT (ours) | Qwen2.5 7B | Score regression | +0.580 |
| Zero-shot | Qwen2.5 72B | None | +0.406 |
| Zero-shot | DeepSeek V3 | None | +0.391 |
| Gen. purpose | Prometheus 2 | Multi-task | +0.145 |
| Zero-shot | Qwen3 8B | None | +0.106 |

### Why Pairwise Preference Helps

**Pairwise supervision is data-efficient.** Qwen3 8B reaches ρ = 0.611 with only 25% of the data (663 pairs) and ρ = 0.669 with 50%, demonstrating that pairwise supervision is data-efficient and maintains stable performance even with limited training data.

| Model | Training Data | Pairs | Spearman ρ |
|:-----:|:-------------:|:-----:|:----------:|
| Qwen2.5 7B | 25% | 663 | +0.434 |
| Qwen2.5 7B | 50% | 1,326 | +0.140 |
| **Qwen2.5 7B** | **100%** | **2,652** | **+0.665** |
| Qwen3 8B | 25% | 663 | +0.611 |
| Qwen3 8B | 50% | 1,326 | +0.669 |
| **Qwen3 8B** | **100%** | **2,652** | **+0.688** |

The 50% anomaly for Qwen2.5 7B (ρ = 0.140) reveals a **position bias** failure mode: the model learned to always predict "the second rewrite wins." Sufficient data (100%) prevents this shortcut.

**Absolute scoring requires more calibration data.** Performance improves from 50 to 1,008 samples, with a clear jump after 400. Even at the full 1,008 balanced samples, absolute scoring remains below pairwise preference training.

| Balanced Samples | Qwen2.5 7B | Qwen3 8B |
|:----------------:|:----------:|:--------:|
| 50 | −0.026 | +0.058 |
| 100 | +0.094 | +0.117 |
| 200 | +0.259 | +0.136 |
| 400 | +0.235 | +0.247 |
| **1,008 (full)** | **+0.580** | **+0.645** |

<p align="center"><img src="code/analysis/figures/learning_curve.png" width="65%"><br><em>Full learning curve including both LoRA variants and baseline methods (dashed lines). All baselines show flat or negative correlation regardless of data size.</em></p>

**Training robustness: LoRA rank ablation.** Pairwise preference learning is not highly sensitive to adapter capacity.

| Model | Rank *r* | α | Params | Spearman ρ |
|:-----:|:--------:|:-:|:------:|:----------:|
| Qwen2.5 7B | 8 | 16 | ~0.1% | +0.685 |
| Qwen2.5 7B | 16 | 32 | ~0.2% | +0.665 |
| **Qwen2.5 7B** | **32** | **64** | **~0.4%** | **+0.705** |
| Qwen3 8B | 8 | 16 | ~0.1% | +0.679 |
| Qwen3 8B | 16 | 32 | ~0.2% | +0.688 |
| **Qwen3 8B** | **32** | **64** | **~0.4%** | **+0.713** |

**Ranking consistency beyond pairwise accuracy.** Pairwise accuracy and ranking correlation are poorly aligned for zero-shot models: Qwen3-235B achieves 72.0% accuracy yet only ρ = 0.132, while our fine-tuned 8B model reaches ρ = 0.688 at 69.5% accuracy.

| Model | Accuracy (%) | Spearman ρ | Paradox? |
|-------|:----------:|:----------:|:--------:|
| Ours (8B LoRA) | 69.5 | +0.688 | No |
| Ours (7B LoRA) | 90.0 | +0.665 | No |
| Qwen3-235B | 72.0 | +0.132 | **Yes** |
| Qwen2.5-72B | 69.3 | +0.406 | Mild |
| DeepSeek V3 | 67.7 | +0.391 | No |
| Kimi K2 | 31.6 | −0.472 | No |

<p align="center"><img src="code/analysis/figures/accuracy_vs_rho.png" width="55%"><br><em>Pairwise accuracy vs. Spearman ρ: high accuracy but low ρ shows that accuracy can mislead for zero-shot evaluators.</em></p>

**Calibration collapse in absolute scoring.** Zero-shot models concentrate predictions in a narrow score range; Prometheus 2 collapses to Score 3; RewriteJudge produces a distribution closer to human annotations across all six score levels.

<p align="center"><img src="code/analysis/figures/score_distribution.png" width="75%"><br><em>Distribution of human annotations and evaluator predictions. RewriteJudge produces a distribution closer to human annotations.</em></p>

<table align="center">
<tr>
<td align="center"><img src="code/analysis/figures/confusion_matrix_lora_balanced_simple.png" width="90%"><br><em>RewriteJudge (Qwen3 8B): strong diagonal trend with very few extreme misclassifications.</em></td>
<td align="center"><img src="code/analysis/figures/confusion_matrix_prometheus2.png" width="90%"><br><em>Prometheus 2: predicts Score 3 for most samples, exhibiting discriminative collapse.</em></td>
</tr>
</table>

### Robustness and Diagnostic Checks

**Bias analysis.** A reliable evaluator should judge quality based on content, not spurious features. RewriteJudge shows minimal bias across all dimensions.

| Method | Output Len. *r* | Verbosity ρ | Position *r* | Bias Score |
|--------|:--------------:|:-----------:|:------------:|:----------:|
| **RewriteJudge (Qwen3 8B)** | **−0.033** | **0.006** | **0.120** | **0.057** |
| RewriteJudge (Qwen2.5 7B) | −0.058 | 0.106 | 0.187 | 0.116 |
| Prometheus 2 | 0.010 | 0.098 | 0.223 | 0.100 |
| Zero-shot Qwen3 8B | 0.325 | 0.382 | 0.437 | 0.328 |
| Char Overlap | 0.306 | 0.760 | 0.762 | 0.468 |

**Length bias.** Both RewriteJudge variants remain nearly flat across output length quartiles, while traditional metrics show increasing scores for longer outputs regardless of quality.

<p align="center"><img src="code/analysis/figures/length_quartile_trend.png" width="60%"><br><em>Mean evaluator score by output length quartile. RewriteJudge maintains consistent scores; traditional metrics favor longer outputs.</em></p>

**Verbosity bias.** RewriteJudge shows almost no verbosity bias, while the length heuristic exhibits extreme preference for verbose outputs (ρ = 0.741).

<p align="center"><img src="code/analysis/figures/verbosity_bias_bars.png" width="60%"><br><em>Verbosity bias comparison across evaluator methods.</em></p>

**Position and overall bias.** Both RewriteJudge variants have low overall bias scores, while zero-shot and traditional methods show larger bias across multiple dimensions.

<p align="center"><img src="code/analysis/figures/bias_radar.png" width="50%"><br><em>Bias heatmap across evaluator methods. Lower values indicate weaker correlation with spurious factors.</em></p>

**Source similarity is not a reliable proxy.** All seven source-similarity metrics are negatively correlated with human judgment (ρ = −0.28 to −0.60), with bootstrap 95% confidence intervals that do not cross zero.

<p align="center"><img src="code/analysis/figures/method_comparison_traditional.png" width="70%"><br><em>All seven source-similarity metrics are negatively aligned with human rewrite quality.</em></p>

Two systematic error patterns explain this failure:

| Case | BLEU | Human | Error mode |
|------|:----:|:-----:|:----------:|
| Minor lexical substitution | 0.55 | 0.7 | High overlap rewards copying |
| Substantial faithful rewriting | 0.00 | 4.0 | Low overlap penalizes useful transformation |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base models | Qwen3 8B, Qwen2.5 7B Instruct |
| Quantization | bf16 (no quantization) |
| LoRA rank (*r*) | 16 |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Target modules | q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj |
| Learning rate | 2×10⁻⁴ |
| LR scheduler | Cosine (warmup 3%) |
| Epochs | 3 |
| Batch size | 4 (grad accum: 4, effective: 16) |
| Precision | bf16 |
| Gradient checkpointing | Enabled |
| Training samples | 1,008 (absolute) / 2,652 pairs (pairwise) |
| GPU | 1× NVIDIA H20 (96GB) |
| Training time | ~16 minutes |

---

## Quick Start

### Setup

```bash
pip install "transformers>=4.45,<4.50" "peft>=0.13,<0.15" "trl>=0.12,<0.15"
```

### Download Data

```python
from datasets import load_dataset
dataset = load_dataset("heihei/llm-rewrite")
```

### Train RewriteJudge (Absolute Scoring)

```bash
python code/evaluator/train_lora.py \
  --data_path data/human_eval/train_score_only_detail_balanced.json \
  --output_dir code/evaluator/checkpoints/balanced_detailed \
  --base_model Qwen/Qwen3-8B
```

### Train Pairwise Evaluator

```bash
python code/evaluator/train_lora.py \
  --data_path data/pairwise/cross_source_train.json \
  --output_dir code/evaluator/checkpoints/pairwise_cross_source \
  --base_model Qwen/Qwen3-8B \
  --task pairwise
```

### Run All Baselines

```bash
bash code/scripts/run_all.sh
```

---

## Repo Structure

```
├── paper/                    # Paper (LaTeX + PDF)
│   ├── main.tex              # Main paper source
│   ├── main.pdf              # Compiled paper
│   ├── refs.bib              # Bibliography
│   └── figures/              # All paper figures (PDF + PNG)
│       └── ai_generated/     # Framework and paradigm overview diagrams
├── code/evaluator/           # LoRA training & evaluation
│   ├── train_lora.py         # Fine-tuning script
│   ├── eval_evaluator.py     # Absolute scoring evaluation
│   ├── eval_pairwise.py      # Pairwise evaluation
│   ├── eval_api_pairwise.py  # API baseline evaluation
│   ├── prompts.py            # Prompt templates
│   └── config.yaml           # Configuration
├── code/baselines/           # All baseline evaluators
│   ├── run_traditional.py    # BLEU, ROUGE, SBERT, etc.
│   ├── run_llm_evaluators.py # Zero-shot LLM eval
│   ├── run_prometheus2.py    # Prometheus 2 eval
│   ├── run_parascore.py      # ParaScore eval
│   ├── run_qwen32b_eval.py   # Qwen32B eval
│   └── correlation_utils.py  # Correlation computation utilities
├── code/analysis/            # Correlation, bias, error analysis
│   ├── correlation_analysis.py
│   ├── bias_analysis.py
│   ├── error_analysis.py
│   ├── generate_figures.py   # All figure generation
│   ├── learning_curves.py
│   ├── results/              # Analysis outputs + LaTeX tables
│   └── figures/              # Generated figures (PDF + PNG)
├── code/downstream/          # Downstream validation pipeline
│   ├── generate_data.py      # Generate rewrites (API)
│   ├── generate_data_local.py # Generate rewrites (local)
│   ├── score_rewrites.py     # Score with evaluator
│   ├── filter_data.py        # Filter by quality
│   ├── train_sft.py          # SFT training
│   └── eval_downstream.py    # Evaluation
└── code/scripts/             # Data prep and run scripts
    ├── create_balanced_data.py
    ├── create_pairwise_data.py
    ├── convert_data.py
    ├── consolidate_results.py
    ├── run_all.sh
    ├── run_all_pairwise_experiments.sh
    └── run_baselines.sh
```

---

## Dataset

All training and evaluation data is available on HuggingFace:

**[heihei/llm-rewrite](https://huggingface.co/datasets/heihei/llm-rewrite)**

| Split | Description | Size |
|-------|-------------|------|
| `human_eval/full.json` | 730 annotated pairs (3 annotators, 0–5) | 537K |
| `human_eval/train.json` | 600 training samples | 441K |
| `human_eval/eval.json` | 130 evaluation samples | 95K |
| `human_eval/train_score_only_detail_balanced.json` | 1,008 class-balanced training | 1.6M |
| `pairwise/cross_source_train.json` | 2,652 cross-source pairs | 4.5M |
| `baselines/all_results.json` | Consolidated predictions from 19 methods | 12K |

---

## Citation

```bibtex
@inproceedings{rewritejudge2026,
  title     = {From Absolute Scores to Pairwise Preferences: Training LLM Judges for Chinese Text Rewriting Evaluation},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

---

## License

- **Code**: MIT
- **Dataset**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Paper**: See `paper/`
