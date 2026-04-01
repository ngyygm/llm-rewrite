"""
ParaScore baseline for Chinese text rewriting evaluation.

Implements paraphrase scoring using the ParaScore approach:
- Character-level paraphrase similarity scoring
- Word-level paraphrase similarity scoring

ParaScore computes a composite score based on semantic similarity
and lexical diversity, measuring how well a text paraphrases another.

Can use the parascore library if available, or computes from scratch
using sentence embeddings + lexical metrics.

Usage:
    python baselines/run_parascore.py
    python baselines/run_parascore.py --model paraphrase-multilingual-MiniLM-L12-v2
"""

import json
import re
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from baselines.correlation_utils import (
    load_eval_data,
    compute_correlations,
    per_score_level_analysis,
    print_correlation_table,
    print_level_analysis,
    save_results,
)

EVAL_PATH = BASE_DIR / "data" / "human_eval" / "eval.json"
RESULTS_DIR = BASE_DIR / "data" / "baselines"


# ============================================================
# Chinese tokenization
# ============================================================

def chinese_char_tokenize(text: str) -> List[str]:
    """Tokenize Chinese text into individual CJK characters."""
    chars = []
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
            chars.append(ch)
    return chars


def chinese_word_tokenize(text: str) -> List[str]:
    """Word segmentation for Chinese. Uses jieba if available."""
    try:
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return list(jieba.cut(text))
    except ImportError:
        # Fallback: character unigrams + bigrams
        chars = chinese_char_tokenize(text)
        tokens = list(chars)
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        return tokens


# ============================================================
# ParaScore implementation (from scratch)
# ============================================================

class ParaScoreEvaluator:
    """Paraphrase scoring from scratch using embeddings and lexical metrics.

    The ParaScore approach combines:
    1. Semantic similarity (embedding cosine similarity)
    2. Lexical diversity (1 - n-gram overlap, measuring how different the text is)
    3. Length ratio penalty (penalize extreme length changes)

    Score = alpha * semantic_sim + (1 - alpha) * lexical_diversity * length_factor

    A good paraphrase has high semantic similarity AND high lexical diversity.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        alpha: float = 0.5,
    ):
        """
        Args:
            model_name: Sentence-transformer model for semantic similarity.
            alpha: Weight for semantic similarity (1-alpha for diversity).
        """
        self.model_name = model_name
        self.alpha = alpha
        self.model = None
        self._loaded = False

    def load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer
        print(f"  Loading model: {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)
        self._loaded = True

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic cosine similarity using sentence embeddings."""
        embeddings = self.model.encode([text_a, text_b])
        cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        ))
        return max(0.0, min(1.0, cos_sim))

    def _ngram_overlap(self, text_a: str, text_b: str, n: int = 2) -> float:
        """Compute n-gram overlap ratio between two texts."""
        def get_ngrams(text, n):
            chars = chinese_char_tokenize(text)
            return set(tuple(chars[i:i + n]) for i in range(len(chars) - n + 1))

        ngrams_a = get_ngrams(text_a, n)
        ngrams_b = get_ngrams(text_b, n)

        if not ngrams_a or not ngrams_b:
            return 0.0

        overlap = len(ngrams_a & ngrams_b)
        smaller = min(len(ngrams_a), len(ngrams_b))
        return overlap / smaller if smaller > 0 else 0.0

    def _lexical_diversity(self, text_a: str, text_b: str) -> float:
        """Compute lexical diversity: how different the output is from input.

        Uses a combination of character and word level metrics.
        """
        # Character-level diversity
        char_bigram_overlap = self._ngram_overlap(text_a, text_b, n=2)
        char_unigram_overlap = self._ngram_overlap(text_a, text_b, n=1)

        # Word-level diversity
        words_a = set(chinese_word_tokenize(text_a))
        words_b = set(chinese_word_tokenize(text_b))
        if words_a and words_b:
            word_overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
        else:
            word_overlap = 0.0

        # Combined diversity: 1 - average overlap
        avg_overlap = (char_bigram_overlap + char_unigram_overlap + word_overlap) / 3.0
        diversity = 1.0 - avg_overlap
        return max(0.0, min(1.0, diversity))

    def _length_factor(self, text_a: str, text_b: str) -> float:
        """Compute length ratio factor.

        Penalizes outputs that are too short or too long relative to input.
        Returns a value between 0 and 1, with 1.0 for similar lengths.
        """
        len_a = len(chinese_char_tokenize(text_a))
        len_b = len(chinese_char_tokenize(text_b))

        if len_a == 0:
            return 0.0

        ratio = len_b / len_a

        # Ideal ratio is close to 1.0
        # Penalty increases as ratio deviates from 1.0
        if ratio >= 0.5 and ratio <= 1.5:
            factor = 1.0 - 0.2 * abs(ratio - 1.0)
        elif ratio >= 0.3 and ratio <= 2.0:
            factor = 0.8 - 0.4 * (abs(ratio - 1.0) - 0.5)
        else:
            factor = max(0.1, 0.4 - 0.3 * (abs(ratio - 1.0) - 1.0))

        return max(0.0, min(1.0, factor))

    def score(
        self,
        input_text: str,
        output_text: str,
        mode: str = "combined",
    ) -> float:
        """Compute ParaScore for a single pair.

        Args:
            input_text: Original text.
            output_text: Rewritten text.
            mode: "combined" (full score), "semantic", "lexical", or "char".

        Returns:
            ParaScore value.
        """
        if not self._loaded:
            self.load_model()

        if mode == "semantic":
            return self._semantic_similarity(input_text, output_text)

        if mode == "lexical":
            return self._lexical_diversity(input_text, output_text)

        if mode == "char":
            # Character-level only: use character n-gram based approach
            sem_sim = self._semantic_similarity(input_text, output_text)
            char_overlap = self._ngram_overlap(input_text, output_text, n=1)
            diversity = 1.0 - char_overlap
            len_factor = self._length_factor(input_text, output_text)
            return float(np.clip(
                self.alpha * sem_sim + (1 - self.alpha) * diversity * len_factor,
                0.0, 1.0
            ))

        # Default: combined mode (word-level)
        sem_sim = self._semantic_similarity(input_text, output_text)
        lex_div = self._lexical_diversity(input_text, output_text)
        len_factor = self._length_factor(input_text, output_text)

        combined = self.alpha * sem_sim + (1 - self.alpha) * lex_div * len_factor
        return float(np.clip(combined, 0.0, 1.0))

    def score_batch(
        self,
        texts_a: List[str],
        texts_b: List[str],
        mode: str = "combined",
    ) -> List[float]:
        """Compute scores for a batch of text pairs.

        Args:
            texts_a: List of original texts.
            texts_b: List of rewritten texts.
            mode: Scoring mode.

        Returns:
            List of ParaScore values.
        """
        if not self._loaded:
            self.load_model()

        # Batch encode for semantic similarity (faster)
        all_texts = texts_a + texts_b
        embeddings = self.model.encode(all_texts)
        n = len(texts_a)
        sem_sims = []
        for i in range(n):
            cos_sim = float(np.dot(embeddings[i], embeddings[n + i]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[n + i])
            ))
            sem_sims.append(max(0.0, min(1.0, cos_sim)))

        scores = []
        for i in range(n):
            if mode == "semantic":
                scores.append(sem_sims[i])
            elif mode == "lexical":
                scores.append(self._lexical_diversity(texts_a[i], texts_b[i]))
            elif mode == "char":
                char_overlap = self._ngram_overlap(texts_a[i], texts_b[i], n=1)
                diversity = 1.0 - char_overlap
                len_factor = self._length_factor(texts_a[i], texts_b[i])
                scores.append(float(np.clip(
                    self.alpha * sem_sims[i] + (1 - self.alpha) * diversity * len_factor,
                    0.0, 1.0
                )))
            else:
                lex_div = self._lexical_diversity(texts_a[i], texts_b[i])
                len_factor = self._length_factor(texts_a[i], texts_b[i])
                combined = self.alpha * sem_sims[i] + (1 - self.alpha) * lex_div * len_factor
                scores.append(float(np.clip(combined, 0.0, 1.0)))

        return scores


# ============================================================
# Try using the parascore library if available
# ============================================================

def run_parascore_library(
    data: List[Dict],
    model_name: str = "prithivida/parrot_paraphraser",
) -> Optional[List[float]]:
    """Try to use the parascore library if installed.

    Args:
        data: Evaluation dataset.
        model_name: Model to use with parascore.

    Returns:
        List of scores, or None if parascore is not available.
    """
    try:
        from parascore import ParaScore
        print(f"  Using parascore library with model: {model_name}")

        scorer = ParaScore(model=model_name, device="cuda")
        inputs = [d["input"] for d in data]
        outputs = [d["output"] for d in data]

        # ParaScore returns dict with various metrics
        results = scorer.score(inputs, outputs)
        return results
    except ImportError:
        print("  parascore library not available, using from-scratch implementation")
        return None
    except Exception as e:
        print(f"  parascore library failed: {e}, using from-scratch implementation")
        return None


# ============================================================
# Main runner
# ============================================================

def run_parascore_evaluation(
    data: List[Dict],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    alpha: float = 0.5,
) -> Dict:
    """Run ParaScore evaluation on the dataset.

    Args:
        data: Evaluation dataset.
        model_name: Sentence-transformer model name.
        alpha: Weight for semantic similarity component.

    Returns:
        Results dict with correlations and per-sample data.
    """
    n = len(data)
    print(f"\nRunning ParaScore evaluation on {n} samples ...")
    print(f"  Model: {model_name}")
    print(f"  Alpha (semantic weight): {alpha}")

    inputs = [d["input"] for d in data]
    outputs = [d["output"] for d in data]
    gt_scores = [d["consensus_score"] for d in data]

    # Try parascore library first
    lib_scores = run_parascore_library(data)

    # Initialize from-scratch evaluator
    evaluator = ParaScoreEvaluator(model_name=model_name, alpha=alpha)

    # Compute scores in different modes
    modes = ["combined", "char", "semantic", "lexical"]
    all_mode_scores = {}

    for mode in modes:
        print(f"  Computing ParaScore ({mode}) ...")
        try:
            scores = evaluator.score_batch(inputs, outputs, mode=mode)
            all_mode_scores[mode] = scores
        except Exception as e:
            print(f"    WARNING: {mode} failed: {e}")
            all_mode_scores[mode] = [0.0] * n

    # If library scores are available, include them
    if lib_scores is not None:
        all_mode_scores["parascore_lib"] = lib_scores

    # --- Compute correlations ---
    all_correlations = {}
    print("\n  Correlation Results:")
    print("  " + "=" * 100)

    for mode, scores in all_mode_scores.items():
        corr = compute_correlations(scores, gt_scores, f"parascore_{mode}")
        all_correlations[mode] = corr
        print(print_correlation_table(corr))

    print("  " + "=" * 100)

    # --- Per-score-level analysis ---
    all_level_analysis = {}
    for mode, scores in all_mode_scores.items():
        level = per_score_level_analysis(scores, gt_scores, f"parascore_{mode}")
        all_level_analysis[mode] = level

    # --- Save sample results ---
    sample_results = []
    for i in range(n):
        result = {
            "idx": i,
            "consensus_score": gt_scores[i],
        }
        for mode, scores in all_mode_scores.items():
            result[f"parascore_{mode}"] = round(scores[i], 4)
        sample_results.append(result)

    results = {
        "n_samples": n,
        "model_name": model_name,
        "alpha": alpha,
        "modes": list(all_mode_scores.keys()),
        "correlations": all_correlations,
        "level_analysis": all_level_analysis,
        "sample_results": sample_results,
    }

    out_path = save_results(results, "parascore_metrics.json")
    print(f"\n  Results saved to: {out_path}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ParaScore baseline evaluation"
    )
    parser.add_argument(
        "--eval-path", type=str, default=str(EVAL_PATH),
        help="Path to eval.json"
    )
    parser.add_argument(
        "--model", type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformer model for semantic similarity"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for semantic similarity (vs lexical diversity)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ParaScore Baseline Evaluation")
    print("=" * 60)

    data = load_eval_data(args.eval_path)
    results = run_parascore_evaluation(
        data,
        model_name=args.model,
        alpha=args.alpha,
    )

    # Print best mode
    print("\nBest ParaScore mode by Spearman rho:")
    best_mode = max(
        results["correlations"].items(),
        key=lambda x: x[1].get("spearman_rho", 0)
    )
    print(f"  {best_mode[0]}: rho = {best_mode[1]['spearman_rho']:.4f}")


if __name__ == "__main__":
    main()
