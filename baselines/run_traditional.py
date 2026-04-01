"""
Traditional baseline metrics for Chinese text rewriting evaluation.

Computes:
- BLEU (sacrebleu)
- ROUGE-L (rouge_score)
- Jaccard similarity (character-level and word-level for Chinese)
- TF-IDF cosine similarity
- Word2Vec cosine similarity (via sentence-transformers)
- Sentence-BERT cosine similarity (paraphrase-multilingual-MiniLM-L12-v2)

Usage:
    python baselines/run_traditional.py
"""

import json
import re
import sys
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore")

# Add parent directory to path
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
# Chinese text tokenization helpers
# ============================================================

def chinese_char_tokenize(text: str) -> List[str]:
    """Tokenize Chinese text into individual characters (CJK unified ideographs only).

    Punctuation and non-CJK characters are filtered out for cleaner comparison.
    """
    chars = []
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs: 4E00-9FFF
        # CJK Extension A: 3400-4DBF
        # CJK Extension B: 20000-2A6DF (supplementary, skip for efficiency)
        # CJK Compatibility Ideographs: F900-FAFF
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
            chars.append(ch)
    return chars


def chinese_word_tokenize(text: str) -> List[str]:
    """Simple word segmentation for Chinese using character bigrams and unigrams.

    For a more sophisticated approach, jieba could be used, but this avoids
    the dependency and works reasonably well for similarity metrics.
    Falls back to jieba if available.
    """
    try:
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return list(jieba.cut(text))
    except ImportError:
        pass

    # Fallback: use character n-grams (uni + bi) as pseudo-word tokens
    chars = chinese_char_tokenize(text)
    tokens = list(chars)  # unigrams
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])  # bigrams
    return tokens


def simple_tokenize(text: str) -> List[str]:
    """Basic whitespace and punctuation tokenization for non-Chinese text."""
    # Split on whitespace and strip punctuation
    tokens = re.findall(r'\w+', text.lower())
    return tokens


# ============================================================
# Jaccard Similarity
# ============================================================

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_jaccard_char(input_text: str, output_text: str) -> float:
    """Character-level Jaccard similarity for Chinese text."""
    chars_input = set(chinese_char_tokenize(input_text))
    chars_output = set(chinese_char_tokenize(output_text))
    return jaccard_similarity(chars_input, chars_output)


def compute_jaccard_word(input_text: str, output_text: str) -> float:
    """Word-level Jaccard similarity for Chinese text."""
    words_input = set(chinese_word_tokenize(input_text))
    words_output = set(chinese_word_tokenize(output_text))
    return jaccard_similarity(words_input, words_output)


# ============================================================
# BLEU Score
# ============================================================

def compute_bleu(input_text: str, output_text: str) -> float:
    """Compute BLEU score using sacrebleu.

    Treats the input as the reference and output as the hypothesis.
    """
    try:
        import sacrebleu
        # sacrebleu expects list of refs (list of lists) and list of hyps
        bleu = sacrebleu.sentence_bleu(output_text, [input_text])
        return bleu.score / 100.0  # Normalize to [0, 1]
    except Exception as e:
        warnings.warn(f"BLEU computation failed: {e}")
        return 0.0


# ============================================================
# ROUGE-L Score
# ============================================================

def compute_rouge_l(input_text: str, output_text: str) -> float:
    """Compute ROUGE-L F1 score using rouge_score.

    Treats the input as the reference and output as the hypothesis.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(input_text, output_text)
        return scores["rougeL"].fmeasure
    except Exception as e:
        warnings.warn(f"ROUGE-L computation failed: {e}")
        return 0.0


# ============================================================
# TF-IDF Cosine Similarity
# ============================================================

class TFIDFSimilarity:
    """Compute TF-IDF cosine similarity between Chinese texts."""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return chinese_word_tokenize(text)

    def fit(self, texts: List[str]):
        """Build vocabulary and compute IDF from a corpus."""
        doc_freq = Counter()
        n_docs = len(texts)

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] += 1

        # Build vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(doc_freq.keys()))}

        # Compute IDF with smoothing
        self.idf = {}
        for token, df in doc_freq.items():
            self.idf[token] = np.log(1 + n_docs / (1 + df)) + 1  # smooth IDF

        self._fitted = True

    def _tfidf_vector(self, text: str) -> np.ndarray:
        """Compute TF-IDF vector for a single text."""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocabulary))

        tf = Counter(tokens)
        total = len(tokens)
        vector = np.zeros(len(self.vocabulary))

        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_val = count / total
                vector[idx] = tf_val * self.idf.get(token, 1.0)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self._tfidf_vector(text_a)
        vec_b = self._tfidf_vector(text_b)
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# ============================================================
# Word2Vec / Sentence Embedding Similarity
# ============================================================

class EmbeddingSimilarity:
    """Compute embedding-based cosine similarity using sentence-transformers."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._loaded = False

    def load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        if not self._loaded:
            self.load_model()

        embeddings = self.model.encode([text_a, text_b])
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(cos_sim)

    def similarity_batch(self, texts_a: List[str], texts_b: List[str]) -> List[float]:
        """Compute cosine similarities for a batch of text pairs."""
        if not self._loaded:
            self.load_model()

        all_texts = texts_a + texts_b
        embeddings = self.model.encode(all_texts)
        n = len(texts_a)
        similarities = []
        for i in range(n):
            cos_sim = np.dot(embeddings[i], embeddings[n + i]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[n + i])
            )
            similarities.append(float(cos_sim))
        return similarities


def compute_word2vec_similarity(
    texts_a: List[str],
    texts_b: List[str],
    model_name: str = "shibing624/text2vec-base-chinese",
) -> List[float]:
    """Compute Word2Vec-style cosine similarity using a Chinese text encoder.

    Uses shibing624/text2vec-base-chinese which is a Chinese-oriented
    sentence embedding model trained on word2vec-style tasks.
    """
    embedder = EmbeddingSimilarity(model_name)
    return embedder.similarity_batch(texts_a, texts_b)


def compute_sentence_bert_similarity(
    texts_a: List[str],
    texts_b: List[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> List[float]:
    """Compute Sentence-BERT cosine similarity.

    Uses paraphrase-multilingual-MiniLM-L12-v2 for multilingual support.
    """
    embedder = EmbeddingSimilarity(model_name)
    return embedder.similarity_batch(texts_a, texts_b)


# ============================================================
# Main evaluation runner
# ============================================================

def run_all_traditional_metrics(
    data: List[Dict],
    skip_embedding: bool = False,
    embedding_model_sbert: str = "paraphrase-multilingual-MiniLM-L12-v2",
    embedding_model_w2v: str = "shibing624/text2vec-base-chinese",
) -> Dict:
    """Run all traditional metrics on the evaluation dataset.

    Args:
        data: Evaluation dataset (list of dicts).
        skip_embedding: If True, skip embedding-based metrics (faster).
        embedding_model_sbert: Sentence-BERT model name.
        embedding_model_w2v: Word2Vec-style model name.

    Returns:
        Dict with all metric scores per sample and aggregate correlations.
    """
    n = len(data)
    print(f"Running traditional metrics on {n} samples ...")

    inputs = [d["input"] for d in data]
    outputs = [d["output"] for d in data]
    gt_scores = [d["consensus_score"] for d in data]

    # --- Lexical overlap metrics ---
    print("  Computing Jaccard (char) ...")
    jaccard_char_scores = [compute_jaccard_char(inp, out) for inp, out in zip(inputs, outputs)]

    print("  Computing Jaccard (word) ...")
    jaccard_word_scores = [compute_jaccard_word(inp, out) for inp, out in zip(inputs, outputs)]

    # --- BLEU ---
    print("  Computing BLEU ...")
    bleu_scores = [compute_bleu(inp, out) for inp, out in zip(inputs, outputs)]

    # --- ROUGE-L ---
    print("  Computing ROUGE-L ...")
    rouge_l_scores = [compute_rouge_l(inp, out) for inp, out in zip(inputs, outputs)]

    # --- TF-IDF cosine similarity ---
    print("  Computing TF-IDF ...")
    # Fit on all texts (both inputs and outputs) for a shared vocabulary
    tfidf = TFIDFSimilarity()
    tfidf.fit(inputs + outputs)
    tfidf_scores = [tfidf.similarity(inp, out) for inp, out in zip(inputs, outputs)]

    # --- Embedding-based metrics ---
    sbert_scores = None
    w2v_scores = None

    if not skip_embedding:
        print(f"  Computing Sentence-BERT ({embedding_model_sbert}) ...")
        try:
            sbert_scores = compute_sentence_bert_similarity(
                inputs, outputs, model_name=embedding_model_sbert
            )
        except Exception as e:
            print(f"    WARNING: Sentence-BERT failed: {e}")

        print(f"  Computing Word2Vec ({embedding_model_w2v}) ...")
        try:
            w2v_scores = compute_word2vec_similarity(
                inputs, outputs, model_name=embedding_model_w2v
            )
        except Exception as e:
            print(f"    WARNING: Word2Vec failed: {e}")
    else:
        print("  Skipping embedding metrics (--skip-embedding)")

    # --- Assemble per-sample results ---
    sample_results = []
    for i in range(n):
        result = {
            "idx": i,
            "input": inputs[i][:100] + "..." if len(inputs[i]) > 100 else inputs[i],
            "output": outputs[i][:100] + "..." if len(outputs[i]) > 100 else outputs[i],
            "consensus_score": gt_scores[i],
            "jaccard_char": round(jaccard_char_scores[i], 4),
            "jaccard_word": round(jaccard_word_scores[i], 4),
            "bleu": round(bleu_scores[i], 4),
            "rouge_l": round(rouge_l_scores[i], 4),
            "tfidf_cosine": round(tfidf_scores[i], 4),
        }
        if sbert_scores is not None:
            result["sbert_cosine"] = round(sbert_scores[i], 4)
        if w2v_scores is not None:
            result["w2v_cosine"] = round(w2v_scores[i], 4)
        sample_results.append(result)

    # --- Compute correlations ---
    all_correlations = {}

    metrics = {
        "jaccard_char": jaccard_char_scores,
        "jaccard_word": jaccard_word_scores,
        "bleu": bleu_scores,
        "rouge_l": rouge_l_scores,
        "tfidf_cosine": tfidf_scores,
    }
    if sbert_scores is not None:
        metrics["sbert_cosine"] = sbert_scores
    if w2v_scores is not None:
        metrics["w2v_cosine"] = w2v_scores

    print("\n  Correlation Results:")
    print("  " + "=" * 100)
    for metric_name, scores in metrics.items():
        corr = compute_correlations(scores, gt_scores, metric_name)
        all_correlations[metric_name] = corr
        print(print_correlation_table(corr))
    print("  " + "=" * 100)

    # --- Per-score-level analysis ---
    all_level_analysis = {}
    for metric_name, scores in metrics.items():
        level = per_score_level_analysis(scores, gt_scores, metric_name)
        all_level_analysis[metric_name] = level

    # --- Save results ---
    results = {
        "n_samples": n,
        "metrics_run": list(metrics.keys()),
        "correlations": {k: v for k, v in all_correlations.items()},
        "level_analysis": all_level_analysis,
        "sample_results": sample_results,
    }

    out_path = save_results(results, "traditional_metrics.json")
    print(f"\n  Results saved to: {out_path}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run traditional baseline metrics")
    parser.add_argument(
        "--eval-path", type=str, default=str(EVAL_PATH),
        help="Path to eval.json"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="Skip embedding-based metrics (faster, fewer dependencies)"
    )
    parser.add_argument(
        "--sbert-model", type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-BERT model name"
    )
    parser.add_argument(
        "--w2v-model", type=str,
        default="shibing624/text2vec-base-chinese",
        help="Word2Vec-style model name"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Traditional Baseline Metrics Evaluation")
    print("=" * 60)

    data = load_eval_data(args.eval_path)
    results = run_all_traditional_metrics(
        data,
        skip_embedding=args.skip_embedding,
        embedding_model_sbert=args.sbert_model,
        embedding_model_w2v=args.w2v_model,
    )

    # Print best method
    print("\nBest traditional metric by Spearman rho:")
    best = max(results["correlations"].items(), key=lambda x: x[1].get("spearman_rho", 0))
    print(f"  {best[0]}: rho = {best[1]['spearman_rho']:.4f}")


if __name__ == "__main__":
    main()
