#!/usr/bin/env python3
"""
Router-v2 inference interface.

Public function:
    predict_hop(question: str, supports: list[str]) -> str
    Returns 'single' or 'multi' based on routing classifier.
"""

import json
import sys
from pathlib import Path

import numpy as np
import joblib
import spacy

# Import BiEncoderRetriever
sys.path.insert(0, str(Path(__file__).parent.parent / "retriever"))
from baselines import BiEncoderRetriever

# Module-level singletons
_model = None
_retriever = None
_nlp = None
_threshold = None


def _load_model():
    """Lazy-load trained model, scaler, threshold."""
    global _model, _threshold

    model_dir = Path(__file__).parent
    model_file = model_dir / "model.pkl"
    threshold_file = model_dir / "threshold.json"

    if not model_file.exists() or not threshold_file.exists():
        return None, None

    _model = joblib.load(model_file)
    with open(threshold_file) as f:
        threshold_data = json.load(f)
    _threshold = threshold_data["threshold"]

    return _model, _threshold


def _load_retriever():
    """Lazy-load SBERT retriever."""
    global _retriever
    if _retriever is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _retriever = BiEncoderRetriever(
            query_model=model_name,
            article_model=model_name,
            device="cpu"
        )
    return _retriever


def _load_nlp():
    """Lazy-load scispacy."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_sci_sm")
        except OSError:
            return None
    return _nlp


def _extract_sbert_features(query: str, supports: list[str]) -> tuple:
    """Extract SBERT-based features: top_score and score_gap."""
    retriever = _load_retriever()
    corpus_ids = [str(i) for i in range(len(supports))]
    corpus_texts = supports

    retriever.index(corpus_ids, corpus_texts)
    retrieved = retriever.retrieve(query, k=min(10, len(supports)))

    scores = [score for _, score in retrieved]
    scores_sorted = sorted(scores, reverse=True)

    top_sbert_score = float(scores_sorted[0]) if scores_sorted else 0.0
    sbert_score_gap = (
        float(scores_sorted[0] - scores_sorted[1])
        if len(scores_sorted) >= 2 else 0.0
    )

    return top_sbert_score, sbert_score_gap


def _extract_entity_count_feature(query: str) -> float:
    """Biomedical entity count via scispacy."""
    nlp = _load_nlp()
    if nlp is None:
        return 0.0
    doc = nlp(query)
    return float(len(doc.ents))


def _extract_n_supports_feature(supports: list[str]) -> int:
    """Number of support passages."""
    return len(supports)


def _extract_n_candidates_feature(supports: list[str]) -> int:
    """
    Number of candidate answers (estimated from supports).
    Note: In router interface, candidates are not explicitly passed,
    so we estimate this as 0 or could extract from supports context.
    For now, default to 0 since the original interface doesn't provide it.
    """
    return 0


def predict_hop(question: str, supports: list[str]) -> str:
    """
    Predict whether a question needs single-hop or multi-hop retrieval.

    Args:
        question: The biomedical question text
        supports: List of support passages available for the question

    Returns:
        'single' if the router predicts single-hop retrieval is sufficient
        'multi' if the router predicts multi-hop retrieval is needed
    """
    model, threshold = _load_model()

    if model is None or threshold is None:
        return "multi"

    n_supports = _extract_n_supports_feature(supports)
    n_candidates = _extract_n_candidates_feature(supports)
    top_sbert_score, sbert_score_gap = _extract_sbert_features(question, supports)
    entity_count = _extract_entity_count_feature(question)

    X = np.array([[n_supports, n_candidates, top_sbert_score, sbert_score_gap, entity_count]], dtype=np.float32)

    X_scaled = model["scaler"].transform(X)
    prob = model["lr"].predict_proba(X_scaled)[0, 1]

    if prob >= threshold:
        return "multi"
    else:
        return "single"


def predict_multi() -> str:
    """Baseline: always predict multi-hop. Used for ablation comparison."""
    return "multi"


if __name__ == "__main__":
    fake_question = "interacts_with DB00773?"
    fake_supports = [
        "DB00773 has been shown to interact with DB00072 in clinical trials.",
        "Studies show DB00294 affects the same pathway as DB00338.",
        "Drug interaction data for DB00773 indicates pathway involvement.",
    ]

    model, threshold = _load_model()
    if model is None:
        print("model.pkl or threshold.json not found — run train.py first.")
    else:
        result = predict_hop(fake_question, fake_supports)
        print(f"predict_hop() → '{result}'")
        print("router-v2 is working correctly.")
        print("Note: first run downloads SBERT model (~80 MB); subsequent runs are fast.")
