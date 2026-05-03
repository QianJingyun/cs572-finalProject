#!/usr/bin/env python3
"""
Phase 2: Extract SBERT-based features for training and validation.

Features extracted:
1. n_supports — number of support passages
2. n_candidates — number of candidate answers (estimated as 0 for v2)
3. top_sbert_score — highest SBERT retrieval score
4. sbert_score_gap — gap between top-1 and top-2 scores
5. entity_count — biomedical entities via scispacy

Output:
- features_sbert_train.npz (shape: 1620 x 5)
- features_sbert_validation.npz (shape: 342 x 5)
"""

import numpy as np
import sys
from pathlib import Path
from datasets import load_dataset
import spacy

sys.path.insert(0, str(Path(__file__).parent.parent / "retriever"))
from baselines import BiEncoderRetriever

def extract_features_for_split(data, split_name):
    """Extract features for train or validation split."""
    print(f"\nProcessing {split_name} split ({len(data)} examples)...")

    retriever = BiEncoderRetriever(
        query_model="sentence-transformers/all-MiniLM-L6-v2",
        article_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        print("WARNING: scispacy not installed, entity_count will be 0")
        nlp = None

    features_list = []
    ids_list = []

    for idx, example in enumerate(data):
        example_id = example["id"]
        question = example["query"]
        supports = example["supports"]

        # Feature 1: n_supports
        n_supports = len(supports)

        # Feature 2: n_candidates (v2 doesn't have this, use 0)
        n_candidates = 0

        # Features 3-4: SBERT scores
        corpus_ids = [str(i) for i in range(len(supports))]
        corpus_texts = supports
        retriever.index(corpus_ids, corpus_texts)
        retrieved = retriever.retrieve(question, k=min(10, len(supports)))

        scores = [score for _, score in retrieved]
        scores_sorted = sorted(scores, reverse=True)

        top_sbert_score = float(scores_sorted[0]) if scores_sorted else 0.0
        sbert_score_gap = (
            float(scores_sorted[0] - scores_sorted[1])
            if len(scores_sorted) >= 2 else 0.0
        )

        # Feature 5: entity_count
        if nlp is not None:
            doc = nlp(question)
            entity_count = float(len(doc.ents))
        else:
            entity_count = 0.0

        features_list.append([n_supports, n_candidates, top_sbert_score, sbert_score_gap, entity_count])
        ids_list.append(example_id)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(data)}] extracted features")

    X = np.array(features_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=object)

    output_file = Path(__file__).parent / f"features_sbert_{split_name}.npz"
    np.savez(output_file, X=X, ids=ids)
    print(f"✓ Saved {output_file} (shape: {X.shape})")

    return X, ids

def main():
    print("Loading MedHop dataset...")
    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")

    train_X, train_ids = extract_features_for_split(dataset["train"], "train")
    val_X, val_ids = extract_features_for_split(dataset["validation"], "validation")

    print(f"\n✓ Feature extraction complete")
    print(f"  Training: {train_X.shape}")
    print(f"  Validation: {val_X.shape}")

if __name__ == "__main__":
    main()
