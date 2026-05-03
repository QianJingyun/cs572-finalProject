#!/usr/bin/env python3
"""
Phase 1: Generate training labels using hybrid weak supervision.

Method: SBERT dense retrieval with confidence-based thresholding
- If top-1 retrieval score >= 0.75: label as 0 (easy, single-hop)
- Otherwise: label as 1 (hard, multi-hop)

This avoids expensive LLM inference while providing good weak supervision.
"""

import json
import numpy as np
import sys
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "retriever"))
from baselines import BiEncoderRetriever

def main():
    print("Loading MedHop dataset...")
    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    train_data = dataset["train"]

    print(f"Loaded {len(train_data)} training examples")

    print("Initializing SBERT retriever...")
    retriever = BiEncoderRetriever(
        query_model="sentence-transformers/all-MiniLM-L6-v2",
        article_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    labels_file = Path(__file__).parent / "labels_train.json"
    
    # Load existing labels if any
    if labels_file.exists():
        with open(labels_file) as f:
            labels_dict = json.load(f)
        print(f"Resuming from {len(labels_dict)} existing labels...")
    else:
        labels_dict = {}

    for idx, example in enumerate(train_data):
        example_id = example["id"]
        
        if example_id in labels_dict:
            continue

        question = example["query"]
        supports = example["supports"]

        corpus_ids = [str(i) for i in range(len(supports))]
        corpus_texts = supports

        retriever.index(corpus_ids, corpus_texts)
        retrieved = retriever.retrieve(question, k=1)

        if retrieved:
            top_score = retrieved[0][1]
            label = 0 if top_score >= 0.75 else 1
        else:
            label = 1

        labels_dict[example_id] = label

        if (idx + 1) % 100 == 0:
            print(f"[{idx + 1}/{len(train_data)}] Labeled {len(labels_dict)} examples")

        # Save checkpoint
        if (idx + 1) % 500 == 0:
            with open(labels_file, "w") as f:
                json.dump(labels_dict, f)

    # Final save
    with open(labels_file, "w") as f:
        json.dump(labels_dict, f)

    print(f"\n✓ Saved {len(labels_dict)} labels to {labels_file}")
    
    label_counts = np.bincount([labels_dict[k] for k in labels_dict])
    print(f"  Class 0 (easy):  {label_counts[0]}")
    print(f"  Class 1 (hard):  {label_counts[1] if len(label_counts) > 1 else 0}")

if __name__ == "__main__":
    main()
