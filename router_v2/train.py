#!/usr/bin/env python3
"""
Phase 3: Train logistic regression classifier with threshold tuning.

Process:
1. Load features (from Phase 2) and labels (from Phase 1)
2. Run 5-fold cross-validation
3. Fit final model on all training data
4. Sweep thresholds 0.50-0.95 to find best by accuracy
5. Save model.pkl (scaler + LR) and threshold.json
"""

import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Train router-v2 classifier")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (default: auto-select)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    print("Loading features and labels...")
    features = np.load(output_dir / "features_sbert_train.npz", allow_pickle=True)
    X_train = features["X"]

    with open(output_dir / "labels_train.json") as f:
        labels_dict = json.load(f)

    ids = features["ids"]
    y_train = np.array([labels_dict[id_] for id_ in ids], dtype=np.int32)

    print(f"Training set: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    label_counts = np.bincount(y_train)
    print(f"  Class 0: {label_counts[0]}")
    print(f"  Class 1: {label_counts[1] if len(label_counts) > 1 else 0}")

    print("\nTraining with 5-fold cross-validation...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nFitting final model...")
    lr.fit(X_scaled, y_train)
    y_pred_proba = lr.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(f"Train Accuracy (threshold=0.5): {accuracy_score(y_train, y_pred):.4f}")

    print("\nTuning threshold (0.50 - 0.95)...")
    thresholds = np.arange(0.50, 1.0, 0.05)
    results = []

    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        acc = accuracy_score(y_train, y_pred_thresh)
        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
        rec = recall_score(y_train, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_train, y_pred_thresh, zero_division=0)
        results.append((thresh, acc, prec, rec, f1))

    print("\nThreshold Sweep Results:")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for thresh, acc, prec, rec, f1 in results:
        print(f"{thresh:<12.2f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

    # Choose best threshold by accuracy
    if args.threshold is None:
        best_idx = np.argmax([r[1] for r in results])
        chosen_threshold = results[best_idx][0]
        print(f"\n✓ Auto-selected threshold: {chosen_threshold:.2f} (accuracy={results[best_idx][1]:.4f})")
    else:
        chosen_threshold = args.threshold
        print(f"\n✓ Using fixed threshold: {chosen_threshold:.2f}")

    # Save model and threshold
    model_file = output_dir / "model.pkl"
    threshold_file = output_dir / "threshold.json"

    model_data = {"scaler": scaler, "lr": lr}
    joblib.dump(model_data, model_file)
    print(f"✓ Saved model to {model_file}")

    with open(threshold_file, "w") as f:
        json.dump({"threshold": chosen_threshold}, f)
    print(f"✓ Saved threshold to {threshold_file}")

    # Final evaluation at chosen threshold
    y_final = (y_pred_proba >= chosen_threshold).astype(int)
    acc_final = accuracy_score(y_train, y_final)
    prec_final = precision_score(y_train, y_final, zero_division=0)
    rec_final = recall_score(y_train, y_final, zero_division=0)
    f1_final = f1_score(y_train, y_final, zero_division=0)
    cm = confusion_matrix(y_train, y_final)

    print(f"\n--- Final Results (Threshold = {chosen_threshold:.2f}) ---")
    print(f"  Accuracy:  {acc_final*100:.2f}%")
    print(f"  Precision: {prec_final*100:.2f}%")
    print(f"  Recall:    {rec_final*100:.2f}%")
    print(f"  F1:        {f1_final*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
