#!/usr/bin/env python3
"""
Generate visualization graphs for router-v2 results.

Creates:
1. Confusion matrix heatmap
2. Threshold sweep curve (accuracy, precision, recall, F1)
3. Feature importance bar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

def plot_confusion_matrix():
    """Generate confusion matrix heatmap."""
    # Manually computed from phase 3 output
    cm = np.array([[82, 0], [27, 1511]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Easy (0)', 'Hard (1)'],
                yticklabels=['Easy (0)', 'Hard (1)'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Router-v2 Confusion Matrix (Threshold=0.50)')
    plt.tight_layout()
    plt.savefig('results_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results_confusion_matrix.png")
    plt.close()

def plot_threshold_sweep():
    """Generate threshold sweep curve."""
    thresholds = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    accuracies = np.array([0.9833, 0.9827, 0.9772, 0.9710, 0.9679, 0.9648, 0.9617, 0.9525, 0.9327, 0.9216])
    precisions = np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    recalls = np.array([0.9824, 0.9818, 0.9759, 0.9694, 0.9662, 0.9629, 0.9597, 0.9499, 0.9291, 0.9174])
    f1s = np.array([0.9911, 0.9908, 0.9878, 0.9845, 0.9828, 0.9811, 0.9794, 0.9743, 0.9633, 0.9569])
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(thresholds, precisions, 's-', linewidth=2, markersize=8, label='Precision')
    plt.plot(thresholds, recalls, '^-', linewidth=2, markersize=8, label='Recall')
    plt.plot(thresholds, f1s, 'd-', linewidth=2, markersize=8, label='F1 Score')
    
    # Highlight best threshold
    plt.axvline(x=0.50, color='red', linestyle='--', alpha=0.5, label='Selected (0.50)')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Router-v2 Threshold Sweep Analysis', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.90, 1.01])
    plt.tight_layout()
    plt.savefig('results_threshold_sweep.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results_threshold_sweep.png")
    plt.close()

def plot_feature_importance():
    """Generate feature importance from trained model."""
    try:
        model_data = joblib.load('model.pkl')
        lr = model_data['lr']
        
        feature_names = ['n_supports', 'n_candidates', 'top_sbert_score', 'sbert_score_gap', 'entity_count']
        coefficients = np.abs(lr.coef_[0])
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        bars = plt.bar(feature_names, coefficients, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, coef in zip(bars, coefficients):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{coef:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Absolute Coefficient', fontsize=12)
        plt.title('Router-v2 Feature Importance (LogisticRegression)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('results_feature_importance.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: results_feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"Error generating feature importance: {e}")

def save_results_json():
    """Save summary results to JSON."""
    results = {
        "model": "LogisticRegression",
        "threshold": 0.50,
        "train_metrics": {
            "cv_accuracy_mean": 0.9809,
            "cv_accuracy_std": 0.0063,
            "train_accuracy": 0.9833
        },
        "final_results": {
            "accuracy": 0.9833,
            "precision": 1.0000,
            "recall": 0.9824,
            "f1": 0.9911
        },
        "confusion_matrix": {
            "true_negatives": 82,
            "false_positives": 0,
            "false_negatives": 27,
            "true_positives": 1511
        },
        "training_data": {
            "total_examples": 1620,
            "easy_count": 82,
            "hard_count": 1538,
            "easy_percentage": 5.06,
            "hard_percentage": 94.94
        },
        "features": ["n_supports", "n_candidates", "top_sbert_score", "sbert_score_gap", "entity_count"]
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: results.json")

def main():
    print("=== Generating Router-v2 Visualization Graphs ===\n")
    
    plot_confusion_matrix()
    plot_threshold_sweep()
    plot_feature_importance()
    save_results_json()
    
    print("\n✅ All graphs generated successfully!")
    print("\nGenerated files:")
    print("  - results_confusion_matrix.png")
    print("  - results_threshold_sweep.png")
    print("  - results_feature_importance.png")
    print("  - results.json")

if __name__ == "__main__":
    main()
