from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay

def evaluate_and_save(name, y_true, y_prob, y_pred, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # AUC if probabilistic
    try:
        auc = roc_auc_score(y_true, y_prob)
        metrics["roc_auc"] = float(auc)
    except Exception:
        metrics["roc_auc"] = None

    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve if possible (and matplotlib available)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.title(f"ROC Curve - {name}")
        fig.savefig(os.path.join(out_dir, f"roc_{name}.png"), bbox_inches="tight", dpi=140)
        plt.close(fig)
    except Exception:
        # matplotlib not installed or plotting failed; skip ROC image
        pass

    return metrics
