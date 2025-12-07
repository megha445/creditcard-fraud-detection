from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def print_confusion_matrix(y_true, y_pred) -> None:
    """
    Print confusion matrix in a readable format.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(cm)
    print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """
    Print classification report and ROC-AUC (if proba provided).
    """
    if title:
        print("=" * 60)
        print(title)
        print("=" * 60)

    print_confusion_matrix(y_true, y_pred)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception as e:
            print("Could not compute ROC-AUC:", e)
