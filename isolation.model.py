import numpy as np
from sklearn.ensemble import IsolationForest

from evaluation import evaluate_classifier


def train_isolation_forest(
    X_train,
    contamination: float = 0.0017,
    random_state: int = 42,
):
    """
    Train an Isolation Forest for anomaly detection.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        n_jobs=-1,
        random_state=random_state,
    )
    print("Training IsolationForest (unsupervised)...")
    iso.fit(X_train)
    print("Training done.\n")
    return iso


def predict_and_evaluate_iso(model, X_test, y_test) -> None:
    """
    Use IsolationForest predictions (-1 anomaly, 1 normal),
    convert to 0/1 and evaluate as classification.
    """
    raw_pred = model.predict(X_test)
    # -1 => anomaly => fraud => 1,  1 => normal => non-fraud => 0
    y_pred = np.where(raw_pred == -1, 1, 0)

    # No probability scores here, so pass None
    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=None,
        title="IsolationForest (Unsupervised Anomaly Detection)",
    )
