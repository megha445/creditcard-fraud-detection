import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluate_classifier


def train_supervised_model(
    X_train,
    y_train,
    n_estimators: int = 200,
    random_state: int = 42,
):
    """
    Apply SMOTE on training data and train RandomForest classifier.
    Returns trained model and resampled data shapes.
    """
    print("Before SMOTE class counts:", np.bincount(y_train))
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("After  SMOTE class counts:", np.bincount(y_res))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )

    print("\nTraining RandomForest on resampled data...")
    model.fit(X_res, y_res)
    print("Training done.\n")

    return model


def predict_and_evaluate(model, X_test, y_test) -> None:
    """
    Predict labels and probabilities, then evaluate.
    """
    y_pred = model.predict(X_test)

    # Some classifiers support predict_proba; RandomForest does.
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        title="RandomForest + SMOTE (Supervised Fraud Detection)",
    )
