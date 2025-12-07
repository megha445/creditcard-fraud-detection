import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset CSV.
    """
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    """
    Print basic info and class distribution.
    """
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nClass distribution:")
    print(df["Class"].value_counts())
    print("\nMissing values per column:")
    print(df.isna().sum())


def prepare_features(df: pd.DataFrame):
    """
    Split into features X and target y.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def train_test_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Train/test split with stratification and scale Time & Amount.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()

    # Copy to avoid SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Some datasets may not have Time/Amount if preprocessed differently
    for col in ["Time", "Amount"]:
        if col in X_train_scaled.columns:
            X_train_scaled[col] = scaler.fit_transform(
                X_train_scaled[[col]]
            )
            X_test_scaled[col] = scaler.transform(
                X_test_scaled[[col]]
            )

    return X_train_scaled, X_test_scaled, y_train, y_test
