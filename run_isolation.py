from data_utils import load_data, basic_info, prepare_features, train_test_scale
from isolation_model import train_isolation_forest, predict_and_evaluate_iso


def main():
    # 1. Load data
    path = "data/creditcard.csv"
    df = load_data(path)

    # 2. Basic info
    print("=== DATA INFO ===")
    basic_info(df)

    # 3. Prepare features
    X, y = prepare_features(df)

    # 4. Train-test split + scaling
    X_train, X_test, y_train, y_test = train_test_scale(X, y)

    # 5. Train Isolation Forest (unsupervised, uses only X_train)
    iso_model = train_isolation_forest(X_train)

    # 6. Evaluate (we have labels y_test, so we can measure performance)
    predict_and_evaluate_iso(iso_model, X_test, y_test)


if __name__ == "__main__":
    main()
