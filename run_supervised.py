from data_utils import load_data, basic_info, prepare_features, train_test_scale
from supervised_model import train_supervised_model, predict_and_evaluate


def main():
    # 1. Load data
    path = "data/creditcard.csv"  # update if your path is different
    df = load_data(path)

    # 2. Basic info
    print("=== DATA INFO ===")
    basic_info(df)

    # 3. Prepare features
    X, y = prepare_features(df)

    # 4. Train-test split + scaling
    X_train, X_test, y_train, y_test = train_test_scale(X, y)

    # 5. Train model with SMOTE
    model = train_supervised_model(X_train, y_train)

    # 6. Evaluate on test set
    predict_and_evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
