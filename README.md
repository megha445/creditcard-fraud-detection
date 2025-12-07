🕵️‍♂️ Credit Card Fraud Detection – Machine Learning Project

This project builds an intelligent system to detect fraudulent credit card transactions using machine learning.
Fraud detection is a classic highly imbalanced classification problem, where fraudulent cases make up less than 0.2% of transactions — making standard models ineffective without special treatment.

📌 Dataset

Dataset used: Credit Card Fraud Detection

Source: Kaggle

Records: 284,807 transactions

Fraud cases: 492 (~0.17%)

Due to licensing constraints, the dataset is not included in this repository.
You can download it from Kaggle and place creditcard.csv inside:

data/creditcard.csv

🎯 Objective

Build a model that can:

✔ Detect fraudulent transactions
✔ Handle severe class imbalance
✔ Achieve high recall on fraud cases and strong ROC–AUC

🧠 Machine Learning Approach
🔹 Techniques Used

Random Forest (Supervised Learning)

SMOTE (Synthetic Minority Oversampling Technique)

Train–test split with stratification

Scaling numeric features

Performance evaluation via:

Confusion Matrix

Precision, Recall, F1-score

ROC–AUC

🔹 Why SMOTE?

Because fraud cases are extremely rare — oversampling helps prevent models from ignoring minority classes.

📂 Project Structure
creditcard_fraud/
├── data/
│   └── creditcard.csv          # dataset (not included in repo)
├── data_utils.py               # data loading, preprocessing, scaling
├── supervised_model.py         # RandomForest + SMOTE model training
├── isolation_model.py          # Isolation Forest anomaly detection (optional)
├── evaluation.py               # metrics & confusion matrix utilities
├── run_supervised.py           # entry point for supervised ML pipeline
├── run_isolation.py            # entry point for anomaly detection pipeline
└── requirements.txt            # required libraries

📊 Results (Supervised Model)

After training using SMOTE + RandomForest, results were approximately:

Metric	Fraud Class (Class 1)
Precision	~0.86
Recall	~0.84
F1-score	~0.85

Overall Accuracy: ~99.95%

ROC–AUC Score: ~0.97

📌 Interpretation:
✔ The model correctly detects ~84% of fraud cases — excellent performance for imbalanced fraud detection.
✔ False positives remain very low.

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add dataset

Place the dataset file inside:

data/creditcard.csv

3️⃣ Run supervised fraud detection pipeline
python run_supervised.py

4️⃣ Run anomaly detection pipeline (optional)
python run_isolation.py

✨ Potential Improvements

Hyperparameter tuning (GridSearch / RandomizedSearch)

Try XGBoost / LightGBM

SHAP explainability

Deploy as a web app (Flask / Streamlit)

REST API service for transaction scoring

📌 Tools & Libraries

Python

Pandas, NumPy

Scikit-learn

Imbalanced-learn

Matplotlib, Seaborn

📜 License & Dataset Notice

Dataset belongs to original authors (Kaggle/UCI repository).
It is excluded from this repository; users must download it manually.

🙌 Author

Megha Reddy
Computer Science Engineering Student — Machine Learning Enthusiast

💡 Feel free to ⭐ star the repo if you found it helpful!