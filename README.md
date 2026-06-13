# TrustGuard — Credit Card Fraud Detection

A machine learning system for detecting fraudulent credit card transactions in highly imbalanced data, built with a focus on **precision-recall tradeoffs** rather than naive accuracy — the right framing for real-world fraud detection.

## Problem

Credit card fraud detection is a classic extreme-imbalance problem: in the dataset used here, only **0.17% of transactions are fraudulent**. A model that predicts "not fraud" every time scores 99.83% accuracy while being completely useless. This project builds a model that actually catches fraud while minimizing false alarms.

## Dataset

[Credit Card Fraud Detection (Kaggle / ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 labeled frauds
- 28 anonymized PCA features (V1–V28) + Time + Amount
- Size: ~150 MB

## Approach

1. **EDA** — quantify class imbalance, transaction amount distributions
2. **Preprocessing** — feature scaling, stratified train/test split (preserves fraud ratio)
3. **Baseline** — class-weighted Logistic Regression
4. **Resampling** — SMOTE oversampling on training data only (avoids leakage)
5. **Model** — XGBoost classifier tuned for AUC-PR
6. **Evaluation** — Precision, Recall, F1, AUC-ROC, AUC-PR, confusion matrix
7. **Explainability** — SHAP feature importance to identify which transaction features drive fraud predictions
8. **Threshold tuning** — sweep decision thresholds to show the precision/recall tradeoff curve, framed as a business decision (cost of missed fraud vs. cost of false alarms)

## Results

| Model | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | TBD | TBD | TBD | TBD | TBD |
| XGBoost + SMOTE | TBD | TBD | TBD | TBD | TBD |

*(Fill in after running `notebooks/fraud_detection.ipynb` on Colab)*

## Tech Stack

- Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP
- pandas, NumPy, matplotlib, seaborn
- Google Colab (GPU/CPU)

## Repo Structure

```
TrustGuard/
├── notebooks/
│   └── TrustGuard.ipynb   # main pipeline — run on Colab
├── src/                          # (optional) refactored scripts
├── models/                        # saved model artifacts
├── results/                       # plots, metrics exports
└── README.md
```

## How to Run

1. Open `notebooks/TrustGuard.ipynb` in Google Colab
2. Download `creditcard.csv` from Kaggle (link above) and upload it, or use the Kaggle API (instructions in the notebook)
3. Run all cells top to bottom
4. Trained model saved as `fraud_xgb_model.pkl`

## Key Takeaways

- Naive accuracy is meaningless on imbalanced data — AUC-PR and recall on the minority class are the metrics that matter
- SMOTE applied only to training data (not test) to avoid data leakage
- SHAP analysis surfaces which anonymized features (V-columns) are most predictive of fraud
- Threshold tuning demonstrates how the same model can be deployed differently depending on business risk tolerance

## Author

Mani Chelluri —https://github.com/MANICHELLURII
