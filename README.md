# Credit Card Fraud Detection 

A production-minded, end-to-end data science project to detect fraudulent credit card transactions using the popular **Kaggle Credit Card Fraud** dataset (Time, V1–V28, Amount, Class).  
Includes: exploratory data analysis (EDA), model training (Logistic Regression & Random Forest), threshold tuning, evaluation (ROC-AUC & PR-AUC), and a minimal Streamlit app for batch scoring.

## Features
- **Jupyter Notebook** for EDA → `credit_card_fraud_detection.ipynb`
- **Imbalanced data handling**: class weighting by default; optional SMOTE (`--use-smote`, off by default, applied inside the pipeline so it only touches training data)
- **Pipelines** with `ColumnTransformer` and `StandardScaler` for `Amount`/`Time`
- Models: **Logistic Regression**, **Random Forest**, **Decision Tree** (baseline) -- choose with `--algo logreg|rf|dt`
- **Metrics**: ROC-AUC, PR-AUC, confusion matrix, precision/recall at custom thresholds
- **Model saving** with `joblib`
- **Streamlit app** to score uploaded CSVs and flag likely frauds

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train a model
Download `creditcard.csv` (Kaggle "Credit Card Fraud Detection", ULB), put it in the project folder, and point `--data-path` to it.

```bash
python train.py --data-path creditcard.csv --model-dir models --algo logreg
# or, each in its own folder so results do not overwrite each other
python train.py --data-path creditcard.csv --model-dir models_rf --algo rf
python train.py --data-path creditcard.csv --model-dir models_dt --algo dt
```

This will create a pipeline artifact at `models/model.joblib` (and `models/metadata.json`).

### 3) Run the Streamlit app
```bash
streamlit run app.py
```
- Upload a CSV with the same columns as the training data (no `Class` required for scoring). For honest metrics run `python make_holdout.py` and upload `test_holdout.csv` (the 20% test split).
- Adjust the **threshold** slider to trade off precision vs. recall.
- See flagged transactions and download predictions as CSV.

### 4) Explore the Notebook
Open `credit_card_fraud_detection.ipynb`
Walks through EDA, preprocessing, training, evaluation, and threshold tuning.

## Project Structure
```
cc-fraud-detection/
├── app.py                         # Streamlit batch-scoring app
├── requirements.txt
├── train.py                       # Train and save the model pipeline
├── README.md
├── credit_card_fraud_detection.ipynb
├── models/
│   ├── model.joblib              # Saved after training
│   └── metadata.json             # Training metadata (algo, timestamp, scores)
└── src/
    └── utils.py                  # Helper functions (thresholding, plots, metrics)
```
## Results
Stratified 80/20 split (`random_state=42`), 284,807 transactions, 492 frauds (0.17%). Every number comes from the `metadata.json` that `train.py` writes.
After training the models you want to compare, run `python results_table.py` -- it reads `models/`, `models_rf/` and `models_dt/` and prints the table below.

| Algorithm | SMOTE | ROC-AUC (test) | PR-AUC (test) |
|-----------|-------|----------------|---------------|
| Logistic Regression (class-weighted) | off | 0.972 | 0.716 |

PR-AUC is the headline metric: with 0.17% positives ROC-AUC is very optimistic.
Logistic Regression on the 20% hold-out (56,962 transactions, 98 frauds), threshold vs. trade-off (from the app):

| Threshold | TP | FP | FN | Precision | Recall |
|-----------|----|----|----|-----------|--------|
| 0.50 | 90 | 1386 | 8 | 0.061 | 0.918 |
| 0.92 | 87 | 217 | 11 | 0.286 | 0.888 |

Raising the threshold cuts false alarms sharply at the cost of a few missed frauds. Choosing a threshold by looking at this same test set is slightly optimistic;
a separate validation split would be needed for a final choice. Only quote precision/recall computed on held-out rows, never on the full CSV (it contains the training rows).

Dataset: Kaggle "Credit Card Fraud Detection" (ULB / Worldline).

## Notes
- The V1–V28 features are PCA components from the original dataset; we **only scale** `Amount` and `Time`.
- **SMOTE** is optional (`--use-smote`, off by default); when enabled it runs inside the pipeline so it only sees training data.
- For real deployments, consider **calibration**, drift monitoring, and adding business rules on top of the ML score.

## License
MIT
