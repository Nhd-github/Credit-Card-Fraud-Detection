"""Print a markdown results table from the metadata.json files written by train.py.
Usage:  python results_table.py
Reads models/ (logreg by default), models_rf/ and models_dt/ if they exist."""
import json, os

NAMES = {"logreg": "Logistic Regression (class-weighted)",
         "rf": "Random Forest",
         "dt": "Decision Tree (max_depth=6)"}

print("| Algorithm | SMOTE | ROC-AUC (test) | PR-AUC (test) |")
print("|-----------|-------|----------------|---------------|")
for folder in ("models", "models_rf", "models_dt"):
    path = os.path.join(folder, "metadata.json")
    if not os.path.exists(path):
        continue
    m = json.load(open(path))
    print(f"| {NAMES.get(m['algo'], m['algo'])} | {'on' if m['use_smote'] else 'off'} "
          f"| {m['roc_auc_test']:.3f} | {m['average_precision_test']:.3f} |")
