"""Write test_holdout.csv = the same 20% test split that train.py evaluates on.
Upload it in the Streamlit app to see honest precision/recall (no training rows).
Usage: python make_holdout.py"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("creditcard.csv")
X, y = df.drop(columns=["Class"]), df["Class"].astype(int)
_, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y))
df.iloc[te].to_csv("test_holdout.csv", index=False)
print("Wrote test_holdout.csv with", len(te), "rows")
