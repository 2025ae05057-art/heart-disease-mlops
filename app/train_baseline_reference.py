"""Baseline training script: downloads the UCI Cleveland heart-disease data,
trains a preprocessing + Logistic Regression pipeline, and saves model/model.joblib.

Run inside the venv you made for the API (same library versions as the server):
    python train_baseline.py
Or use a local copy of the data file:
    python train_baseline.py path\\to\\processed.cleveland.data
"""
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

source = sys.argv[1] if len(sys.argv) > 1 else URL
df = pd.read_csv(source, header=None, names=COLUMNS, na_values="?")
print("Rows:", len(df), "| missing values:", int(df.isna().sum().sum()))

X = df.drop(columns="target")
y = (df["target"] > 0).astype(int)  # 0 = no disease, 1 = disease

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUM_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
])
pipeline = Pipeline([("prep", preprocess),
                     ("clf", LogisticRegression(max_iter=1000))])

cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"5-fold CV ROC-AUC: {cv.mean():.3f} (+/- {cv.std():.3f})")

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]), 3))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.joblib")
print("Saved model/model.joblib")
