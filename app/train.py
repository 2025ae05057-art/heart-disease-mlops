"""Train and compare classification pipelines; log all runs to MLflow."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ROOT = Path(__file__).resolve().parents[1]

def load_data(source: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(source or URL, header=None, names=COLUMNS, na_values="?")
    if "target" not in df: raise ValueError("Dataset must contain target column")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
    for c in COLUMNS[:-1]: df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.empty: raise ValueError("Dataset is empty after cleaning")
    return df

def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUM_COLS), ("cat", categorical, CAT_COLS)])

def save_eda(df: pd.DataFrame, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    df.hist(figsize=(14,10), bins=20)
    plt.tight_layout(); plt.savefig(figures/"feature_histograms.png", dpi=150); plt.close("all")
    ax=df["target"].value_counts().sort_index().plot(kind="bar", rot=0)
    ax.set_xticks([0,1], labels=["No disease (0)","Disease (1)"]); ax.set_ylabel("Count"); ax.set_title("Target class balance")
    plt.tight_layout(); plt.savefig(figures/"class_balance.png", dpi=150); plt.close("all")
    plt.figure(figsize=(12,9)); corr=df.corr(numeric_only=True)
    plt.imshow(corr, aspect="auto", vmin=-1, vmax=1); plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90); plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature correlation heatmap"); plt.tight_layout(); plt.savefig(figures/"correlation_heatmap.png", dpi=150); plt.close()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data", default=None, help="Optional local CSV in UCI processed Cleveland format"); parser.add_argument("--quick", action="store_true", help="Use smaller search grids for a quick demo")
    args=parser.parse_args()
    df=load_data(args.data)
    print(f"Rows={len(df)}; columns={len(df.columns)}; missing cells={int(df.isna().sum().sum())}")
    save_eda(df, ROOT/"reports"/"figures")
    X=df.drop(columns="target"); y=df["target"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    grids={
      "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        {"clf__C": [0.1,1.0] if args.quick else [0.01,0.1,1.0,10.0]}),
      "random_forest": (RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"clf__n_estimators": [100], "clf__max_depth": [None,8]} if args.quick else {"clf__n_estimators": [100,250], "clf__max_depth": [None,5,10]})
    }
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT/"mlruns")))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-classification")
    best=None; best_score=-1; summary=[]
    for name,(estimator,param_grid) in grids.items():
      pipe=Pipeline([("prep",make_preprocessor()),("clf",estimator)])
      search=GridSearchCV(pipe,param_grid=param_grid,scoring="roc_auc",cv=cv,n_jobs=-1,refit=True)
      with mlflow.start_run(run_name=name) as run:
        search.fit(X_train,y_train)
        pred=search.predict(X_test); proba=search.predict_proba(X_test)[:,1]
        metrics={"cv_roc_auc_mean":float(search.best_score_),"test_accuracy":float(accuracy_score(y_test,pred)),"test_precision":float(precision_score(y_test,pred,zero_division=0)),"test_recall":float(recall_score(y_test,pred,zero_division=0)),"test_roc_auc":float(roc_auc_score(y_test,proba))}
        mlflow.log_params({"model":name,"cv_folds":5,"random_state":42,**{k:str(v) for k,v in search.best_params_.items()}})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            search.best_estimator_,
            artifact_path="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList","sklearn.tree._tree.Tree",
            ],
        )
        
        report=classification_report(y_test,pred,output_dict=True,zero_division=0)
        (ROOT/"reports"/f"{name}_classification_report.json").write_text(json.dumps(report,indent=2))
        mlflow.log_artifact(str(ROOT/"reports"/f"{name}_classification_report.json"))
        ConfusionMatrixDisplay.from_predictions(y_test,pred); plt.tight_layout(); cm=ROOT/"reports"/"figures"/f"{name}_confusion_matrix.png"; plt.savefig(cm,dpi=150); plt.close()
        RocCurveDisplay.from_predictions(y_test,proba); plt.tight_layout(); roc=ROOT/"reports"/"figures"/f"{name}_roc_curve.png"; plt.savefig(roc,dpi=150); plt.close()
        mlflow.log_artifact(str(cm)); mlflow.log_artifact(str(roc))
        summary.append({"model":name,"run_id":run.info.run_id,"best_params":search.best_params_,**metrics})
        if search.best_score_>best_score: best_score=search.best_score_; best=search.best_estimator_
        print(name, json.dumps(summary[-1],default=str))
    model_dir=ROOT/"model"; model_dir.mkdir(exist_ok=True); joblib.dump(best,model_dir/"model.joblib")
    (ROOT/"reports"/"model_comparison.json").write_text(json.dumps(summary,indent=2,default=str))
    print(f"Saved best pipeline to {model_dir/'model.joblib'}; selected by mean 5-fold CV ROC-AUC={best_score:.3f}")

if __name__=="__main__": main()
