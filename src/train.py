import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

def load_xy(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def build_baseline(model: str):
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            random_state=RANDOM_STATE, class_weight="balanced_subsample"
        )
        return Pipeline([("model", clf)])
    elif model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    else:
        raise ValueError("model must be 'rf' or 'logreg'")

def main(input_csv: str, target: str, model: str, out_path: str, test_size: float):
    X, y = load_xy(input_csv, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    pipe = build_baseline(model)
    pipe.fit(Xtr, ytr)

    y_proba = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe[-1],"predict_proba") else pipe.decision_function(Xte)
    y_pred = (y_proba >= 0.5).astype(int)

    roc = roc_auc_score(yte, y_proba)
    pr_auc = average_precision_score(yte, y_proba)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | Precision@0.5: {prec:.4f} | Recall@0.5: {rec:.4f}")

    outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outp); print(f"Saved model to: {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a baseline fraud model")
    ap.add_argument("--input", required=True, help="Path to processed CSV")
    ap.add_argument("--target", default="Class", help="Target column name (default: Class)")
    ap.add_argument("--model", choices=["rf","logreg"], default="rf", help="Which model to train")
    ap.add_argument("--out", default="models/baseline_rf.pkl", help="Where to save the trained model")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    args = ap.parse_args()
    main(args.input, args.target, args.model, args.out, args.test_size)
