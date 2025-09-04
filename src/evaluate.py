import argparse, joblib, pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def main(model_path: str, csv_path: str, target: str):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target]); y = df[target].astype(int)
    y_proba = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X)
    y_pred = (y_proba >= 0.5).astype(int)
    print(classification_report(y, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y, y_proba):.4f}")
    print(f"PR-AUC: {average_precision_score(y, y_proba):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a saved model")
    ap.add_argument("--model", required=True, help="Path to saved model (.pkl/.joblib)")
    ap.add_argument("--data", required=True, help="Path to CSV with target column")
    ap.add_argument("--target", default="Class", help="Target column (default: Class)")
    args = ap.parse_args()
    main(args.model, args.data, args.target)
