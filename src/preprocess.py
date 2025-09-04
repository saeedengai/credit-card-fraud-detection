import argparse
import pandas as pd
from pathlib import Path

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal placeholder cleaning; customize to match your notebook if needed
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    return df

def main(input_path: str, output_path: str):
    inp = Path(input_path); out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    cleaned = preprocess(df)
    cleaned.to_csv(out, index=False)
    print(f"Saved cleaned data to: {out} (rows={len(cleaned)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess raw credit card data")
    ap.add_argument("--input", required=True, help="Path to raw CSV")
    ap.add_argument("--output", required=True, help="Where to write cleaned CSV")
    args = ap.parse_args()
    main(args.input, args.output)
