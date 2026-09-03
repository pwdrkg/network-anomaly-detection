"""
predict.py — score new network connections with the trained detector.

Loads the fitted preprocessor and the selected model, applies the identical
leakage-safe transform, and outputs an attack probability + label per row.
This is the same entry point used by the deployment service (Step 8).

Usage:
    python src/predict.py --input data/UNSW_NB15_testing-set.csv --threshold 0.5
    python src/predict.py --input new_traffic.csv --output scored.csv
"""
import argparse
import os

import joblib
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp


def load(model_dir="models"):
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    return preprocessor, model


def score(df, preprocessor, model, threshold=0.5):
    X = pp.transform(df, preprocessor, with_target=False)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "attack_probability": proba.round(4),
        "prediction": (proba >= threshold).astype(int),
    }, index=df.index)
    out["label"] = out["prediction"].map({0: "normal", 1: "attack"})
    return out


def main(input_path, output_path, model_dir, threshold):
    df = pp.load_raw(input_path)
    preprocessor, model = load(model_dir)
    result = score(df, preprocessor, model, threshold)
    n_attack = int(result["prediction"].sum())
    print(f"scored {len(result):,} connections — flagged {n_attack:,} as attacks "
          f"({n_attack / len(result):.1%}) at threshold {threshold}")
    if output_path:
        result.to_csv(output_path, index=False)
        print("written:", output_path)
    else:
        print(result.head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV of raw connections (UNSW-NB15 schema)")
    ap.add_argument("--output", default=None, help="optional CSV path for scored results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    main(a.input, a.output, a.models, a.threshold)
