"""
features.py — build the processed feature matrix from raw UNSW-NB15.

Fits the preprocessing pipeline on the TRAIN split only (leakage-safe),
applies it to both splits, and caches the result for modelling.

Usage:
    python src/features.py \
        --train data/UNSW_NB15_training-set.csv \
        --test  data/UNSW_NB15_testing-set.csv
"""
import argparse
import os
import joblib
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp


def main(train_path, test_path, out_dir, model_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train = pp.load_raw(train_path)
    test = pp.load_raw(test_path)
    print(f"raw shapes  train={train.shape}  test={test.shape}")

    artifact = pp.fit_preprocessor(train)          # fit on train only
    Xtr, ytr, ytr_m = pp.transform(train, artifact)
    Xte, yte, yte_m = pp.transform(test, artifact)
    print(f"processed features: {Xtr.shape[1]}")

    Xtr.assign(label=ytr, attack_cat=ytr_m).to_parquet(
        os.path.join(out_dir, "train_processed.parquet"))
    Xte.assign(label=yte, attack_cat=yte_m).to_parquet(
        os.path.join(out_dir, "test_processed.parquet"))
    joblib.dump(artifact, os.path.join(model_dir, "preprocessor.joblib"))
    print("saved processed splits + preprocessor.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/UNSW_NB15_training-set.csv")
    ap.add_argument("--test", default="data/UNSW_NB15_testing-set.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    main(a.train, a.test, a.out, a.models)
