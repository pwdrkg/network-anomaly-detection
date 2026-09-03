"""
Preprocessing utilities for UNSW-NB15 network anomaly detection.

Design principles (aligned with 2026 NIDS best practice):
- Per-split normalization: scalers/encoders are FIT ON TRAIN ONLY, then applied
  to test. This prevents information leakage and mirrors deployment, where the
  model meets future traffic it never saw during fitting.
- Skew-robust handling: heavy-tailed volume features are log1p-compressed and
  scaled with RobustScaler (median/IQR) so genuine bursts/floods are retained
  rather than clipped away.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Columns
ID_COL = "id"
TARGET_BIN = "label"
TARGET_MULTI = "attack_cat"
CATEGORICAL = ["proto", "service", "state"]

# Heavy-tailed volume/rate features to log-compress before scaling
SKEWED = ["dur", "sbytes", "dbytes", "rate", "sload", "dload",
          "sinpkt", "dinpkt", "sjit", "djit",
          "smean", "dmean", "response_body_len"]


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    return df


def add_domain_features(df):
    """Domain-derived features an analyst would reason about."""
    df = df.copy()
    eps = 1e-6
    df["bytes_per_pkt_src"] = df["sbytes"] / (df["spkts"] + eps)
    df["bytes_per_pkt_dst"] = df["dbytes"] / (df["dpkts"] + eps)
    df["src_dst_byte_ratio"] = df["sbytes"] / (df["dbytes"] + eps)
    df["total_bytes"] = df["sbytes"] + df["dbytes"]
    df["total_pkts"] = df["spkts"] + df["dpkts"]
    df["pkt_ratio"] = df["spkts"] / (df["dpkts"] + eps)
    df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
    return df


def clean(df):
    """Drop id, treat service '-' as explicit category, remove dup rows."""
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    df["service"] = df["service"].replace("-", "none")
    return df


def fit_preprocessor(train_df):
    """
    Fit encoders + scaler on TRAIN only. Returns an artifact dict used to
    transform any split identically.
    """
    train_df = clean(train_df)
    train_df = add_domain_features(train_df)

    # One-hot for low-cardinality; frequency-encode high-cardinality proto
    proto_freq = train_df["proto"].value_counts(normalize=True).to_dict()

    oh_cols = ["service", "state"]
    oh_categories = {c: sorted(train_df[c].unique().tolist()) for c in oh_cols}

    feature_cols = [c for c in train_df.columns
                    if c not in (CATEGORICAL + [TARGET_BIN, TARGET_MULTI])]

    # Build the encoded training frame to fit the scaler
    X = _encode(train_df, proto_freq, oh_cols, oh_categories, feature_cols)
    skewed_present = [c for c in SKEWED if c in X.columns]
    X[skewed_present] = np.log1p(X[skewed_present].clip(lower=0))

    scaler = RobustScaler()
    scaler.fit(X)

    return {
        "proto_freq": proto_freq,
        "oh_cols": oh_cols,
        "oh_categories": oh_categories,
        "feature_cols": feature_cols,
        "skewed_present": skewed_present,
        "scaler": scaler,
        "final_columns": X.columns.tolist(),
    }


def _encode(df, proto_freq, oh_cols, oh_categories, feature_cols):
    out = df[feature_cols].copy()
    # frequency encode proto (unseen -> 0)
    out["proto_freq"] = df["proto"].map(proto_freq).fillna(0.0)
    # one-hot with fixed category set from train
    for c in oh_cols:
        for cat in oh_categories[c]:
            out[f"{c}_{cat}"] = (df[c] == cat).astype(int)
    return out


def transform(df, artifact, with_target=True):
    """Apply a fitted preprocessor to any split."""
    df = clean(df)
    df = add_domain_features(df)
    X = _encode(df, artifact["proto_freq"], artifact["oh_cols"],
                artifact["oh_categories"], artifact["feature_cols"])
    X[artifact["skewed_present"]] = np.log1p(
        X[artifact["skewed_present"]].clip(lower=0))
    # align columns to training layout
    X = X.reindex(columns=artifact["final_columns"], fill_value=0)
    Xs = pd.DataFrame(artifact["scaler"].transform(X),
                      columns=artifact["final_columns"], index=X.index)
    if with_target:
        y = df[TARGET_BIN].values
        y_multi = df[TARGET_MULTI].values
        return Xs, y, y_multi
    return Xs
