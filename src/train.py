"""
train.py — train and compare models on the processed UNSW-NB15 splits.

Trains Logistic Regression, Decision Tree, Random Forest, XGBoost and an
unsupervised Isolation Forest; evaluates on the held-out test split; writes a
metrics JSON, comparison figures, and the selected model (Random Forest,
chosen on recall then PR-AUC — the primary metrics defined in Step 1).

Usage:
    python src/features.py      # first, to build data/*_processed.parquet
    python src/train.py
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

ACCENT = "#1F4E79"


def main(data_dir, model_dir, fig_dir, report_dir):
    for d in (model_dir, fig_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    tr = pd.read_parquet(os.path.join(data_dir, "train_processed.parquet"))
    te = pd.read_parquet(os.path.join(data_dir, "test_processed.parquet"))
    ytr, yte = tr["label"].values, te["label"].values
    Xtr = tr.drop(columns=["label", "attack_cat"])
    Xte = te.drop(columns=["label", "attack_cat"])

    spw = (ytr == 0).sum() / (ytr == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=18, n_jobs=-1,
                                                class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                 scale_pos_weight=spw, n_jobs=-1, random_state=42),
    }

    results, curves = {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = _metrics(yte, pred, proba, time.time() - t0)
        fpr, tpr, _ = roc_curve(yte, proba)
        prec, rec, _ = precision_recall_curve(yte, proba)
        curves[name] = (fpr, tpr, prec, rec)
        joblib.dump(m, os.path.join(model_dir, name.replace(" ", "_").lower() + ".joblib"),
                    compress=3)
        print(f"{name:20s} recall={results[name]['recall']} f1={results[name]['f1']} "
              f"pr_auc={results[name]['pr_auc']}")

    # unsupervised baseline
    iso = IsolationForest(n_estimators=200, contamination=0.45, random_state=42, n_jobs=-1)
    iso.fit(Xtr[ytr == 0])
    score = -iso.score_samples(Xte)
    thr = np.quantile(score, 1 - yte.mean())
    pred_iso = (score >= thr).astype(int)
    results["Isolation Forest*"] = _metrics(yte, pred_iso, score, 0.0)

    _plot_comparison(results, fig_dir)
    _plot_curves(curves, results, fig_dir)

    # select best by primary metric: recall, then PR-AUC
    best = max([n for n in results if not n.endswith("*")],
               key=lambda n: (results[n]["recall"], results[n]["pr_auc"]))
    bm = joblib.load(os.path.join(model_dir, best.replace(" ", "_").lower() + ".joblib"))
    _plot_confusion(yte, (bm.predict_proba(Xte)[:, 1] >= 0.5).astype(int), best, fig_dir)
    joblib.dump(bm, os.path.join(model_dir, "best_model.joblib"), compress=3)

    json.dump({"results": results, "best_model": best,
               "test_prevalence": round(float(yte.mean()), 4)},
              open(os.path.join(report_dir, "step4_results.json"), "w"), indent=2)
    print(f"\nSelected best model: {best}")


def _metrics(y, pred, score, secs):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
        "train_sec": round(secs, 1),
    }


def _plot_comparison(results, fig_dir):
    metrics = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    names = list(results.keys())
    x = np.arange(len(metrics)); w = 0.15
    plt.figure(figsize=(11, 5))
    palette = ["#9DC3E6", "#5B9BD5", ACCENT, "#C00000", "#7F7F7F"]
    for i, n in enumerate(names):
        plt.bar(x + i * w, [results[n][mt] for mt in metrics], w, label=n,
                color=palette[i % len(palette)])
    plt.xticks(x + w * (len(names) - 1) / 2, [m.upper() for m in metrics])
    plt.ylim(0, 1.05); plt.ylabel("score"); plt.title("Model comparison (held-out test)")
    plt.legend(fontsize=8, ncol=3); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison.png"), dpi=110, bbox_inches="tight")
    plt.close()


def _plot_curves(curves, results, fig_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for n, (fpr, tpr, prec, rec) in curves.items():
        ax[0].plot(fpr, tpr, label=f"{n} ({results[n]['roc_auc']})")
        ax[1].plot(rec, prec, label=f"{n} ({results[n]['pr_auc']})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_title("ROC curves")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
    ax[1].set_title("Precision-Recall curves"); ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_roc_pr.png"), dpi=110, bbox_inches="tight")
    plt.close()


def _plot_confusion(y, pred, name, fig_dir):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(f"Confusion matrix — {name}"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_confusion.png"), dpi=110, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    main(a.data, a.models, a.figures, a.reports)
