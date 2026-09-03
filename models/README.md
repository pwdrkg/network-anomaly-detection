# Models

Saved, versioned artifacts. All are reproducible from the raw data via
`src/features.py` then `src/train.py`.

| File | Description |
|------|-------------|
| `best_model.joblib` | **Selected model — Random Forest** (recall 0.974, PR-AUC 0.989). Compressed. Used by `src/predict.py` and the deployment service. |
| `preprocessor.joblib` | Fitted preprocessing pipeline (encoders + scaler), fit on the training split only. Required for inference. |
| `feature_selection.json` | The 21-feature embedded + filter selection and the full 69-column layout. |
| `xgboost.joblib` | XGBoost model — the close-second alternative (recall 0.971). |
| `decision_tree.joblib`, `logistic_regression.joblib` | Baseline models for comparison. |

## Not committed (regenerate or use Git LFS)

The full-size Random Forest variants (`random_forest.joblib`,
`rf_for_shap.joblib`, `rf_smote_mitigated.joblib`, ~40–75 MB each) are
git-ignored. Recreate them with:

```bash
python src/train.py         # random_forest.joblib + best_model.joblib
python src/mitigations.py   # rf_smote_mitigated.joblib
```

For teams that prefer to version large binaries, track `models/*.joblib` with
[Git LFS](https://git-lfs.com/) instead.
