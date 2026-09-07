import json, os
import mlflow

pass
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nad-model-comparison")

res = json.load(open("reports/ext_models.json"))["results"]
# approximate hyperparameters logged per model (as configured in training)
params = {
    "Logistic Regression": {"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000},
    "SVM (linear)": {"model": "LinearSVC+Calibrated", "class_weight": "balanced", "C": 1.0},
    "Decision Tree": {"model": "DecisionTree", "max_depth": 12, "class_weight": "balanced"},
    "Random Forest": {"model": "RandomForest", "n_estimators": 200, "max_depth": 18, "class_weight": "balanced"},
    "XGBoost": {"model": "XGBoost", "n_estimators": 300, "max_depth": 8, "learning_rate": 0.1},
    "Neural Net (MLP)": {"model": "MLPClassifier", "hidden_layers": "64-32", "activation": "relu", "early_stopping": True},
    "Isolation Forest*": {"model": "IsolationForest", "n_estimators": 200, "contamination": 0.45},
}
for name, m in res.items():
    with mlflow.start_run(run_name=name):
        for k, v in params.get(name, {"model": name}).items():
            mlflow.log_param(k, v)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            if metric in m:
                mlflow.log_metric(metric, m[metric])
        mlflow.set_tag("dataset", "UNSW-NB15")
        mlflow.set_tag("primary_metric", "recall + pr_auc")
# log the comparison figure as an artifact under a summary run
with mlflow.start_run(run_name="comparison-summary"):
    if os.path.exists("figures/model_comparison_ext.png"):
        mlflow.log_artifact("figures/model_comparison_ext.png")
    mlflow.set_tag("note", "seven-model comparison")

runs = mlflow.search_runs(experiment_names=["nad-model-comparison"])
print("MLFLOW DONE — logged", len(runs), "runs to ./mlruns")
print(runs[["tags.mlflow.runName", "metrics.recall", "metrics.pr_auc"]].to_string(index=False))
