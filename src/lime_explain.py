import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer
warnings.filterwarnings("ignore")
FIG = "figures"

tr = pd.read_parquet("data/train_processed.parquet")
te = pd.read_parquet("data/test_processed.parquet")
Xtr = tr.drop(columns=["label", "attack_cat"]); Xte = te.drop(columns=["label", "attack_cat"])
yte = te["label"].values
model = joblib.load("models/best_model.joblib")

explainer = LimeTabularExplainer(
    Xtr.values, feature_names=list(Xtr.columns), class_names=["normal", "attack"],
    discretize_continuous=True, random_state=42)

# explain a confidently-predicted attack instance
proba = model.predict_proba(Xte)[:, 1]
i = int(np.where((yte == 1) & (proba > 0.95))[0][0])
exp = explainer.explain_instance(Xte.values[i], model.predict_proba, num_features=10)

fig = exp.as_pyplot_figure()
fig.set_size_inches(8, 5)
plt.title(f"LIME — why connection #{i} was flagged as ATTACK (p={proba[i]:.2f})", fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/lime_explanation.png", dpi=110, bbox_inches="tight"); plt.close()

top = exp.as_list()
json.dump({"instance": i, "proba_attack": round(float(proba[i]), 3), "top_reasons": top[:8]},
          open("reports/ext_lime.json", "w"), indent=2, default=str)
print("LIME DONE — instance", i, "p(attack)=", round(float(proba[i]), 3))
for feat, wt in top[:6]:
    print(f"   {feat:40s} {wt:+.3f}")
