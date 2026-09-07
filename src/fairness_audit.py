import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, sys
sys.path.insert(0, "src"); import preprocessing as pp
warnings.filterwarnings("ignore")
ACCENT, FIG = "#1F4E79", "figures"

# raw test set to recover the categorical group columns (proto/service)
raw = pp.load_raw("data/UNSW_NB15_testing-set.csv")
pre = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/best_model.joblib")
Xte, yte, cat = pp.transform(raw, pre)
proba = model.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)

raw2 = raw.copy()
raw2["service"] = raw2["service"].replace("-", "none")
raw2["y"] = yte; raw2["pred"] = pred

def rates(sub):
    y = sub["y"].values; p = sub["pred"].values
    P = (y == 1).sum(); N = (y == 0).sum()
    tpr = ((p == 1) & (y == 1)).sum() / P if P else np.nan   # recall / sensitivity
    fpr = ((p == 1) & (y == 0)).sum() / N if N else np.nan   # false-positive rate
    return tpr, fpr, len(sub)

# group by service (needs both classes present and enough rows)
groups = {}
for g, sub in raw2.groupby("service"):
    if len(sub) >= 300 and (sub["y"] == 1).sum() >= 30 and (sub["y"] == 0).sum() >= 30:
        tpr, fpr, n = rates(sub)
        groups[g] = {"tpr": round(float(tpr), 3), "fpr": round(float(fpr), 3), "n": int(n)}

# equalised-odds gaps: max-min across groups for TPR and FPR
tprs = [v["tpr"] for v in groups.values()]; fprs = [v["fpr"] for v in groups.values()]
R = {"per_service": groups,
     "tpr_gap": round(max(tprs) - min(tprs), 3),
     "fpr_gap": round(max(fprs) - min(fprs), 3),
     "note": "Equalised odds requires similar TPR and FPR across groups; gaps quantify the disparity."}

# plot grouped TPR/FPR
names = list(groups.keys())
x = np.arange(len(names)); w = 0.38
plt.figure(figsize=(9, 4.4))
plt.bar(x - w/2, [groups[n]["tpr"] for n in names], w, label="TPR (recall)", color=ACCENT)
plt.bar(x + w/2, [groups[n]["fpr"] for n in names], w, label="FPR (false alarms)", color="#C00000")
plt.xticks(x, names, rotation=30, ha="right"); plt.ylim(0, 1.05); plt.ylabel("rate")
plt.title("Equalised-odds view: TPR and FPR by service group")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG}/fairness_equalised_odds.png", dpi=110, bbox_inches="tight"); plt.close()

json.dump(R, open("reports/ext_fairness.json", "w"), indent=2)
print("FAIRNESS (equalised odds) DONE")
print("groups:", {k: (v["tpr"], v["fpr"]) for k, v in groups.items()})
print("TPR gap:", R["tpr_gap"], "| FPR gap:", R["fpr_gap"])
