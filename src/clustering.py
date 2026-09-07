import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
ACCENT, FIG = "#1F4E79", "figures"

df = pd.read_parquet("data/train_processed.parquet")
y = df["label"].values
X = df.drop(columns=["label", "attack_cat"])
# standardize for distance-based clustering; sample for tractable silhouette
Xz = StandardScaler().fit_transform(X)
rng = np.random.RandomState(42)
idx = rng.choice(len(Xz), 6000, replace=False)
Xs = Xz[idx]; ys = y[idx]

ks = list(range(2, 11))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.0f}  silhouette={sils[-1]:.3f}")

best_k = ks[int(np.argmax(sils))]

# how well does 2-cluster solution align with the true normal/attack split?
km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xs)
ari = adjusted_rand_score(ys, km2.labels_)
# purity: majority-label fraction per cluster
purity = np.mean([np.bincount(ys[km2.labels_ == c]).max() / (km2.labels_ == c).sum()
                  for c in range(2)])

R = {"ks": ks, "inertias": [float(i) for i in inertias], "silhouettes": [float(s) for s in sils],
     "best_k_by_silhouette": int(best_k), "k2_adjusted_rand_index": round(float(ari), 3),
     "k2_mean_purity": round(float(purity), 3)}

# Elbow + Silhouette side by side
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(ks, inertias, marker="o", color=ACCENT)
ax[0].set_title("Elbow method (inertia vs k)"); ax[0].set_xlabel("number of clusters (k)"); ax[0].set_ylabel("inertia (within-cluster SS)")
ax[1].plot(ks, sils, marker="s", color="#C00000")
ax[1].axvline(best_k, ls="--", color="gray"); ax[1].set_title("Silhouette score vs k")
ax[1].set_xlabel("number of clusters (k)"); ax[1].set_ylabel("mean silhouette")
ax[1].text(best_k + 0.1, min(sils), f"best k = {best_k}", color="gray")
plt.tight_layout(); plt.savefig(f"{FIG}/cluster_elbow_silhouette.png", dpi=110, bbox_inches="tight"); plt.close()

# PCA view coloured by cluster vs by true label (2 clusters)
emb = PCA(n_components=2, random_state=42).fit_transform(Xs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(emb[:, 0], emb[:, 1], c=km2.labels_, cmap="coolwarm", s=6, alpha=.5)
ax[0].set_title("K-Means clusters (k=2)")
ax[1].scatter(emb[:, 0], emb[:, 1], c=ys, cmap="coolwarm", s=6, alpha=.5)
ax[1].set_title("True labels (normal vs attack)")
for a in ax: a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{FIG}/cluster_pca_compare.png", dpi=110, bbox_inches="tight"); plt.close()

json.dump(R, open("reports/ext_clustering.json", "w"), indent=2)
print("\nCLUSTERING DONE")
print(f"best k by silhouette = {best_k}; 2-cluster ARI vs truth = {ari:.3f}; purity = {purity:.3f}")
