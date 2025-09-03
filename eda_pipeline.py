# ============================
# 📊 Full EDA Pipeline for Parkinson's Project
# ============================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ======================
# 📂 Setup Directories
# ======================
os.makedirs("parkinsons_project/eda", exist_ok=True)
os.makedirs("parkinsons_project/data", exist_ok=True)

# ======================
# 📊 Load Data
# ======================
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(uci_url)

if "name" in df.columns:
    df = df.drop(columns=["name"])

# ✅ Rename columns for consistency
df.columns = df.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True).str.strip("_")

# Save clean dataset
df.to_csv("parkinsons_project/data/parkinsons.csv", index=False)

X = df.drop("status", axis=1)
y = df["status"]

# ======================
# 📊 Dataset Statistics
# ======================
stats_dir = "parkinsons_project/eda"

df.describe().T.to_csv(os.path.join(stats_dir, "summary_stats.csv"))
y.value_counts().rename({0:"Healthy", 1:"Parkinson's"}).to_csv(os.path.join(stats_dir, "target_distribution.csv"))
df.corr()["status"].abs().sort_values(ascending=False).to_csv(os.path.join(stats_dir, "correlation_with_target.csv"))

# ======================
# 🎨 Visualizations
# ======================
sns.set_theme(style="whitegrid", palette="muted")

# Target Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y, palette="Set2", ax=ax[0])
ax[0].set_title("Target Distribution (Count)")
ax[1].pie(y.value_counts(), labels=["Healthy", "Parkinson's"], autopct="%1.1f%%",
          colors=["#66c2a5", "#fc8d62"])
ax[1].set_title("Target Distribution (Pie)")
plt.savefig(os.path.join(stats_dir, "target_distribution.png"))
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(stats_dir, "corr_heatmap.png"))
plt.close()

# Pairplot of top correlated features
top_feats = df.corr()["status"].abs().sort_values(ascending=False).index[1:5]
sns.pairplot(df[top_feats.to_list() + ["status"]], hue="status", diag_kind="kde", palette="Set2")
plt.savefig(os.path.join(stats_dir, "pairplot_top_features.png"))
plt.close()

# Histograms & Violin plots
fig, axes = plt.subplots(len(top_feats), 2, figsize=(12, len(top_feats)*3))
for i, col in enumerate(top_feats):
    sns.histplot(df, x=col, hue="status", kde=True, ax=axes[i,0], palette="Set2", alpha=0.6)
    axes[i,0].set_title(f"Histogram + KDE: {col}")
    sns.violinplot(x="status", y=col, data=df, palette="Set2", ax=axes[i,1])
    axes[i,1].set_title(f"Violin Plot: {col}")
plt.tight_layout()
plt.savefig(os.path.join(stats_dir, "distributions_violin.png"))
plt.close()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set2", alpha=0.7)
plt.title("PCA Projection")
plt.savefig(os.path.join(stats_dir, "pca.png"))
plt.close()

# t-SNE
sample_size = min(300, len(X))
X_sample = X.sample(sample_size, random_state=42)
y_sample = y.loc[X_sample.index]
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
X_tsne = tsne.fit_transform(X_sample)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_sample, palette="Set2", alpha=0.7)
plt.title("t-SNE Projection (sample)")
plt.savefig(os.path.join(stats_dir, "tsne.png"))
plt.close()

print("✅ Full EDA completed – results saved in parkinsons_project/eda/")
