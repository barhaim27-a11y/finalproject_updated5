# ============================
# 🤖 Full Model Training Pipeline for Parkinson's Project
# ============================

import os, json, joblib, shap, shutil, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ======================
# 📂 Setup Directories
# ======================
os.makedirs("parkinsons_project/assets", exist_ok=True)
os.makedirs("parkinsons_project/models", exist_ok=True)
os.makedirs("parkinsons_project/data", exist_ok=True)

# ======================
# 📊 Load Data
# ======================
df = pd.read_csv("parkinsons_project/data/parkinsons.csv")
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 🤖 Define Models
# ======================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "NeuralNet": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])
}

# ======================
# 📈 Train & Evaluate
# ======================
metrics = {}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics[name] = auc
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

# Save metrics
with open("parkinsons_project/assets/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
pd.DataFrame(results).T.to_csv("parkinsons_project/assets/results.csv")

# Best model
best_name = max(metrics, key=metrics.get)
best_model = models[best_name]
joblib.dump(best_model, "parkinsons_project/models/best_model.joblib")
with open("parkinsons_project/models/best_model.txt", "w") as f:
    f.write(best_name)
print(f"✅ Best model: {best_name} (ROC-AUC={metrics[best_name]:.3f})")

# ======================
# 🏆 Leaderboard
# ======================
df_results = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
df_results.insert(0, "Rank", range(1, len(df_results)+1))
df_results.iloc[0, df_results.columns.get_loc("Rank")] = "🏆 1"
df_results.to_csv("parkinsons_project/assets/results_ranked.csv")

# ======================
# 📊 ROC & PR – All Models
# ======================
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test,y_proba):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curves – All Models")
plt.savefig("parkinsons_project/assets/roc_all.png")
plt.close()

plt.figure(figsize=(8,6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=name)
plt.legend()
plt.title("PR Curves – All Models")
plt.savefig("parkinsons_project/assets/pr_all.png")
plt.close()

# Plotly HTML
fig = go.Figure()
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name}"))
fig.write_html("parkinsons_project/assets/roc_all.html")

fig = go.Figure()
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name}"))
fig.write_html("parkinsons_project/assets/pr_all.html")

# ======================
# 📉 Learning Curve – Best Model
# ======================
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 5)
)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train AUC")
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV AUC")
plt.legend()
plt.title(f"Learning Curve – {best_name}")
plt.savefig("parkinsons_project/assets/learning_curve.png")
plt.close()

# ======================
# 📊 Confusion Matrix
# ======================
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy","Parkinson's"],
            yticklabels=["Healthy","Parkinson's"])
plt.title("Confusion Matrix")
plt.savefig("parkinsons_project/assets/confusion_matrix.png")
plt.close()

# ======================
# 🔎 SHAP – Best Model
# ======================
try:
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("parkinsons_project/assets/shap_summary.png")
    plt.close()
except Exception as e:
    print("⚠️ SHAP failed:", e)

# ======================
# 📄 README + requirements
# ======================
readme_md = """# Parkinson's Prediction – ML & AI Final Project

## Overview
This project implements a Machine Learning pipeline for predicting Parkinson's disease.

## Structure
- data/
- eda/
- models/
- assets/
- README.md
- requirements.txt
"""
with open("parkinsons_project/README.md", "w") as f:
    f.write(readme_md)

requirements = """pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
tensorflow
joblib
shap
openpyxl
plotly
"""
with open("parkinsons_project/requirements.txt", "w") as f:
    f.write(requirements)

# ======================
# 📦 Create ZIP
# ======================
shutil.make_archive("parkinsons_project", 'zip', "parkinsons_project")
print("🎉 Pipeline finished, ZIP created: parkinsons_project.zip")
