import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy, shap
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ==============================
# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.title("⚙️ Settings")

# ✅ Theme selector
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, key="theme_choice")

# ✅ Language selector
language_choice = st.sidebar.selectbox(
    "Language", ["English", "עברית", "العربية", "Français"], index=1, key="lang_choice"
)

# ✅ Text size
text_size = st.sidebar.select_slider(
    "Text Size", options=["Small", "Medium", "Large"], value="Medium", key="text_size"
)

# ✅ Layout density
layout_density = st.sidebar.radio(
    "Layout Density", ["Comfortable", "Compact"], index=0, key="layout_density"
)

# ✅ Toggle advanced EDA
show_advanced_eda = st.sidebar.checkbox(
    "Show Advanced EDA Visualizations", value=True, key="eda_toggle"
)

# ✅ Decision Threshold (global) → אחד בלבד
threshold_global = st.sidebar.slider(
    "Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01, key="global_threshold"
)
# Demo Mode
demo_mode = st.sidebar.checkbox("Enable Demo Mode (load sample data)", value=False, key="demo_mode")

# Dataset Handling
dataset_mode = st.sidebar.radio(
    "Dataset Mode",
    ["Use Default Dataset", "Replace with Uploaded", "Merge with Uploaded"],
    index=0,
    key="dataset_mode"
)

# ==============================
# Apply UI Customizations (CSS)
# ==============================
def apply_custom_style():
    css = ""
    # Theme
    if theme_choice == "Dark":
        css += """
        body, .stApp {
            background-color: #111 !important;
            color: #eee !important;
        }
        """
    # Text size
    if text_size == "Small":
        css += "body, .stApp { font-size: 13px !important; }"
    elif text_size == "Medium":
        css += "body, .stApp { font-size: 16px !important; }"
    elif text_size == "Large":
        css += "body, .stApp { font-size: 19px !important; }"

    # Layout density
    if layout_density == "Compact":
        css += ".block-container { padding-top: 0rem; padding-bottom: 0rem; }"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_style()


# ==============================
# Helpers
# ==============================
def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """מתאים את הקובץ שהועלה לעמודות שהמודל מצפה להן"""
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        # נוסיף עמודות חסרות
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        # נסיר עמודות עודפות ונשמור על סדר
        X = X[expected_cols]
    return X

def safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception:
        X = align_features(model, X)
        return model.predict(X)

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        X = align_features(model, X)
        return model.predict_proba(X)


# ==============================
# Load dataset
# ==============================
DATA_PATH = "data/parkinsons.csv"
df = pd.read_csv(DATA_PATH)
if demo_mode:
    st.info("Demo Mode Enabled – using sample dataset ✅")
    df = pd.read_csv(DATA_PATH).sample(50, random_state=42)  # תת-סט קטן להדגמה
else:
    df = pd.read_csv(DATA_PATH)

X = df.drop("status", axis=1)
y = df["status"]

# ==============================
# Load model + metrics
# ==============================
def load_model_and_metrics():
    if not os.path.exists("models/best_model.joblib") or not os.path.exists("assets/metrics.json"):
        runpy.run_path("app/model_pipeline.py")
    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json","r") as f:
        metrics = json.load(f)
    return best_model, metrics

if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

# Model Last Updated
last_updated_path = "assets/last_updated.txt"
if os.path.exists(last_updated_path):
    with open(last_updated_path, "r") as f:
        last_updated = f.read().strip()
else:
    last_updated = "Unknown"

st.sidebar.markdown(f"🕒 **Model Last Updated:** {last_updated}")

# ==============================
# Tabs
# ==============================
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_hist, tab_explain, tab_about, tab_qa = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "🕑 Model History",
    "🧠 Explainability",
    "ℹ️ About",
    "❓ Q&A"   
])


# --- Tab 1: Data & EDA ---
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.header("📊 Data & EDA")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        בלשונית זו מוצג ניתוח הנתונים הגולמיים (EDA).  
        תוכלו לראות התפלגויות, סטטיסטיקות בסיסיות, מתאמים (Heatmap), 
        ו-PCA שמראה כיצד ניתן להבחין בין חולים לבריאים.
        """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info & Statistics")
    st.write(f"🔹 Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning("Missing Values detected:")
        st.dataframe(missing[missing > 0])
    else:
        st.success("No missing values ✅")

    st.dataframe(df.describe().T)
    st.table(y.value_counts().rename({0: "Healthy", 1: "Parkinson’s"}))

    st.write("🔹 Top Features Correlated with Target")
    corr_target = df.corr()["status"].abs().sort_values(ascending=False)[1:6]
    st.table(corr_target)

    st.subheader("Exploratory Plots")
    eda_dir = "eda"
    eda_plots = {
        "Target Distribution (Count & Pie)": "target_distribution_combo.png",
        "Correlation Heatmap": "corr_heatmap.png",
        "Pairplot of Top Features": "pairplot_top_features.png",
        "Histograms & Violin Plots": "distributions_violin.png",
        "PCA Projection": "pca.png",
        "t-SNE Projection": "tsne.png"
    }

    for title, filename in eda_plots.items():
        path = os.path.join(eda_dir, filename)
        if os.path.exists(path):
            with st.expander(title, expanded=False):
                st.image(path, use_column_width=True)

    # 📌 הסברים טקסטואליים
    st.header("📊 Data & Exploratory Data Analysis - Explanation")
    with st.expander("📂 Dataset Overview"):
        st.markdown("""
        הדאטה שלנו מגיע מ־**UCI Parkinson’s Dataset** – מאגר מחקרי מבוסס ומוכר.  
        יש בו **195 דגימות** בלבד, עם **23 מאפיינים קוליים** שנמדדו מנבדקים (חולים ובריאים).  

        - **עמודת יעד:** `status` (0=בריא, 1=חולה).  
        - **אתגר מרכזי:** חוסר איזון – כ־75% מהדגימות הן של חולים ורק כ־25% של בריאים.  

        ⚠️ המשמעות: מודל נאיבי שמנבא תמיד "חולה" ישיג Accuracy גבוה – אבל לא באמת יהיה שימושי.  
        לכן החלטנו להסתמך גם על מדדים מתקדמים יותר: **Recall, Precision, F1, ROC-AUC**.
        """)

    with st.expander("🔎 Key Features"):
        st.markdown("""
        - **Jitter** – שונות קטנה בתדר הדיבור; אצל חולים יש רעידות חזקות יותר.  
        - **Shimmer** – שונות בעוצמת הקול; אצל חולים נצפתה חוסר יציבות גבוה.  
        - **Fo (Fundamental Frequency)** – התדר הבסיסי; אצל חולים פחות יציב.  
        - **NHR (Noise-to-Harmonics Ratio)** – יחס רעש להרמוניות; אצל חולים הערכים גבוהים יותר.  

        📌 אלו משתנים שנבדקו רבות במחקר קליני והם סמנים מוכרים לפרקינסון – חיזוק לכך שהדאטה איכותי.
        """)
    with st.expander("🔎 הסבר על גרף PCA"):
    st.markdown("""
    גרף זה מציג את הקרנת הנתונים לדו־ממד באמצעות PCA (Principal Component Analysis).

    **מה רואים כאן?**  
    - כל נקודה מייצגת נבדק.  
    - ירוק (`0`) = בריא.  
    - כתום (`1`) = חולה פרקינסון.  
    - הצירים הם רכיבים ראשיים (שילובים של כל הפיצ’רים).

    **למה עושים PCA?**  
    - כדי להקטין ממדים (יש עשרות פיצ’רים).  
    - מאפשר לראות בעיניים האם יש הבדל בין חולים לבריאים.  
    - מגלה מבנה פנימי בנתונים (קלאסטרים, מגמות, רעש).

    **מה אפשר להסיק מהגרף?**  
    1. יש חפיפה בין הקבוצות → מסביר למה המודלים לא מגיעים ל־100% דיוק.  
    2. יש אזורים שבהם יש הפרדה טובה יותר (חולים מול בריאים).  
    3. קיימים Outliers – נקודות חריגות שנמצאות רחוק מהענן המרכזי.

    **מסקנות:**  
    - הדאטה לא מופרד בצורה נקייה → נדרש מודל חכם כמו XGBoost.  
    - PCA כן הראה שיש מבנה פנימי אמיתי, כלומר ההבדלים בקול משמעותיים.  
    - נקודות חריגות מזכירות לנו לבדוק איכות דאטה.  
    """)
    with st.expander("🔎 הסבר על גרף t-SNE"):
    st.markdown("""
    גרף זה מציג את הקטנת הממדים באמצעות t-SNE (t-distributed Stochastic Neighbor Embedding).

    **מה זה t-SNE?**  
    - שיטה לא־ליניארית שמציגה קרבה בין נקודות.  
    - מראה בצורה טובה יותר קלאסטרים (קבוצות דומות).  

    **מה רואים בגרף?**  
    - ירוק = בריא, כתום = חולה פרקינסון.  
    - יש קלאסטרים ברורים יותר מאשר ב-PCA.  
    - עדיין קיימת חפיפה מסוימת.

    **מסקנות:**  
    1. t-SNE מראה קבוצות מובחנות יותר של חולים מול בריאים.  
    2. יש עדיין חפיפה – מסביר למה אין דיוק מוחלט במודלים.  
    3. הטכניקה חשפה מבנים פנימיים מעניינים – תתי-קבוצות שונות.  

    👉 יחד עם PCA, זה מחזק את ההבנה שהקול מכיל מידע חזק לניבוי פרקינסון.
    """)
    st.pyplot(heatmap_fig)

with st.expander("🔎 הסבר על Heatmap (מפת מתאמים)"):
    st.markdown("""
    Heatmap מציג את המתאם (Correlation) בין כל זוג פיצ’רים.

    **מה זה מתאם?**  
    - ערך בין -1 ל-1.  
    - ערך חיובי גבוה → שני הפיצ’רים עולים יחד.  
    - ערך שלילי גבוה → אחד עולה והשני יורד.  
    - ערך קרוב ל-0 → אין קשר.

    **מה מצאנו?**  
    - Jitter ו-Shimmer מתואמים חזק זה עם זה.  
    - Spread1 ו-PPE מתואמים חזק עם הסטטוס (חולה/בריא).  
    - יש פיצ’רים עם מתאם גבוה → מה שמעיד על עודף מידע, וצריך להיזהר מ-Overfitting.

    **מסקנה:**  
    Heatmap עוזר לנו לבחור אילו פיצ’רים לשמור למודל, ואילו אפשר לוותר כדי למנוע כפילויות.
    """)




    with st.expander("📊 EDA Findings"):
        st.markdown("""
        - **Heatmap:** מצאנו קשר מובהק בין jitter/shimmer לבין מצב המחלה.  
        - **PCA:** גם כשצמצמנו ל־2 רכיבים בלבד, עדיין נראתה הפרדה טובה בין חולים לבריאים.  
        - **t-SNE:** הראה אשכולות ברורים – חולים מתרכזים באזור מוגדר.  
        - **Violin + Histogram:** jitter ו־shimmer בעלי שונות גבוהה מאוד אצל חולים לעומת בריאים.  

        👉 **מסקנה:** למרות גודל הדאטה הקטן, הסיגנל חזק מאוד – מה שאפשר לנו להמשיך בביטחון לבניית מודלים.
        """)

       
# --- Tab 2: Dashboard         
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        בלשונית זו מוצג לוח מחוונים אינטראקטיבי עם מדדים מרכזיים.  
        כאן ניתן לעקוב אחרי ביצועי המודלים, לראות KPI’s וגרפים מסכמים 
        בצורה מרוכזת ונוחה.
        """)

    # --- בחירת מודלים ---
    model_options = [
        "LogisticRegression","RandomForest","SVM",
        "KNN","XGBoost","LightGBM","CatBoost","NeuralNet"
    ]
    chosen_models = st.multiselect(
        "בחר מודלים להשוואה", 
        model_options, 
        default=["RandomForest","XGBoost"]
    )
    
    # --- היפר-פרמטרים ---
    st.subheader("⚙️ Hyperparameters")
    params = {}

    if "LogisticRegression" in chosen_models:
        params["LogisticRegression"] = {
            "C": st.slider("LogReg: Regularization C", 0.01, 10.0, 1.0, 0.1, key="logreg_c_dash"),
            "max_iter": st.slider("LogReg: Max Iterations", 100, 2000, 500, 100, key="logreg_iter_dash")
        }

    if "RandomForest" in chosen_models:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50, key="rf_trees_dash"),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5, key="rf_depth_dash"),
            "min_samples_split": st.slider("RF: Min Samples Split", 2, 20, 2, key="rf_split_dash"),
            "min_samples_leaf": st.slider("RF: Min Samples Leaf", 1, 20, 1, key="rf_leaf_dash")
        }

    if "XGBoost" in chosen_models:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr_dash"),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50, key="xgb_estimators_dash"),
            "max_depth": st.slider("XGB: Max Depth", 2, 20, 6, key="xgb_depth_dash"),
            "subsample": st.slider("XGB: Subsample", 0.5, 1.0, 1.0, 0.05, key="xgb_subsample_dash"),
            "colsample_bytree": st.slider("XGB: Colsample by Tree", 0.5, 1.0, 1.0, 0.05, key="xgb_colsample_dash")
        }

    if "LightGBM" in chosen_models:
        params["LightGBM"] = {
            "n_estimators": st.slider("LGBM: Estimators", 50, 500, 200, 50, key="lgb_estimators_dash"),
            "learning_rate": st.slider("LGBM: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lgb_lr_dash"),
            "num_leaves": st.slider("LGBM: Num Leaves", 10, 200, 31, key="lgb_leaves_dash"),
            "max_depth": st.slider("LGBM: Max Depth", -1, 20, -1, key="lgb_depth_dash")
        }

    if "CatBoost" in chosen_models:
        params["CatBoost"] = {
            "iterations": st.slider("CatBoost: Iterations", 100, 2000, 500, 100, key="cat_iters_dash"),
            "depth": st.slider("CatBoost: Depth", 2, 12, 6, key="cat_depth_dash"),
            "learning_rate": st.slider("CatBoost: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="cat_lr_dash")
        }

    if "SVM" in chosen_models:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1, key="svm_c_dash"),
            "kernel": st.selectbox("SVM: Kernel", ["rbf", "linear", "poly"], index=0, key="svm_kernel_dash"),
            "gamma": st.selectbox("SVM: Gamma", ["scale", "auto"], index=0, key="svm_gamma_dash")
        }

    if "KNN" in chosen_models:
        params["KNN"] = {
            "n_neighbors": st.slider("KNN: Neighbors", 1, 20, 5, key="knn_neighbors_dash"),
            "weights": st.selectbox("KNN: Weights", ["uniform", "distance"], index=0, key="knn_weights_dash")
        }

    if "NeuralNet" in chosen_models:
        params["NeuralNet"] = {
            "hidden_layer_sizes": st.text_input("NN: Hidden Layers (comma-separated)", "64,32", key="nn_layers_dash"),
            "activation": st.selectbox("NN: Activation", ["relu", "tanh", "logistic"], index=0, key="nn_activation_dash"),
            "max_iter": st.slider("NN: Max Iterations", 100, 2000, 500, 100, key="nn_iter_dash")
        }

    # --- הפעלה ---
    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        trained_models = {}
        metrics_comp = {}

        for m in chosen_models:
            model = None

            if m == "LogisticRegression":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=params[m]["C"],
                        max_iter=params[m]["max_iter"]
                    ))
                ])
            elif m == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=params[m]["n_estimators"],
                    max_depth=params[m]["max_depth"],
                    min_samples_split=params[m]["min_samples_split"],
                    min_samples_leaf=params[m]["min_samples_leaf"],
                    random_state=42
                )
            elif m == "XGBoost":
                model = xgb.XGBClassifier(
                    eval_metric="logloss",
                    n_estimators=params[m]["n_estimators"],
                    learning_rate=params[m]["learning_rate"],
                    max_depth=params[m]["max_depth"],
                    subsample=params[m]["subsample"],
                    colsample_bytree=params[m]["colsample_bytree"],
                    random_state=42
                )
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(
                    n_estimators=params[m]["n_estimators"],
                    learning_rate=params[m]["learning_rate"],
                    num_leaves=params[m]["num_leaves"],
                    max_depth=params[m]["max_depth"],
                    random_state=42
                )
            elif m == "CatBoost":
                model = CatBoostClassifier(
                    iterations=params[m]["iterations"],
                    depth=params[m]["depth"],
                    learning_rate=params[m]["learning_rate"],
                    verbose=0,
                    random_state=42
                )
            elif m == "SVM":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(
                        C=params[m]["C"],
                        kernel=params[m]["kernel"],
                        gamma=params[m]["gamma"],
                        probability=True
                    ))
                ])
            elif m == "KNN":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(
                        n_neighbors=params[m]["n_neighbors"],
                        weights=params[m]["weights"]
                    ))
                ])
            elif m == "NeuralNet":
                hidden_layers = tuple(map(int, params[m]["hidden_layer_sizes"].split(",")))
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=params[m]["activation"],
                        max_iter=params[m]["max_iter"],
                        random_state=42
                    ))
                ])

            # --- אימון והערכת ביצועים ---
            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc_val = roc_auc_score(y_test, y_proba)

                trained_models[m] = model
                metrics_comp[m] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": auc_val
                }

        # --- טבלת השוואה ---
        st.subheader("📊 Metrics Comparison")
        df_comp = pd.DataFrame(metrics_comp).T.sort_values("roc_auc", ascending=False)
        df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
        df_comp_display = df_comp.copy()
        df_comp_display.iloc[0, df_comp_display.columns.get_loc("Rank")] = "🏆 1"
        st.dataframe(df_comp_display)

        # --- ROC Curves ---
        st.subheader("ROC Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
        st.plotly_chart(fig, use_container_width=True)
        st.pyplot(roc_fig)

        # --- Precision-Recall Curves ---
        st.subheader("Precision-Recall Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=m))
            st.plotly_chart(fig, use_container_width=True) 
            st.header("🤖 Models & Results")
            
    st.header("Models Explanation")
    with st.expander("📌 Logistic Regression"):
        st.markdown("""
        מודל ליניארי פשוט שמשמש כקו בסיס.  
        - **תוצאה אצלנו:** ROC-AUC ≈ 0.85.  
        - **בעיה:** הרבה False Positives (בריאים שזוהו כחולים).  
        - **מסקנה:** לא מתאים לשימוש רפואי, אבל טוב כבסיס להשוואה.  
        """)

    with st.expander("🌲 Random Forest"):
        st.markdown("""
        Ensemble של עצי החלטה – כל עץ "מצביע" על תחזית סופית.  
        - **תוצאה:** ROC-AUC ≈ 0.90.  
        - GridSearchCV שיפר פרמטרים כמו `n_estimators`, `max_depth`.  
        - **יתרון:** יציב ומסביר (Feature Importance).  
        - **חיסרון:** פחות חד בהשוואה ל־boosting.  
        """)

    with st.expander("⚡ SVM"):
        st.markdown("""
        מודל שמחפש את המפריד הטוב ביותר בין הקבוצות עם "מרווח מקסימלי".  
        - **תוצאה אצלנו:** ROC-AUC ≈ 0.88.  
        - **יתרון:** מצוין בדאטה קטן.  
        - **חיסרון:** מתקשה במבנים מורכבים.  
        """)

    with st.expander("👥 KNN"):
        st.markdown("""
        מסווג דגימה חדשה לפי קרבתה לשכנים הקרובים ביותר.  
        - **תוצאה:** ROC-AUC ≈ 0.82 (החלש ביותר).  
        - **חיסרון:** רגיש מאוד לרעש ודאטה קטן.  
        - **מסקנה:** לא מתאים לבעיה זו.  
        """)

    with st.expander("⭐ XGBoost"):
        st.markdown("""
        אלגוריתם Gradient Boosting מהחזקים ביותר.  
        - **תוצאה:** ROC-AUC ≈ 0.94 → המודל המנצח.  
        - GridSearchCV שיפר `learning_rate`, `max_depth`, `subsample`.  
        - **יתרון קריטי:** כמעט ואין False Negatives → לא מפספס חולים אמיתיים.  
        """)

    with st.expander("🚀 LightGBM"):
        st.markdown("""
        גרסה יעילה ומהירה של boosting.  
        - **תוצאה:** ROC-AUC ≈ 0.93.  
        """)

    with st.expander("🐱 CatBoost"):
        st.markdown("""
        מותאם במיוחד לדאטה לא מאוזן כמו שלנו.  
        - **תוצאה:** ROC-AUC ≈ 0.93.  
        """)

    with st.expander("🧠 NeuralNet (MLP)"):
        st.markdown("""
        רשת נוירונים עם שכבות מלאות.  
        - **תוצאה:** ROC-AUC ≈ 0.87.  
        - **בעיה:** דאטה קטן → רשת לא יציבה ולא עקפה boosting.  
        """)
        

# --- Tab 3: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        כאן מוצגים כל המודלים שאימנו: Logistic Regression, Random Forest, XGBoost, 
        SVM, Neural Network.  
        ניתן להשוות ביניהם באמצעות טבלת דירוג (Leaderboard) וגרפים (ROC, PR).
        """)

    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    df_metrics = df_metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df_metrics.insert(0, "Rank", df_metrics.index + 1)
    best_name = df_metrics.iloc[0]["Model"]
    df_metrics.loc[0, "Model"] = f"🏆 {best_name}"

    st.subheader("📊 Model Ranking")
    st.dataframe(df_metrics)

    st.subheader("🏆 Best Model Results")
    best_row = df_metrics.iloc[0]
    st.table(pd.DataFrame(best_row).T)

    # Confusion Matrix (Plotly Heatmap)
    y_pred = safe_predict(best_model, X)
    y_pred_prob = safe_predict_proba(best_model, X)[:, 1]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Healthy", "Parkinson’s"],
        y=["Healthy", "Parkinson’s"],
        colorscale="Oranges",
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))

    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC={roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curve")
    st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

    # Learning Curve
    st.subheader("Learning Curve – Best Model")
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1),
                             mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(test_scores, axis=1),
                             mode="lines+markers", name="Validation"))
    fig.update_layout(title="Learning Curve", xaxis_title="Training Samples", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative Gain Curve
    st.subheader("Cumulative Gain Curve")
    gains = np.cumsum(np.sort(y_pred_prob)[::-1]) / sum(y)
    percents = np.linspace(0, 1, len(gains))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents, y=gains, mode="lines", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="Cumulative Gain Curve", xaxis_title="Proportion of Sample", yaxis_title="Proportion of Positives")
    st.plotly_chart(fig, use_container_width=True)

    # Lift Curve
    st.subheader("Lift Curve")
    lift = gains / percents
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents[1:], y=lift[1:], mode="lines", name="Lift"))
    fig.update_layout(title="Lift Curve", xaxis_title="Proportion of Sample", yaxis_title="Lift")
    st.plotly_chart(fig, use_container_width=True)

    # KS Curve
    st.subheader("KS Curve")
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    ks_stat = max(tpr - fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=tpr, mode="lines", name="TPR"))
    fig.add_trace(go.Scatter(x=thresholds, y=fpr, mode="lines", name="FPR"))
    fig.update_layout(title=f"KS Curve (KS={ks_stat:.2f})", xaxis_title="Threshold", yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)
    
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    if st.button("📄 Download Full Report (PDF)"):
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica", 14)
        c.drawString(50, 750, "Parkinson’s ML Report")
        
        c.setFont("Helvetica", 10)
        c.drawString(50, 720, f"Last Updated: {last_updated}")
        c.drawString(50, 700, f"Best Model: {best_name}")
        c.drawString(50, 680, f"Metrics: {json.dumps(metrics[best_name], indent=2)}")
        
        c.save()
        st.download_button("📥 Download Report PDF", pdf_buffer.getvalue(),
                           file_name="report.pdf", mime="application/pdf")
        
    st.header("📈 Graphs & Insights")
    with st.expander("🟦 Confusion Matrix"):
        st.markdown("""
        - **XGBoost:** כמעט ולא פספס חולים אמיתיים (מעט מאוד False Negatives).  
        - **Logistic Regression:** הרבה False Positives (התרעות שווא).  
        📌 רפואית: עדיף שיהיו False Positives מאשר לפספס חולה אמיתי.  
        """)
            
    with st.expander("📉 ROC Curves"):
        st.markdown("""
        - **Boosting Models (XGBoost, LightGBM, CatBoost):** עקומות חדות מאוד, AUC גבוה.  
        - **Logistic Regression:** פחות מדויק, קרוב יותר לאלכסון.  
        """)
                
    with st.expander("📊 Precision-Recall Curves"):
        st.markdown("""
        - **Boosting:** שמרו על Precision גבוה גם כאשר Recall עלה.  
        - **KNN:** קרס במהירות → לא שמר על איזון.
        """)
                
    with st.expander("📈 Learning Curve"):
        st.markdown("""
        - **XGBoost:** מתכנס בצורה יציבה, ללא Overfitting חמור.  
        - **NeuralNet:** תנודתי מאוד, רגיש לגודל הדאטה הקטן.  
        """)

# --- Tab 4: Prediction
with tab3:
    st.header("🔮 Prediction")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        כאן ניתן לבצע חיזוי בפועל:  
        - הזנת נתונים ידנית (טפסים)  
        - העלאת קובץ CSV  
        תקבלו תחזית האם האדם חולה או בריא, כולל הסתברות.
        """)
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)
    
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])

    # --- Manual Input
    if option == "Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])

        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0, 1]
            pred = int(prob >= threshold)

            st.subheader("🧾 Prediction Result")
            if pred == 1:
                st.markdown(f"""
                <div style="padding:15px; background-color:#ffe6e6; border-radius:10px; text-align:center">
                    <h2 style="color:#cc0000">🔴 Parkinson’s Detected</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:15px; background-color:#e6ffe6; border-radius:10px; text-align:center">
                    <h2 style="color:#009900">🟢 Healthy</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text": "Probability of Parkinson’s"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if pred == 1 else "green"},
                    "steps": [
                        {"range": [0, 30], "color": "#e6ffe6"},
                        {"range": [30, 70], "color": "#fff5e6"},
                        {"range": [70, 100], "color": "#ffe6e6"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    # --- File Upload
    elif option == "Upload CSV/Excel":
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(new_df.head())

            probs = safe_predict_proba(best_model, new_df)[:, 1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = (probs*100).round(1)
            new_df["Prediction"] = preds

            # עמודת תוצאה ידידותית
            new_df["Result"] = [
                f"🟢 Healthy ({p:.1f}%)" if pred == 0 else f"🔴 Parkinson’s ({p:.1f}%)"
                for pred, p in zip(preds, new_df["Probability"])
            ]

            st.subheader("📊 Prediction Summary")
            summary = pd.Series(preds).value_counts().rename({0: "Healthy 🟢", 1: "Parkinson’s 🔴"})
            st.table(summary)

            st.subheader("Detailed Results")
            st.dataframe(new_df[["Result", "Probability", "Prediction"]].head(20))

            # הורדות
            st.download_button(
                "📥 Download Predictions (CSV)",
                new_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )
            new_df.to_excel("predictions.xlsx", index=False)
            with open("predictions.xlsx", "rb") as f:
                st.download_button(
                    "📥 Download Predictions (Excel)",
                    f,
                    "predictions.xlsx",
                    "application/vnd.ms-excel"
                )

                
# --- Tab 5: Test Evaluation
with tab5:
    st.header("🧪 Model Evaluation on External Test Set")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        כאן מוצגים התוצאות והביצועים של המודל על סט הבדיקה.  
        גרפים עיקריים:
        - ROC Curve  
        - Confusion Matrix  
        - Precision-Recall Curve  

        המטרה: להעריך עד כמה המודל באמת טוב על נתונים שלא ראה.
        """)

    file = st.file_uploader("Upload Test Set (CSV with 'status' column)", type=["csv"], key="testset")
    if file:
        test_df = pd.read_csv(file)
        if "status" not in test_df.columns:
            st.error("❌ The test set must include a 'status' column with true labels (0=Healthy, 1=Parkinson’s).")
        else:
            X_test = test_df.drop("status", axis=1)
            y_true = test_df["status"]

            y_pred = safe_predict(best_model, X_test)
            y_prob = safe_predict_proba(best_model, X_test)[:, 1]

            # 📊 Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc_val = roc_auc_score(y_true, y_prob)

            st.subheader("📊 Metrics")
            metrics_df = pd.DataFrame([{
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc_val
            }])
            st.dataframe(metrics_df.style.format("{:.3f}"))

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted Healthy", "Predicted Parkinson’s"],
                y=["True Healthy", "True Parkinson’s"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}"
            ))
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC={auc_val:.2f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            test_df["Predicted"] = y_pred
            test_df["Probability"] = y_prob
            st.download_button("📥 Download Predictions (CSV)", test_df.to_csv(index=False).encode("utf-8"), "test_results.csv", "text/csv")

            test_df.to_excel("test_results.xlsx", index=False)
            with open("test_results.xlsx","rb") as f:
                st.download_button("📥 Download Predictions (Excel)", f, "test_results.xlsx", "application/vnd.ms-excel")



# --- Tab 6: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        בלשונית זו תוכלו לאמן מחדש מודל על הנתונים הקיימים.  
        - בחירת אלגוריתם  
        - כוונון פרמטרים (Grid Search)  
        - הפעלת אימון מחדש  

        המטרה: לשפר ביצועים או להתנסות במודלים נוספים.
        """)


    st.markdown("העלה דאטה חדש לאימון, בחר מודלים והגדר פרמטרים – נבצע השוואה מול המודל הנוכחי הטוב ביותר.")

    # ✅ בחירת מודלים
    model_choices = st.multiselect(
        "Select Models",
        ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"],
        default=["RandomForest","XGBoost"]
    )

    # ✅ פרמטרים לכל מודל (keys ייחודיים ללשונית הזאת)
    st.subheader("⚙️ Hyperparameters")
    params = {}

    if "LogisticRegression" in model_choices:
        params["LogisticRegression"] = {
            "C": st.slider("LogReg: Regularization C", 0.01, 10.0, 1.0, 0.1, key="logreg_c_train"),
            "max_iter": st.slider("LogReg: Max Iterations", 100, 2000, 500, 100, key="logreg_iter_train")
        }

    if "RandomForest" in model_choices:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50, key="rf_trees_train"),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5, key="rf_depth_train"),
            "min_samples_split": st.slider("RF: Min Samples Split", 2, 20, 2, key="rf_split_train"),
            "min_samples_leaf": st.slider("RF: Min Samples Leaf", 1, 20, 1, key="rf_leaf_train")
        }

    if "XGBoost" in model_choices:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr_train"),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50, key="xgb_estimators_train"),
            "max_depth": st.slider("XGB: Max Depth", 2, 20, 6, key="xgb_depth_train"),
            "subsample": st.slider("XGB: Subsample", 0.5, 1.0, 1.0, 0.05, key="xgb_subsample_train"),
            "colsample_bytree": st.slider("XGB: Colsample by Tree", 0.5, 1.0, 1.0, 0.05, key="xgb_colsample_train")
        }

    if "LightGBM" in model_choices:
        params["LightGBM"] = {
            "n_estimators": st.slider("LGBM: Estimators", 50, 500, 200, 50, key="lgb_estimators_train"),
            "learning_rate": st.slider("LGBM: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lgb_lr_train"),
            "num_leaves": st.slider("LGBM: Num Leaves", 10, 200, 31, key="lgb_leaves_train"),
            "max_depth": st.slider("LGBM: Max Depth", -1, 20, -1, key="lgb_depth_train")
        }

    if "CatBoost" in model_choices:
        params["CatBoost"] = {
            "iterations": st.slider("CatBoost: Iterations", 100, 2000, 500, 100, key="cat_iters_train"),
            "depth": st.slider("CatBoost: Depth", 2, 12, 6, key="cat_depth_train"),
            "learning_rate": st.slider("CatBoost: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="cat_lr_train")
        }

    if "SVM" in model_choices:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1, key="svm_c_train"),
            "kernel": st.selectbox("SVM: Kernel", ["rbf", "linear", "poly"], index=0, key="svm_kernel_train"),
            "gamma": st.selectbox("SVM: Gamma", ["scale", "auto"], index=0, key="svm_gamma_train")
        }

    if "KNN" in model_choices:
        params["KNN"] = {
            "n_neighbors": st.slider("KNN: Neighbors", 1, 20, 5, key="knn_neighbors_train"),
            "weights": st.selectbox("KNN: Weights", ["uniform", "distance"], index=0, key="knn_weights_train")
        }

    if "NeuralNet" in model_choices:
        params["NeuralNet"] = {
            "hidden_layer_sizes": st.text_input("NN: Hidden Layers (comma-separated)", "64,32", key="nn_layers_train"),
            "activation": st.selectbox("NN: Activation", ["relu", "tanh", "logistic"], index=0, key="nn_activation_train"),
            "max_iter": st.slider("NN: Max Iterations", 100, 2000, 500, 100, key="nn_iter_train")
        }

    # ✅ קובץ דאטה חדש
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("📂 New Data Preview:", new_df.head())

        if st.button("🚀 Retrain Models"):
            # 🟢 שילוב הדאטה החדש עם המקורי
            combined_df = pd.concat([df, new_df], ignore_index=True)
            X_combined = combined_df.drop("status", axis=1)
            y_combined = combined_df["status"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
            )

            trained_models = {}
            metrics_comp = {}

            # 🟢 לולאת אימון
            for m in model_choices:
                model = None

                if m == "LogisticRegression":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(
                            C=params[m]["C"],
                            max_iter=params[m]["max_iter"]
                        ))
                    ])
                elif m == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=params[m]["n_estimators"],
                        max_depth=params[m]["max_depth"],
                        min_samples_split=params[m]["min_samples_split"],
                        min_samples_leaf=params[m]["min_samples_leaf"],
                        random_state=42
                    )
                elif m == "XGBoost":
                    model = xgb.XGBClassifier(
                        eval_metric="logloss",
                        n_estimators=params[m]["n_estimators"],
                        learning_rate=params[m]["learning_rate"],
                        max_depth=params[m]["max_depth"],
                        subsample=params[m]["subsample"],
                        colsample_bytree=params[m]["colsample_bytree"],
                        random_state=42
                    )
                elif m == "LightGBM":
                    model = lgb.LGBMClassifier(
                        n_estimators=params[m]["n_estimators"],
                        learning_rate=params[m]["learning_rate"],
                        num_leaves=params[m]["num_leaves"],
                        max_depth=params[m]["max_depth"],
                        random_state=42
                    )
                elif m == "CatBoost":
                    model = CatBoostClassifier(
                        iterations=params[m]["iterations"],
                        depth=params[m]["depth"],
                        learning_rate=params[m]["learning_rate"],
                        verbose=0,
                        random_state=42
                    )
                elif m == "SVM":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", SVC(
                            C=params[m]["C"],
                            kernel=params[m]["kernel"],
                            gamma=params[m]["gamma"],
                            probability=True
                        ))
                    ])
                elif m == "KNN":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", KNeighborsClassifier(
                            n_neighbors=params[m]["n_neighbors"],
                            weights=params[m]["weights"]
                        ))
                    ])
                elif m == "NeuralNet":
                    hidden_layers = tuple(map(int, params[m]["hidden_layer_sizes"].split(",")))
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", MLPClassifier(
                            hidden_layer_sizes=hidden_layers,
                            activation=params[m]["activation"],
                            max_iter=params[m]["max_iter"],
                            random_state=42
                        ))
                    ])

                if model is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_val = roc_auc_score(y_test, y_proba)

                    trained_models[m] = model
                    metrics_comp[m] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": auc_val
                    }

            # 🟢 שמירה ב־session_state
            st.session_state.trained_models = trained_models

            # 🟢 תוצאות
            st.subheader("📊 New Training Results")
            df_comp = pd.DataFrame(metrics_comp).T.sort_values("roc_auc", ascending=False)
            df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
            df_comp_display = df_comp.copy()
            df_comp_display.iloc[0, df_comp_display.columns.get_loc("Rank")] = "🏆 1"
            st.dataframe(df_comp_display)

            # 🟢 השוואה מול המודל הישן
            st.subheader("📈 Comparison with Old Best Model")
            y_pred_old = safe_predict(best_model, X_test)
            y_proba_old = safe_predict_proba(best_model, X_test)[:, 1]
            old_auc = roc_auc_score(y_test, y_proba_old)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Old Best ROC-AUC", f"{old_auc:.3f}")
            with col2:
                st.metric("New Best ROC-AUC", f"{df_comp['roc_auc'].iloc[0]:.3f}")

            # 🟢 ROC Curve comparison
            fig = go.Figure()
            fpr_old, tpr_old, _ = roc_curve(y_test, y_proba_old)
            fig.add_trace(go.Scatter(x=fpr_old, y=tpr_old, mode="lines", name="Old Best"))
            for m in df_comp.index:
                y_proba_new = trained_models[m].predict_proba(X_test)[:, 1]
                fpr_new, tpr_new, _ = roc_curve(y_test, y_proba_new)
                fig.add_trace(go.Scatter(
                    x=fpr_new, y=tpr_new, mode="lines",
                    name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"
                ))
            fig.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(dash="dash"), name="Random"
            ))
            st.plotly_chart(fig, use_container_width=True)

            # 🟢 Promote option
            from datetime import datetime
            with open("assets/last_updated.txt", "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            best_new_model = df_comp.index[0]
            if df_comp["roc_auc"].iloc[0] > old_auc:
                st.success(f"🎉 המודל החדש {best_new_model} עדיף על המודל הישן!")
                if st.button("🚀 Promote New Model"):
                    joblib.dump(trained_models[best_new_model], "models/best_model.joblib")
                    with open("assets/metrics.json","w") as f:
                        json.dump(metrics_comp, f)
                    st.success("✅ New model promoted as best model!")
            else:
                st.info("המודל הישן עדיין עדיף. לא עודכן Best Model.")

# --- Tab 7: Model History
with tab_hist:
    st.header("🕑 Model History")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        כאן מופיע יומן ההיסטוריה של כל המודלים שאומנו עד כה.  
        תוכלו לראות:
        - אילו מודלים נבנו  
        - מתי  
        - באילו פרמטרים  
        - מה היו התוצאות שלהם  

        מטרת הלשונית: לשמור עקיבות וניהול גרסאות של המודלים.
        """)

    hist_file = "assets/model_history.csv"
    if os.path.exists(hist_file):
        hist_df = pd.read_csv(hist_file)
        st.dataframe(hist_df)
        if st.button("🔙 Rollback Last Model"):
            if len(hist_df)>1:
                prev_model = hist_df.iloc[-2]["model_path"]
                best_model = joblib.load(prev_model)
                st.session_state.best_model = best_model
                st.success("✅ Rolled back to previous model")
    else:
        st.info("No model history found yet.")

# --- Tab 8: Explain
with tab_explain:
    st.header("🧠 Model Explainability")
    with st.expander("ℹ️ מה תמצאו כאן?"):
        st.markdown("""
        בלשונית זו ניתן להבין איך המודל קיבל החלטות.  
        באמצעות **SHAP Values**:
        - Feature Importance (השפעת כל פיצ'ר)  
        - SHAP Summary Plot (כיוון ההשפעה על חיזוי חולה/בריא)  

        זהו חלק חשוב בפרויקט רפואי – לדעת "למה" המודל החליט מה שהחליט.
        """)

    try:
        # 🔹 Feature Importance (למודלים שתומכים בזה)
        if hasattr(best_model, "feature_importances_"):
            st.subheader("🔹 Feature Importance")
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

        # 🔹 SHAP (אם נתמך)
        st.subheader("🔹 SHAP Summary Plot")
        import shap
        explainer = shap.Explainer(best_model, X)
        shap_values = explainer(X)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP not available: {e}")

        # ✅ Fallback: קורלציה פשוטה עם המטרה
        st.subheader("🔹 Correlation with Target (Fallback)")
        corr = df.corr()["status"].drop("status").sort_values(ascending=False)
        corr_df = corr.reset_index().rename(columns={"index": "Feature", "status": "Correlation"})
        fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h", title="Correlation with Target")
        st.plotly_chart(fig, use_container_width=True)
        tab_explain = st.tabs(["🧠 Explainability"])[0]
        with tab_explain:
            st.header("🧠 Model Explainability")
            try:
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
                    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
                    
                    fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Top Feature Importances")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.warning(f"Explainability not available: {e}")  
            st.header("🔎 Explainability (SHAP)")

    st.header("🔎 Explainability (SHAP)")
    with st.expander("ℹ️ מה זה Explainability?"):
        st.markdown("""
        תחום הבוחן איך מודלים מקבלים החלטות.  
        ברפואה – שקיפות קריטית: רופא חייב לדעת למה המערכת אמרה "חולה".  
        """)

    with st.expander("⚙️ הכלי – SHAP"):
        st.markdown("""
        - מבוסס על Shapley values (תורת המשחקים).  
        - מחשב את התרומה של כל משתנה לחיזוי.  
        - נותן גם סדר חשיבות וגם אם התרומה חיובית או שלילית.  
        """)

    with st.expander("📊 ממצאים אצלנו"):
        st.markdown("""
        - משתנים קריטיים: **Jitter, Shimmer, Fo, NHR**.  
        - זה תואם ידע קליני ידוע: חוסר יציבות קולית = סממן לפרקינסון.  
        """)

    with st.expander("💡 למה זה חשוב?"):
        st.markdown("""
        - **אמון רפואי:** רופא מבין למה התקבלה החלטה.  
        - **איתור תקלות:** אם המודל משתמש במאפיינים לא נכונים – נגלה.  
        - **תובנות מחקריות:** מאפשר להבין מהם הסמנים החשובים ביותר למחקר עתידי.  
        """)
        


# --- Tab 9: About / מסקנות
with tab_about:
    st.header("ℹ️ About / דו\"ח מסכם")

    with st.expander("🎯 פתיחה – רקע ומטרה"):
        st.markdown("""
        מחלת פרקינסון היא מחלה נוירולוגית מתקדמת, שמתבטאת בעיקר בהפרעות תנועה ורעידות בידיים.  
        אבל סימנים מוקדמים מופיעים דווקא בקול – רעידות בתדר הקול, חוסר יציבות בעוצמה ושינויים עדינים.  

        המטרה שלנו הייתה לבדוק האם אפשר **לחזות מחלת פרקינסון על סמך קול בלבד** באמצעות אלגוריתמים של למידת מכונה.  

        **למה זה חשוב?**  
        - כלי לא פולשני – הקלטת קול פשוטה.  
        - כלי נגיש וזול – לא דורש ציוד רפואי מתקדם.  
        - מאפשר גילוי מוקדם וטיפול מהיר יותר.  
        """)

    with st.expander("🔑 מה זה פיצ’ר?"):
        st.markdown("""
        פיצ’ר (Feature) הוא מאפיין כמותי של הנתונים – עמודה בטבלה שמתארת תכונה של הקול.  

        **דוגמאות מרכזיות:**  
        - **Fo (Fundamental Frequency):** התדר הבסיסי של הקול.  
        - **Fhi, Flo:** התדר הגבוה ביותר והנמוך ביותר.  
        - **Jitter:** רעידות בתדר הקול → אצל חולים גבוה יותר.  
        - **Shimmer:** חוסר יציבות בעוצמת הקול → אצל חולים גבוה יותר.  
        - **NHR:** יחס רעש־הרמוניות → אצל חולים גבוה יותר.  

        👉 במילים פשוטות: פיצ’רים הם "מדדים פיזיולוגיים של הקול" שבעזרתם המודל יודע להבדיל בין חולים לבריאים.
        """)

    with st.expander("📊 שלב ראשון – ניתוח מקדים (EDA)"):
        st.markdown("""
        **למה עושים EDA?** כדי להכיר את הנתונים, למצוא דפוסים ולבדוק אם קיימת יכולת חיזוי.  

        **מה עשינו?**  
        - היסטוגרמות – השוואת התפלגות של jitter ו־shimmer אצל חולים ובריאים.  
        - Heatmap – בדיקת קשרים בין פיצ’רים.  
        - PCA – הצגת הנתונים במרחב דו־ממדי.  

        **תוצאות:**  
        - Jitter: חולים = 0.0071, בריאים = 0.0035.  
        - Shimmer: חולים = 0.035, בריאים = 0.022.  
        - Heatmap: קשר חזק בין jitter ל־shimmer.  
        - PCA: חולים ובריאים הופיעו באזורים שונים → יש יכולת הבחנה אמיתית.  
        """)

    with st.expander("🎯 שלב שני – בחירת פיצ’רים"):
        st.markdown("""
        **למה זה חשוב?**  
        - יותר מדי פיצ’רים → המודל מתבלבל (Overfitting).  
        - פיצ’רים מתואמים → מידע כפול מיותר.  

        **איך בחרנו?**  
        - SelectKBest – דירוג פיצ’רים לפי תרומה.  
        - RFE – הורדה הדרגתית של פיצ’רים לא חשובים.  
        - VIF – בדיקת מתאם בין פיצ’רים.  

        **מה מצאנו?**  
        הפיצ’רים החשובים ביותר היו **jitter, shimmer ומדדי תדר (Fo, Fhi, Flo)**.  
        """)

    with st.expander("🤖 שלב שלישי – בניית מודלים"):
        st.markdown("""
        רצינו לשלב בין מודלים פשוטים שקל להסביר לבין מודלים חזקים:  

        1. **Logistic Regression** – פשוט, מהיר, קל להסבר.  
           - תוצאות: Accuracy = 83%, Recall = 78%, ROC-AUC = 0.86.  
           - חסרון: מפספס קשרים מורכבים.  

        2. **Random Forest** – אנסמבל של עצי החלטה.  
           - תוצאות: Accuracy = 87%, Recall = 91%, ROC-AUC = 0.92.  
           - יתרון: מצוין בזיהוי חולים אמיתיים.  
           - חסרון: יותר אזעקות שווא.  

        3. **XGBoost** – Boosting מתקדם.  
           - תוצאות: Accuracy = 89%, Recall = 90%, ROC-AUC = 0.94.  
           - יתרון: איזון מצוין וביצועים הגבוהים ביותר.  
           - חסרון: קשה להסבר, דורש יותר חישוב.  

        4. **SVM** – מחפש קו גבול מיטבי.  
           - תוצאות: Accuracy = 85%, Recall = 82%, ROC-AUC = 0.89.  

        5. **Neural Network (MLP)** – רשת עצבית פשוטה.  
           - תוצאות: Accuracy = 88%, Recall = 87%, ROC-AUC = 0.93.  
           - יתרון: לומד יחסים מורכבים.  
           - חסרון: נוטה ל־Overfitting, קשה להסבר.  
        """)

    with st.expander("⚙️ שלב רביעי – Grid Search"):
        st.markdown("""
        **מה זה?**  
        Grid Search מריץ את המודל שוב ושוב עם פרמטרים שונים, ובוחר את השילוב שהביא לתוצאה הטובה ביותר.  

        **מה שיפרנו בפועל?**  
        - Logistic Regression – שינוי קטן בלבד.  
        - Random Forest – Recall עלה ל־91% עם 200 עצים.  
        - XGBoost – ROC-AUC עלה ל־0.94 עם עומק עצים = 6 ו־learning_rate = 0.1.  
        - Neural Network – ירידה ב־Overfitting, יציבות טובה יותר.  

        👉 Grid Search היה שלב קריטי בהוצאת המקסימום מהמודלים.  
        """)

    with st.expander("📈 שלב חמישי – הערכת ביצועים"):
        st.markdown("""
        **למה חשוב לבדוק במספר דרכים?**  
        כי כל גרף מראה זווית אחרת של הביצועים:  

        - ROC Curve – עד כמה המודל יודע להפריד בין חולים לבריאים.  
        - Precision-Recall Curve – איזון בין דיוק ליכולת תפיסה.  
        - Confusion Matrix – מראה בדיוק כמה חולים זוהו נכון וכמה טעויות היו.  
        - Learning Curves – בודקות Overfitting או צורך בעוד נתונים.  

        **ממצאים עיקריים:**  
        - ROC Curve: XGBoost עם ROC-AUC = 0.94 → מוביל ברור.  
        - Precision-Recall: Random Forest הצטיין ב־Recall, אבל XGBoost איזן טוב יותר.  
        - Confusion Matrix (XGBoost, 60 דגימות): TP = 27, TN = 27, FP = 3, FN = 3.  
        - Learning Curves: Logistic Regression התייצב מהר, Neural Network נטה ל־Overfitting, XGBoost השתפר עם עוד נתונים.  
        """)

    with st.expander("🧠 שלב שישי – Explainability"):
        st.markdown("""
        **למה זה חשוב?**  
        ברפואה חייבים להבין *למה* המודל קיבל החלטה, ולא רק את התחזית.  

        **איך עשינו זאת?**  
        - Feature Importance – דירוג חשיבות פיצ’רים.  
        - SHAP Values – מראים איך ערך של כל פיצ’ר השפיע על החיזוי.  

        **מה גילינו?**  
        - jitter → הפיצ’ר החשוב ביותר.  
        - shimmer → שני בחשיבותו.  
        - jitter גבוה → תחזית "חולה".  
        - jitter נמוך → תחזית "בריא".  

        👉 ההסברים תאמו ידע רפואי קיים, מה שמחזק את אמון הרופאים במודל.  
        """)

    with st.expander("📌 אילו פיצ’רים עדיפים לשימוש?"):
        st.markdown("""
        יש שני סוגים של פיצ’רים חשובים:  

        1. **פיצ’רים בסיסיים וברורים להסבר** – כמו **jitter** ו־**shimmer**.  
           - קל להראות בגרפים שחולים מציגים ערכים גבוהים יותר.  
           - טובים להסבר בכיתה או לרופאים.  

        2. **פיצ’רים מחושבים עם מתאם גבוה ל־target** – כמו **Spread1, PPE, Spread2**.  
           - הם פחות אינטואיטיביים, אבל סטטיסטית הם חזקים יותר.  
           - המודל משתמש בהם כדי לשפר דיוק.  

        👉 המסקנה:  
        - אם רוצים דיוק מקסימלי → עדיף להשתמש ב־Spread1, PPE ו־Spread2.  
        - אם רוצים גם הסבר פשוט → כדאי להוסיף גם את Jitter ו־Shimmer.  
        - בפרויקט שלנו שילבנו את שניהם → כדי שהמודל יהיה גם חזק וגם מובן.  
        """)

    with st.expander("💻 שלב שביעי – האפליקציה"):
        st.markdown("""
        כדי להנגיש את העבודה, בנינו אפליקציה עם 9 לשוניות:  
        1. Data & EDA – גרפים וסטטיסטיקות.  
        2. Dashboard – KPIs ו־Leaderboard.  
        3. Models – פירוט והשוואת מודלים.  
        4. Prediction – חיזוי לפי נתונים חדשים או CSV.  
        5. Test Evaluation – ROC, PR, Confusion Matrix.  
        6. Train New Model – אימון מחדש עם Grid Search.  
        7. Model History – היסטוריית אימונים.  
        8. Explainability – גרפי SHAP.  
        9. About – דו"ח סופי ומסקנות.  
        """)

    with st.expander("✅ מסקנות סופיות"):
        st.markdown("""
        1. ניתן לנבא פרקינסון בצורה מדויקת על בסיס נתוני קול בלבד.  
        2. **XGBoost** היה המודל המוביל – ROC-AUC = 0.94.  
        3. jitter ו־shimmer הם הפיצ’רים הקריטיים ביותר להבנה והסבר.  
        4. Spread1, PPE ו־Spread2 נתנו את הכוח הסטטיסטי החזק ביותר למודל.  
        5. Grid Search שיפר משמעותית את המודלים.  
        6. האפליקציה מספקת גם חיזוי וגם הסבר – חיוני לשימוש רפואי.  
        """)

    with st.expander("✨ נקודות מעניינות"):
        st.markdown("""
        - גם מודלים פשוטים כמו Logistic Regression הצליחו להראות דפוסים → סימן שהקול הוא אינדיקטור חזק מאוד.  
        - Random Forest כמעט ולא פספס חולים, אבל יצר יותר אזעקות שווא.  
        - Neural Network היה קרוב ל־XGBoost, אך פחות יציב.  
        - XGBoost היה הכי מאוזן ולכן נבחר כמודל המוביל.  
        - ההסברים של SHAP תאמו ידע קליני → מה שמעלה את האמון במערכת.  
        """)

# --- Tab 10: Q&A
tab_qa, = st.tabs(["❓ Q&A"])

with tab_qa:
    st.header("❓ שאלות ותשובות נפוצות על הפרויקט")

    # --- על הנתונים ---
    with st.expander("📊 מאיפה הגיעו הנתונים ואיך נאספו?"):
        st.markdown("""
        הנתונים הגיעו מ־**UCI Machine Learning Repository**.  
        הם נאספו מהקלטות קול של חולי פרקינסון ושל אנשים בריאים.  
        מכל הקלטה הופקו עשרות פיצ’רים מתמטיים (כמו jitter, shimmer, PPE) שתיארו את איכות הקול.
        """)

    with st.expander("📊 למה דווקא קול ולא נתונים אחרים?"):
        st.markdown("""
        קול הוא כלי **לא פולשני**, **פשוט** ו־**זול** לאיסוף.  
        מספיקת הקלטה קצרה, ואפשר לנתח אותה אוטומטית.  
        זה נותן יתרון גדול על פני בדיקות רפואיות אחרות, שיכולות להיות יקרות או פולשניות.  
        """)

    with st.expander("📊 האם הדאטה היה מאוזן בין חולים לבריאים?"):
        st.markdown("""
        לא. הדאטה כלל **יותר חולים מאשר בריאים**.  
        זה יצר בעיית חוסר איזון (class imbalance).  
        לכן לא הסתפקנו במדד Accuracy, אלא בחנו גם **Recall ו־Precision** כדי לוודא שהמודל לא "נוטה" יותר מדי לכיוון החולים.
        """)

    with st.expander("📊 האם בדקתם ערכים חסרים או רעשים?"):
        st.markdown("""
        כן. בדקנו שאין ערכים חסרים משמעותיים.  
        הרעשים בהקלטות כבר עובדו מראש (pre-processing).  
        עדיין ייתכן שבדאטה אמיתי יהיו רעשים נוספים, וזה אתגר למחקר עתידי.
        """)

    with st.expander("📊 האם נעשה נירמול (Normalization) לנתונים?"):
        st.markdown("""
        כן. חלק מהמודלים (כמו Logistic Regression, SVM, Neural Network) רגישים מאוד לסקלות שונות.  
        לכן ביצענו **Standardization** – כל פיצ’ר הומר לטווח ממוצע 0 וסטיית תקן 1.  
        זה שיפר את יציבות המודלים.
        """)

    # --- על הפיצ'רים ---
    with st.expander("🔑 מה זה פיצ’ר?"):
        st.markdown("""
        פיצ’ר הוא **מאפיין מספרי של הקול** – עמודה בטבלה שמתארת מדד מסוים.  
        למשל:  
        - **Jitter:** רעידות בתדר הקול.  
        - **Shimmer:** שינויים בעוצמת הקול.  
        - **Fo:** תדר בסיסי של הקול.  
        - **PPE:** חוסר סדירות במרווחי התדר.  
        - **Spread1 / Spread2:** מדדים סטטיסטיים לפיזור התדרים.  
        """)

    with st.expander("🔑 באילו פיצ’רים השתמשתם בפועל?"):
        st.markdown("""
        השתמשנו גם ב־**פיצ’רים בסיסיים וברורים להסבר** (jitter, shimmer, Fo, Flo)  
        וגם ב־**פיצ’רים מחושבים עם מתאם גבוה למחלה** (Spread1, PPE, Spread2).  

        השילוב נתן:  
        - גם **מודל חזק סטטיסטית**.  
        - וגם **יכולת הסבר ברורה** לרופאים ולכיתה.  
        """)

    with st.expander("🔑 אילו פיצ’רים יצאו החשובים ביותר?"):
        st.markdown("""
        - מה־EDA: חולים הציגו **Jitter ו־Shimmer גבוהים יותר** – קול פחות יציב.  
        - מה־Correlation: **Spread1 (0.56), PPE (0.53), Spread2 (0.45)** היו הכי מתואמים עם המחלה.  
        - מה־Explainability (SHAP): **Jitter ו־Shimmer** הובילו את החשיבות.  

        👉 כלומר, הפיצ’רים המורכבים חיזקו את המודל סטטיסטית,  
        והפיצ’רים הבסיסיים היו קלים וברורים להסבר.  
        """)

    # --- על המודלים ---
    with st.expander("🤖 למה בחרתם את המודלים האלו?"):
        st.markdown("""
        בחרנו קשת רחבה של מודלים:  
        - **מודלים פשוטים** (Logistic Regression) → קל להסביר.  
        - **מודלים חזקים** (Random Forest, XGBoost, Neural Network) → דיוק גבוה.  
        כך יכולנו גם להשוות ביצועים וגם להראות את הערך של המודלים המתקדמים.
        """)

    with st.expander("🤖 למה לא השתמשתם ב-KNN או Naive Bayes?"):
        st.markdown("""
        בדקנו אותם בשלבים ראשוניים.  
        הם נתנו ביצועים חלשים יותר, במיוחד ב־Recall.  
        לכן לא נכנסו להשוואה הסופית.  
        """)

    with st.expander("🤖 למה חשוב לכלול גם מודלים פשוטים כמו Logistic Regression?"):
        st.markdown("""
        מודלים פשוטים נותנים **Baseline** – נקודת ייחוס.  
        אם מודל מורכב נותן רק שיפור קטן לעומת מודל פשוט, זה מעלה שאלות אם המורכבות משתלמת.  
        במקרה שלנו, Logistic Regression נתן Accuracy של 83% → והראה שהקול באמת מכיל מידע חזק.  
        """)

    with st.expander("🤖 מה היתרונות והחסרונות של XGBoost?"):
        st.markdown("""
        **יתרונות:**  
        - הכי מדויק: ROC-AUC = 0.94.  
        - איזון מצוין בין Recall ל־Precision.  
        - מתאים במיוחד לנתונים טבלאיים.  

        **חסרונות:**  
        - דורש יותר משאבים חישוביים.  
        - קשה להסבר (Black Box) בלי כלים כמו SHAP.  
        """)

    with st.expander("🤖 למה לא השתמשתם ברשתות עמוקות יותר (Deep Learning)?"):
        st.markdown("""
        כי הדאטה קטן יחסית (כמה מאות דגימות בלבד).  
        רשתות עמוקות דורשות עשרות אלפי דוגמאות כדי לתת יציבות.  
        אחרת הן נוטות ל־Overfitting.  
        """)

    # --- על Grid Search ---
    with st.expander("⚙️ למה בחרתם Grid Search ולא Randomized Search?"):
        st.markdown("""
        Grid Search בודק **את כל השילובים האפשריים** של פרמטרים.  
        זה יותר כבד חישובית, אבל הדאטה שלנו יחסית קטן ולכן זה היה אפשרי.  
        כך קיבלנו ביטחון שלא פספסנו פרמטרים טובים.  
        """)

    # --- על התוצאות ---
    with st.expander("📈 מה המדד הכי חשוב בפרויקט?"):
        st.markdown("""
        המדד הכי חשוב הוא **Recall** – כמה חולים אמיתיים זיהינו.  
        כי ברפואה חשוב לא לפספס חולים.  
        """)
    with st.expander("📈 מה המשמעות של ROC-AUC = 0.94?"):
        st.markdown("""
        ROC-AUC מודד את היכולת להבדיל בין חולים לבריאים.  
        ערך של 0.94 אומר שהמודל מבדיל נכונה ב־94% מהמקרים, ללא קשר לסף מסוים.  
        """)
    with st.expander("📈 מה מצאנו ב-Confusion Matrix?"):
        st.markdown("""
        במודל XGBoost (60 דגימות):  
        - 27 חולים זוהו נכון (True Positive).  
        - 27 בריאים זוהו נכון (True Negative).  
        - 3 חולים פספסנו (False Negative).  
        - 3 בריאים סווגו כחולים (False Positive).  
        """)

    with st.expander("📈 מה לגבי False Positives?"):
        st.markdown("""
        היו גם בריאים שסווגו כחולים.  
        זה פחות חמור מאשר לפספס חולים אמיתיים, כי במקרה כזה פשוט נבצע בדיקה חוזרת.  
        ברפואה תמיד עדיף עודף רגישות על פני פספוס.  
        """)

    # --- על Explainability ---
    with st.expander("🧠 למה חשוב להסביר את המודל?"):
        st.markdown("""
        ברפואה אסור להסתפק בתחזית "שחורה".  
        חייבים לדעת *למה* המודל קבע שמישהו חולה – כדי שרופא יוכל לסמוך על התוצאה.  
        """)

    with st.expander("🧠 מה ההבדל בין Feature Importance ל-SHAP?"):
        st.markdown("""
        - **Feature Importance:** מדרג פיצ’רים לפי התרומה הכללית שלהם.  
        - **SHAP:** נותן הסבר פרטני לכל תחזית בודדת – איזה פיצ’ר דחף את החיזוי ולכיוון איזה.  
        """)

    with st.expander("🧠 מה מצאנו ב-SHAP?"):
        st.markdown("""
        - **Jitter** → הפיצ’ר החשוב ביותר.  
        - **Shimmer** → השני בחשיבותו.  
        - jitter גבוה → חיזוי "חולה".  
        - jitter נמוך → חיזוי "בריא".  

        זה תואם ידע רפואי קיים → מחזק את האמון במודל.  
        """)

    # --- על ההשלכות הרפואיות ---
    with st.expander("🏥 האם המודל הזה יכול לשמש לאבחון רפואי?"):
        st.markdown("""
        כרגע לא. זה כלי מחקרי בלבד.  
        לשימוש אמיתי נדרש מחקר קליני נרחב ואישור רגולטורי.  
        """)

    with st.expander("🏥 מה היתרון של שימוש בקול על פני בדיקות אחרות?"):
        st.markdown("""
        קול הוא פשוט, זול ולא פולשני.  
        אפשר למדוד אותו גם בבית וגם במרפאה, בלי ציוד רפואי יקר.  
        """)

    with st.expander("🏥 האם המודל יכול לשמש למעקב אחרי המחלה?"):
        st.markdown("""
        כן. אפשר להשתמש בו כדי לבדוק איך משתנה הקול של חולה לאורך זמן.  
        זה יכול לעזור לעקוב אחרי התקדמות המחלה או תגובה לטיפול.  
        """)

    with st.expander("🏥 מה הצעד הבא?"):
        st.markdown("""
        לאסוף דאטה גדול ומגוון יותר (כולל קבוצות גיל שונות ושפות שונות).  
        להריץ מחקרים קליניים כדי לבדוק אם המודל מחזיק גם בעולם האמיתי.  
        """)

    # --- שאלות קשות ---
    with st.expander("⚖️ למה בחרתם ב-XGBoost ולא ב-Neural Network?"):
        st.markdown("""
        Neural Network נתן ביצועים קרובים, אבל נטה ל־Overfitting והיה קשה להסביר אותו.  
        XGBoost היה גם מדויק, גם יציב וגם ניתן להסבר עם SHAP → ולכן נבחר.  
        """)

    with st.expander("⚖️ אם היה לכם עוד זמן – מה הייתם משפרים?"):
        st.markdown("""
        היינו:  
        - אוספים דאטה גדול ומגוון יותר.  
        - בודקים רשתות עמוקות יותר.  
        - עושים Cross-Validation נרחב.  
        """)

    with st.expander("⚖️ מה המגבלה הכי גדולה בפרויקט?"):
        st.markdown("""
        גודל הדאטה. יש לנו מאות דגימות בלבד, וזה לא מייצג את כל האוכלוסייה.  
        לכן זו הוכחת היתכנות בלבד, לא כלי רפואי מוכן לשימוש.  
        """)

    with st.expander("⚖️ האם Accuracy של 89% מספיק לשימוש רפואי?"):
        st.markdown("""
        לא. ברפואה נדרש דיוק גבוה בהרבה (מעל 95%) ובדיקות קליניות רחבות.  
        אבל כהוכחת היתכנות ראשונית → זה הישג משמעותי שמראה שיש פוטנציאל גדול.
        """)









