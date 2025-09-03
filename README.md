# Parkinson's ML Project – Final Version

## 📌 Overview
End-to-end machine learning pipeline to predict Parkinson's disease from voice features.

## 📂 Contents
- `eda_pipeline.py` → Full EDA analysis with plots and stats
- `model_pipeline.py` → Training pipeline with GridSearchCV (RandomForest + XGBoost)
- `Parkinsons_Project_Full_GridSearch.ipynb` → Full Colab notebook (EDA + Models + GridSearch + Plots + SHAP + ZIP)
- `app/streamlit_app.py` → Streamlit application placeholder
- `requirements.txt` → Python dependencies
- `README.md` → Documentation

## 🚀 How to Run
```bash
pip install -r requirements.txt
python eda_pipeline.py
python model_pipeline.py
streamlit run app/streamlit_app.py
```

## 🎯 Features
- EDA: stats, heatmaps, PCA, t-SNE, distributions
- Models: LogisticRegression, RandomForest, SVM, KNN, XGBoost, LightGBM, CatBoost, NeuralNet
- Hyperparameter tuning with GridSearchCV (RF + XGBoost)
- Evaluation: Confusion Matrix, ROC, PR Curves, Learning Curve, SHAP
- ZIP export with all results
