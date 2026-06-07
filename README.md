# Black Friday High Spender Classifier

A machine learning app that predicts whether a retail customer is a **High Spender** based on their transaction and demographic features.

## Dataset

**Source:** [Black Friday Sales Dataset — Kaggle](https://www.kaggle.com/)  
**License:** CC0: Public Domain  
**Size:** 100,000 retail transactions captured during Black Friday sales

## Model

- **Task:** Binary classification (`high_spender`: 0 = Not High Spender, 1 = High Spender)
- **Target:** `purchase_amount` > median (€161.15) → High Spender
- **Winner:** Decision Tree (max_depth=3) — 98.25% accuracy
- **Key predictors:** `final_price`, `quantity`

## App Tabs

| Tab | Description |
|-----|-------------|
| **EDA** | Descriptive statistics and feature distributions |
| **Model Card** | Comparison of all trained models |
| **Predict** | Enter transaction details and get a prediction |

## Built With

- scikit-learn, pandas, Gradio, Plotly, joblib

---
*MSc in Business Information Systems & Analytics — Python for Data Science, ML and AI*
