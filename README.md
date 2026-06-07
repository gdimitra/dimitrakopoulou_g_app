# Black Friday High Spender Classifier

A machine learning app that predicts whether a retail customer is a **High Spender** based on their transaction and demographic features.

## Dataset

**Source:** [Black Friday Sales Dataset — Kaggle](https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset)  
**License:** apache-2.0

**Size:** 100,000 retail transactions captured during Black Friday sales

## Model

- **Task:** Binary classification (`high_spender`: 0 = Not High Spender, 1 = High Spender)
- **Target:** `purchase_amount` > median (€161.15) → High Spender
- **Winner:** Decision Tree (max_depth=3) — 98.25% accuracy
- **Key predictors:** `final_price`, `quantity`

## Built With

- scikit-learn, pandas, Gradio, Plotly, joblib