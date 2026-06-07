"""
Black Friday High Spender Classifier — Gradio App
==================================================
Serves the pre-trained model from model.joblib.
NO .fit() calls. NO sklearn imports. Training lives in train_and_save_model.ipynb.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# ── Load artifacts (once at startup) ─────────────────────────────────────────
artifact     = joblib.load(SCRIPT_DIR / "model.joblib")
PIPELINES    = artifact["pipelines"]
WINNER       = artifact["winner"]
FEATURES     = artifact["feature_names"]
POS_LABEL    = artifact["positive_label"]
NEG_LABEL    = artifact["negative_label"]
THRESHOLD    = artifact["threshold"]

df_sample    = pd.read_csv(SCRIPT_DIR / "classification_sample.csv")
df_compare   = pd.read_csv(SCRIPT_DIR / "model_comparison.csv")

ALGO_CHOICES = list(PIPELINES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
def eda_histogram(feature):
    fig = px.histogram(
        df_sample, x=feature, color="high_spender",
        barmode="overlay",
        color_discrete_map={0: "purple", 1: "hotpink"},
        labels={"high_spender": "High Spender"},
        title=f"Distribution of {feature} by High Spender",
    )
    fig.update_layout(legend_title_text="High Spender (0=No, 1=Yes)")
    return fig

numeric_features = ["final_price", "discount_pct", "quantity",
                    "purchase_amount", "purchase_hour"]

with gr.Blocks() as tab_eda:
    gr.Markdown("## Exploratory Data Analysis\nBased on a stratified 2,000-row sample of the Black Friday Sales dataset.")
    gr.Markdown(f"**Dataset shape:** {df_sample.shape[0]} rows × {df_sample.shape[1]} columns")
    gr.DataFrame(df_sample.describe().round(2).reset_index(), label="Descriptive Statistics")
    gr.Markdown("### Feature vs High Spender")
    feat_dd  = gr.Dropdown(choices=numeric_features, value="final_price", label="Select Feature")
    hist_plt = gr.Plot()
    feat_dd.change(eda_histogram, inputs=feat_dd, outputs=hist_plt)
    tab_eda.load(lambda: eda_histogram("final_price"), outputs=hist_plt)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Card
# ─────────────────────────────────────────────────────────────────────────────
def make_comparison_chart():
    fig = px.bar(
        df_compare.sort_values("Accuracy"),
        x="Accuracy", y="Algorithm",
        orientation="h",
        color="Winner",
        color_discrete_map={True: "purple", False: "lightgray"},
        title="Model Comparison — Accuracy on Test Set",
        range_x=[0.90, 1.01],
        text="Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig

with gr.Blocks() as tab_model:
    gr.Markdown("## Model Card")
    gr.Markdown(f"""
**Dataset:** Black Friday Sales Data (Kaggle) — 100,000 transactions  
**Task:** Binary classification — predict whether a customer is a *High Spender*  
**Target:** `high_spender` = 1 if `purchase_amount` > median (€{THRESHOLD:.2f}), else 0  
**Class balance:** 50 / 50 (stratified split)

### Winner: `{WINNER}`
Selected over Decision Tree (depth=10) which showed signs of **overfitting** (99.99% accuracy),
and over Logistic Regression (96.76%) which was outperformed.
The Decision Tree (depth=3) achieves **98.25% accuracy** while remaining interpretable.

**Key predictors identified in EDA:** `final_price` and `quantity`
""")
    gr.Plot(make_comparison_chart())
    gr.DataFrame(df_compare[["Algorithm", "Accuracy", "Winner"]], label="Full Comparison")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────
# Feature ranges from sample
feature_ranges = {
    "age_group"       : (0, 4, 2),
    "gender"          : (0, 2, 1),
    "city"            : (0, 9, 4),
    "customer_segment": (0, 3, 1),
    "product_category": (0, 9, 4),
    "discount_pct"    : (5, 60, 25),
    "final_price"     : (2, 2400, 150),
    "quantity"        : (1, 5, 1),
    "payment_method"  : (0, 5, 2),
    "is_weekend"      : (0, 1, 0),
    "is_black_friday" : (0, 1, 0),
}

def predict(algo, age_group, gender, city, customer_segment, product_category,
            discount_pct, final_price, quantity, payment_method,
            is_weekend, is_black_friday):
    row = pd.DataFrame([{
        "age_group"       : int(age_group),
        "gender"          : int(gender),
        "city"            : int(city),
        "customer_segment": int(customer_segment),
        "product_category": int(product_category),
        "discount_pct"    : float(discount_pct),
        "final_price"     : float(final_price),
        "quantity"        : int(quantity),
        "payment_method"  : int(payment_method),
        "is_weekend"      : int(is_weekend),
        "is_black_friday" : int(is_black_friday),
    }])[FEATURES]

    pipe  = PIPELINES[algo]
    pred  = pipe.predict(row)[0]
    label = POS_LABEL if pred == 1 else NEG_LABEL
    
    # Estimated purchase amount
    est_purchase = final_price * quantity
    note = f"Estimated purchase amount: €{est_purchase:.2f} (threshold: €{THRESHOLD:.2f})"
    
    return f"**{label}**", note

with gr.Blocks() as tab_predict:
    gr.Markdown("## Predict High Spender\nFill in the customer and transaction details below.")
    algo_dd = gr.Dropdown(choices=ALGO_CHOICES, value=WINNER, label="Algorithm")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Customer Features**")
            age_group        = gr.Slider(*feature_ranges["age_group"],        step=1,   label="Age Group (0=18-25, 1=26-35, 2=36-45, 3=46-55, 4=56+)")
            gender           = gr.Slider(*feature_ranges["gender"],           step=1,   label="Gender (0=Female, 1=Male, 2=Other)")
            city             = gr.Slider(*feature_ranges["city"],             step=1,   label="City (0–9 encoded)")
            customer_segment = gr.Slider(*feature_ranges["customer_segment"], step=1,   label="Customer Segment (0=Loyal, 1=New, 2=Returning, 3=VIP)")
        with gr.Column():
            gr.Markdown("**Transaction Features**")
            product_category = gr.Slider(*feature_ranges["product_category"], step=1,   label="Product Category (0–9 encoded)")
            discount_pct     = gr.Slider(*feature_ranges["discount_pct"],     step=5,   label="Discount %")
            final_price      = gr.Slider(*feature_ranges["final_price"],      step=10,  label="Final Price (€)")
            quantity         = gr.Slider(*feature_ranges["quantity"],         step=1,   label="Quantity")
            payment_method   = gr.Slider(*feature_ranges["payment_method"],   step=1,   label="Payment Method (0–5 encoded)")
            is_weekend       = gr.Slider(*feature_ranges["is_weekend"],       step=1,   label="Is Weekend (0=No, 1=Yes)")
            is_black_friday  = gr.Slider(*feature_ranges["is_black_friday"],  step=1,   label="Is Black Friday (0=No, 1=Yes)")

    btn    = gr.Button("Predict", variant="primary")
    result = gr.Markdown()
    note   = gr.Markdown()

    btn.click(
        predict,
        inputs=[algo_dd, age_group, gender, city, customer_segment,
                product_category, discount_pct, final_price, quantity,
                payment_method, is_weekend, is_black_friday],
        outputs=[result, note],
    )

# ─────────────────────────────────────────────────────────────────────────────
# Assemble app
# ─────────────────────────────────────────────────────────────────────────────
app = gr.TabbedInterface(
    [tab_eda, tab_model, tab_predict],
    ["EDA", "Model Card", "Predict"],
    title="Black Friday High Spender Classifier",
)

if __name__ == "__main__":
    app.launch()
