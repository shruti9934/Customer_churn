"""Streamlit app: predict churn for one customer or a CSV of customers.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from churn.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH
from churn.data import clean
from churn.explain import explain_customer, predict_proba, risk_band

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m churn.train` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


artifact = load_artifact()
metrics = load_metrics()
threshold = artifact["threshold"]

st.title("📉 Customer Churn Predictor")
st.caption(f"Model: **{artifact['model_name']}** · decision threshold {threshold:.2f} "
           "(tuned to maximise F1 on cross-validated predictions)")

tab_single, tab_batch, tab_model = st.tabs(["Single customer", "Batch (CSV)", "Model performance"])

# --------------------------------------------------------------------------- single customer
with tab_single:
    yes_no = ["No", "Yes"]
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless billing", yes_no, index=1)
            monthly = st.number_input("Monthly charges", 18.0, 120.0, 85.0, step=0.5)
            total = st.number_input("Total charges", 0.0, 9000.0, float(round(85.0 * 5, 2)), step=10.0)
        with c2:
            st.subheader("Customer")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", yes_no)
            partner = st.selectbox("Partner", yes_no)
            dependents = st.selectbox("Dependents", yes_no)
            phone = st.selectbox("Phone service", yes_no, index=1)
            lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        with c3:
            st.subheader("Services")
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            addon_opts = ["No", "Yes", "No internet service"]
            security = st.selectbox("Online security", addon_opts)
            backup = st.selectbox("Online backup", addon_opts)
            protection = st.selectbox("Device protection", addon_opts)
            support = st.selectbox("Tech support", addon_opts)
            tv = st.selectbox("Streaming TV", addon_opts)
            movies = st.selectbox("Streaming movies", addon_opts)
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
            "SeniorCitizen": int(senior == "Yes"), "gender": gender, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }])
        prob = float(predict_proba(artifact, customer).iloc[0])
        band = risk_band(prob, threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Risk", band)
        m3.metric("Prediction", "Will churn" if prob >= threshold else "Will stay")

        with st.spinner("Working out what drives this prediction..."):
            drivers = explain_customer(artifact, customer)
        top = drivers.head(8).copy()
        top["value"] = top["value"].astype(str)
        top["effect"] = top["contribution"].map(lambda c: f"{c:+.1%}")
        top["direction"] = top["contribution"].map(lambda c: "raises risk" if c > 0 else "lowers risk")
        st.subheader("Key drivers for this customer")
        st.caption("How much each input moves the churn probability compared with a typical customer.")
        st.bar_chart(top.set_index("feature")["contribution"], horizontal=True)
        st.dataframe(top[["feature", "value", "effect", "direction"]], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- batch scoring
with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset "
             "(a `Churn` column is optional and ignored).")
    file = st.file_uploader("Customer CSV", type="csv")
    if file is not None:
        df = clean(pd.read_csv(file))
        missing = [c for c in artifact["features"] if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df["churn_probability"] = predict_proba(artifact, df).round(4)
            df["risk"] = df["churn_probability"].map(lambda p: risk_band(p, threshold))
            df = df.sort_values("churn_probability", ascending=False)
            st.metric("Customers flagged as likely to churn", int((df["churn_probability"] >= threshold).sum()))
            st.dataframe(df, width="stretch")
            st.download_button("Download scored CSV", df.to_csv(index=False), "scored_customers.csv", "text/csv")

# --------------------------------------------------------------------------- model performance
with tab_model:
    if metrics is None:
        st.info("Run `python -m churn.train` to generate metrics.")
    else:
        bm = metrics["best_model_test_metrics"]
        cols = st.columns(5)
        for col, key in zip(cols, ["roc_auc", "f1", "precision", "recall", "accuracy"]):
            col.metric(key.replace("_", "-").upper(), f"{bm[key]:.3f}")
        st.subheader("Model comparison (held-out test set, threshold 0.5)")
        table = pd.DataFrame(metrics["all_models"]).T[["cv_roc_auc", "roc_auc", "f1", "precision",
                                                       "recall", "accuracy", "train_seconds"]]
        st.dataframe(table, width="stretch")
        imgs = [p for p in ["roc_curves.png", "confusion_matrix.png", "feature_importance.png",
                            "churn_by_category.png", "numeric_distributions.png"]
                if (FIGURES_DIR / p).exists()]
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for col, img in zip(cols, imgs[i:i + 2]):
                col.image(str(FIGURES_DIR / img), width="stretch")
