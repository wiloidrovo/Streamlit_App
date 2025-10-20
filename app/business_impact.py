# app/business_impact.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def _models_dir():
    return os.path.join(os.path.dirname(__file__), "..", "models")

def list_bundles():
    md = _models_dir()
    return sorted([f for f in os.listdir(md) if f.endswith(".pkl") and not f.startswith("model_metrics")])

def load_bundle(model_name: str, version: str):
    path = os.path.join(_models_dir(), f"{model_name.lower()}_{version.lower()}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def business_impact_page(df_clean: pd.DataFrame):
    st.title("💼 Business Impact")

    if df_clean is None or df_clean.empty:
        st.warning("Please prepare/load a dataset first (EDA tab).")
        return

    bundles = list_bundles()
    if not bundles:
        st.error("No trained models found in `models/`.")
        return

    col1, col2 = st.columns([2,1])
    with col1:
        model_names = sorted(set(x.split("_")[0].upper() for x in bundles))
        mdl = st.selectbox("Prediction model:", model_names, index=0)
    with col2:
        version = st.radio("Version:", ["ALL", "TOP"], horizontal=True, index=0)

    bundle = load_bundle(mdl, version)
    if bundle is None:
        st.error("Model bundle not found.")
        return

    pipe = bundle["pipeline"]
    target = bundle.get("target_name", "Churn")
    if target not in df_clean.columns:
        st.info("No ground-truth `Churn` column found; only predictions will be shown.")

    # KPIs de negocio: supuestos
    st.markdown("### 💵 Business assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        avg_revenue = st.number_input("Average Monthly Charges (ARPU)", min_value=0.0, value=float(df_clean.get("MonthlyCharges", pd.Series([70])).median()))
    with c2:
        retention_cost = st.number_input("Retention Campaign Cost per user", min_value=0.0, value=10.0)
    with c3:
        retention_effect = st.slider("Retention effectiveness (%)", 0, 100, 30, help="% of at-risk users that you expect to retain with the campaign.")

    # Predicción de churn prob para todos los registros
    with st.spinner("Scoring current dataset…"):
        X = df_clean.drop(columns=[target]) if target in df_clean.columns else df_clean.copy()
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[:, 1]
        else:
            scores = pipe.decision_function(X)
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        df_scores = df_clean.copy()
        df_scores["churn_proba"] = proba

    # % clientes en riesgo (umbral configurable)
    st.markdown("### ⚠️ Risk segmentation")
    thr = st.slider("Churn probability threshold (%)", 0, 100, 70)
    risky = df_scores["churn_proba"] >= (thr/100)

    risk_rate = risky.mean() * 100
    at_risk = risky.sum()

    # Revenue at risk (aprox)
    revenue_col = "MonthlyCharges" if "MonthlyCharges" in df_scores.columns else None
    if revenue_col:
        revenue_at_risk = df_scores.loc[risky, revenue_col].sum()
        avg_arpu = df_scores[revenue_col].mean()
    else:
        revenue_at_risk = at_risk * avg_revenue
        avg_arpu = avg_revenue

    # Potenciales ahorros si se retiene X% de los de riesgo
    retained = int(at_risk * (retention_effect/100))
    potential_savings = retained * avg_arpu - retained * retention_cost

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("% at risk", f"{risk_rate:.2f}%")
    m2.metric("# at risk", f"{at_risk}")
    m3.metric("Revenue at risk (≈/mo)", f"${revenue_at_risk:,.2f}")
    m4.metric("Potential savings (≈/mo)", f"${potential_savings:,.2f}")

    st.markdown("---")

    # Distribución de probabilidades
    st.subheader("📉 Churn probability distribution")
    fig_hist = px.histogram(df_scores, x="churn_proba", nbins=40, color=risky.map({True:"At risk", False:"Safe"}),
                            color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("⭐ Key drivers (feature importance)")
    importances = bundle.get("feature_importances", [])
    selected = bundle.get("selected_features", None)
    if importances and (selected or not selected):
        if selected:
            feat_names = selected
        else:
            try:
                pre = bundle["pipeline"].named_steps["preprocessing"]
                feat_names = list(pre._columns) if hasattr(pre, "_columns") else [f"F{i}" for i,_ in enumerate(importances)]
            except Exception:
                feat_names = [f"F{i}" for i,_ in enumerate(importances)]

        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
        st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h"), use_container_width=True)
    else:
        st.info("This model doesn't expose feature importances.")

    st.markdown("---")

    st.subheader("🤖 Automatic recommendation (per-customer)")
    sample_n = st.slider("Show top-N at-risk customers", 5, 50, 10)
    top_at_risk = df_scores.sort_values("churn_proba", ascending=False).head(sample_n).copy()
    top_at_risk["recommendation"] = np.where(
        top_at_risk["churn_proba"] >= 0.7,
        "High risk → Offer discount or add-on service",
        np.where(top_at_risk["churn_proba"] >= 0.4,
                 "Medium risk → Loyalty offers / follow-up",
                 "Low risk → Normal monitoring")
    )
    cols_to_show = ["churn_proba"]
    if revenue_col:
        cols_to_show = [revenue_col] + cols_to_show
    st.dataframe(top_at_risk[cols_to_show + ["recommendation"]].style.format({"churn_proba": "{:.2%}"}))
