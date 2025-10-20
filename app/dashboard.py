import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Helpers
def _models_dir():
    return os.path.join(os.path.dirname(__file__), "..", "models")

def load_metrics():
    path = os.path.join(_models_dir(), "model_metrics_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalizar nombres
    df["Model"] = df["Model"].str.upper()
    df["Version"] = df["Version"].str.upper()
    return df

def list_bundles():
    md = _models_dir()
    return sorted([f for f in os.listdir(md) if f.endswith(".pkl") and not f.startswith("model_metrics")])

def load_bundle(model_name: str, version: str):
    path = os.path.join(_models_dir(), f"{model_name.lower()}_{version.lower()}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def fig_confusion(cm, labels=("No", "Yes"), title="Confusion Matrix"):
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
        text=z, texttemplate="%{text}", hoverinfo="skip", colorscale="Blues"
    ))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    return fig

# Page
def dashboard_page(df_clean=None):
    st.title("📈 Model Dashboard")

    # Panel de métricas globales (Val set)
    metrics = load_metrics()
    if metrics is None or metrics.empty:
        st.warning("No metrics file found in `models/model_metrics_summary.csv`. Train & export first.")
        return

    # Selectores
    left, right = st.columns([2, 1])
    with left:
        models = sorted(metrics["Model"].unique().tolist())
        selected_models = st.multiselect("Select models to compare:", models, default=models)
    with right:
        metric_to_show = st.selectbox("Metric:", ["f1", "auc", "accuracy", "precision", "recall"], index=0)

    df_show = metrics[metrics["Model"].isin(selected_models)].copy()
    if df_show.empty:
        st.info("No models selected.")
        return

    # Gráfico comparativo (All vs Top)
    st.subheader("📊 Metrics comparison (Validation)")
    bar = px.bar(
        df_show, x="Model", y=metric_to_show, color="Version",
        barmode="group", text=metric_to_show, range_y=[0,1],
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    bar.update_layout(yaxis_title=metric_to_show.upper())
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("---")

    # Sección por modelo: confusion matrix + feature importance
    st.subheader("🔍 Drill-down: Confusion Matrix & Feature Importance")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        mdl = st.selectbox("Model:", models, index=0)
    with c2:
        version = st.radio("Version:", ["ALL", "TOP"], horizontal=True, index=0)
    with c3:
        show_imp = st.checkbox("Show Feature Importance", value=True)

    bundle = load_bundle(mdl, version)
    if bundle is None:
        st.error("Bundle not found.")
        return

    # Confusion matrix (val)
    cm = bundle.get("confusion_val", None)
    if cm is not None:
        st.plotly_chart(fig_confusion(cm, labels=("No","Yes"), title=f"{mdl} - {version} (Validation)"), use_container_width=True)
    else:
        st.info("Confusion matrix not stored in bundle.")

    # Feature importance
    if show_imp:
        importances = bundle.get("feature_importances", [])
        # Si el pipeline es 'top', en bundle guardamos selected_features
        selected = bundle.get("selected_features", None)
        if importances and (selected or not selected):
            # nombres de features de salida
            if selected:
                feat_names = selected
            else:
                # Para 'all', intentamos obtener nombres del DataFramePreparer+OHE
                # Recuperar columnas desde el paso "preprocessing"
                try:
                    pre = bundle["pipeline"].named_steps["preprocessing"]
                    # Intentar atributo _columns (definido en DataFramePreparer)
                    feat_names = list(pre._columns) if hasattr(pre, "_columns") else [f"F{i}" for i,_ in enumerate(importances)]
                except Exception:
                    feat_names = [f"F{i}" for i,_ in enumerate(importances)]

            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("This model does not expose feature importances or they were not saved.")