import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils import apply_style

# =====================================================
# AUTOMATIC ANALYSIS VS TARGET
# =====================================================
def analizar_vs_target(df, target_col):
    st.subheader(f"🎯 Automatic Analysis with respect to Target Variable: `{target_col}`")

    if target_col not in df.columns:
        st.warning("⚠️ The target variable is not in the dataset.")
        return

    n_clases = df[target_col].nunique()
    if n_clases > 20:
        st.error(f"🚫 The target variable has {n_clases} unique classes — too many for automatic analysis.")
        return

    for col in df.columns:
        if col == target_col:
            continue

        st.markdown(f"### 📊 {col} vs {target_col}")

        # =====================================================
        # NUMÉRICAS
        # =====================================================
        if pd.api.types.is_numeric_dtype(df[col]):
            # Boxplot
            fig_box = px.box(
                df,
                x=target_col,
                y=col,
                color=target_col,
                title=f"Distribution of {col} by {target_col}",
                color_discrete_sequence=px.colors.sequential.Teal,
            )
            fig_box.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#fafafa", family="Inter"),
                title=dict(
                    text=fig_box.layout.title.text,
                    x=0,
                    font=dict(size=18, color="#00bcd4")
                ),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Histograma
            fig_hist = px.histogram(
                df,
                x=col,
                color=target_col,
                barmode="overlay",
                opacity=0.7,
                title=f"Histogram of {col} grouped by {target_col}",
                color_discrete_sequence=px.colors.sequential.Teal,
            )
            fig_hist.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#fafafa", family="Inter"),
                title=dict(
                    text=fig_hist.layout.title.text,
                    x=0,
                    font=dict(size=18, color="#00bcd4")
                ),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # =====================================================
        # CATEGÓRICAS
        # =====================================================
        else:
            n_unicos = df[col].nunique()
            cross_tab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
            cross_tab = cross_tab.reset_index().melt(
                id_vars=col, var_name=target_col, value_name="Percentage"
            )

            if n_unicos == 2:
                # Binarias → barras agrupadas con texto dentro
                fig = px.bar(
                    cross_tab,
                    x=col,
                    y="Percentage",
                    color=target_col,
                    text="Percentage",
                    barmode="group",
                    title=f"Percentage distribution of {col} by {target_col}",
                    color_discrete_sequence=px.colors.sequential.Teal,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
            else:
                # Categóricas → barras apiladas
                fig = px.bar(
                    cross_tab,
                    x=col,
                    y="Percentage",
                    color=target_col,
                    barmode="stack",
                    title=f"Percentage distribution of {col} by {target_col}",
                    color_discrete_sequence=px.colors.sequential.Teal,
                )

            fig.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#fafafa", family="Inter"),
                title=dict(
                    text=fig.layout.title.text,
                    x=0,
                    font=dict(size=18, color="#00bcd4")
                ),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla de distribución
            st.markdown("##### 📋 Distribution Table (% by row)")
            tabla = pd.crosstab(df[col], df[target_col], normalize="index") * 100
            st.dataframe(tabla.round(2), use_container_width=True)

        st.markdown("<hr style='border: 1px solid #1f2937; margin: 40px 0;'>", unsafe_allow_html=True)

# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def ejecutar_eda_target(df):
    apply_style()

    target_col = st.session_state.get("target_col", None)

    if not target_col:
        st.info("ℹ️ A target variable has not yet been defined.")
        return

    analizar_vs_target(df, target_col)