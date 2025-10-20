import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils import apply_style


# =====================================================
# KEY PERFORMANCE INDICATORS (KPIs)
# =====================================================
def mostrar_kpis(df):
    st.subheader("📈 Key Performance Indicators (KPIs)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col1.metric("Total Columns", df.shape[1])

    total_celdas = df.shape[0] * df.shape[1]
    total_nulos = df.isnull().sum().sum()
    pct_nulos = (total_nulos / total_celdas) * 100
    col2.metric("% of Null Values", f"{pct_nulos:.2f}%")

    # Promedio de una variable numérica elegida
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        col_selec = col3.selectbox("Select numeric column for KPI:", num_cols, key="kpi_col")
        promedio = df[col_selec].mean()
        col3.metric(f"Average of {col_selec}", f"{promedio:.2f}")
    else:
        col3.info("There are no numeric columns in the dataset.")


# =====================================================
# INTERACTIVE CHART GENERATOR
# =====================================================
def generador_graficos(df):
    st.subheader("🧠 Interactive Chart Generator")

    # Inicializar variable target si no existe
    if "target_col" not in st.session_state:
        st.session_state.target_col = None

    options = [None] + list(df.columns)
    default_index = (
        options.index(st.session_state.target_col)
        if st.session_state.target_col in df.columns
        else 0
    )

    # Selector de variable objetivo
    target_col = st.selectbox(
        "Select target variable (optional):",
        options,
        index=default_index,
        key="target_col"
    )

    # =====================================================
    # DISTRIBUCIÓN DEL TARGET
    # =====================================================
    if target_col:
        st.markdown(f"### 🎯 Distribution of Target Variable: `{target_col}`")

        if pd.api.types.is_numeric_dtype(df[target_col]):
            fig = px.histogram(
                df, x=target_col, nbins=20,
                title=f"Distribution of {target_col}",
                color_discrete_sequence=["#00bcd4"]
            )
        else:
            dist = df[target_col].value_counts(normalize=True).reset_index()
            dist.columns = [target_col, "Proportion"]
            dist["Proportion"] *= 100
            fig = px.pie(
                dist,
                names=target_col,
                values="Proportion",
                title=f"Distribution of {target_col}",
                hole=0.3,
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig.update_traces(textinfo="label+percent")

        # Estilo visual
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            title=dict(
                text=fig.layout.title.text,
                x=0,
                font=dict(size=18, color="#00bcd4", family="Inter, sans-serif")
            ),
            margin=dict(t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SELECTOR DE TIPO DE GRÁFICO
    # =====================================================
    tipos = ["Histogram", "Bar Chart", "Boxplot", "Scatterplot", "Heatmap (Correlation)"]
    tipo = st.selectbox("📊 Select Chart Type:", tipos)

    # =====================================================
    # CONSTRUCCIÓN DE GRÁFICOS SEGÚN SELECCIÓN
    # =====================================================
    if tipo == "Histogram":
        col_x = st.selectbox("Variable (X):", df.columns)
        fig = px.histogram(
            df,
            x=col_x,
            color=target_col if target_col else None,
            barmode="group",
            color_discrete_sequence=px.colors.sequential.Teal,
            title=f"{col_x} vs {target_col}" if target_col else f"Histogram of {col_x}"
        )

    elif tipo == "Bar Chart":
        col_x = st.selectbox("Categorical Variable (X):", df.columns)
        col_y = st.selectbox("Numeric Variable (Y):", df.select_dtypes(include='number').columns)
        fig = px.bar(
            df,
            x=col_x,
            y=col_y,
            color=target_col if target_col else None,
            barmode="group",
            color_discrete_sequence=px.colors.sequential.Teal,
            title=f"{col_y} by {col_x} grouped by {target_col}" if target_col else f"Bar Chart of {col_y} by {col_x}"
        )

    elif tipo == "Boxplot":
        col_x = st.selectbox("Categorical Variable (X):", df.columns)
        col_y = st.selectbox("Numeric Variable (Y):", df.select_dtypes(include='number').columns)
        fig = px.box(
            df,
            x=col_x,
            y=col_y,
            color=target_col if target_col else None,
            color_discrete_sequence=px.colors.sequential.Teal,
            title=f"{col_y} by {col_x} grouped by {target_col}" if target_col else f"Boxplot of {col_y} by {col_x}"
        )

    elif tipo == "Scatterplot":
        col_x = st.selectbox("Numeric Variable (X):", df.select_dtypes(include='number').columns)
        col_y = st.selectbox("Numeric Variable (Y):", df.select_dtypes(include='number').columns)
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            color=target_col if target_col else None,
            color_discrete_sequence=px.colors.sequential.Teal,
            title=f"{col_y} vs {col_x} grouped by {target_col}" if target_col else f"Scatterplot of {col_y} vs {col_x}"
        )

    elif tipo == "Heatmap (Correlation)":
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
        else:
            st.warning("⚠️ Not enough numerical variables to compute correlation.")
            return

    # =====================================================
    # ESTILO GLOBAL DEL GRÁFICO
    # =====================================================
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        title=dict(
            text=fig.layout.title.text,
            font=dict(size=18, color="#00bcd4", family="Inter, sans-serif"),
            x=0,
            y=0.95
        ),
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# FUNCIÓN PRINCIPAL DE KPIs + GRÁFICOS
# =====================================================
def ejecutar_eda_2(df):
    apply_style()

    mostrar_kpis(df)
    generador_graficos(df)
