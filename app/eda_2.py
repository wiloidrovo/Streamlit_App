import streamlit as st
import pandas as pd
import plotly.express as px

# Mostrar KPIs
def mostrar_kpis(df, target_col=None):
    st.subheader("Key Performance Indicators (KPIs)")

    col1, col2, col3 = st.columns(3)

    # Shape
    col1.metric("Total rows", df.shape[0])
    col1.metric("Total columns", df.shape[1])

    # Porcentaje de nulos
    total_celdas = df.shape[0] * df.shape[1]
    total_nulos = df.isnull().sum().sum()
    pct_nulos = (total_nulos / total_celdas) * 100
    col2.metric("% of null values", f"{pct_nulos:.2f}%")

    # Promedio de una variable numérica a elección del usuario
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        col_selec = col3.selectbox("Select numeric column for KPI:", num_cols, key="kpi_col")
        promedio = df[col_selec].mean()
        col3.metric(f"Average of {col_selec}", f"{promedio:.2f}")
    else:
        col3.info("There are no numeric columns")

    # Si el target está definido, mostramos su distribución
    if target_col and target_col in df.columns:
        st.markdown(f"### Distribution of the target variable: `{target_col}`")

        if pd.api.types.is_numeric_dtype(df[target_col]):
            fig = px.histogram(
                df, x=target_col, nbins=20,
                title=f"Distribution of {target_col}"
            )
        else:
            dist = df[target_col].value_counts(normalize=True).reset_index()
            dist.columns = [target_col, "Proportion"]
            dist["Proportion"] = dist["Proportion"] * 100

            fig = px.pie(
                dist, names=target_col, values="Proportion",
                title=f"Distribution of {target_col}",
                hole=0.3
            )
            fig.update_traces(textinfo="label+percent")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("A target variable has not yet been defined.")

# Generar gráficos interactivos
def generador_graficos(df):
    st.subheader("Interactive chart generator")

    # Seleccionar variable objetivo
    if "target_col" not in st.session_state:
        st.session_state.target_col = None

    options = [None] + list(df.columns)
    default_index = (
        options.index(st.session_state.target_col)
        if st.session_state.target_col in df.columns else 0
    )

    st.session_state.target_col = st.selectbox(
        "Select target variable (optional):",
        options,
        index=default_index
    )

    target_col = st.session_state.target_col

    # Selección de tipo de gráfico
    tipos = ["Histogram", "Bar chart", "Boxplot", "Scatterplot", "Heatmap (correlation)"]
    tipo = st.selectbox("Selecciona el tipo de gráfico:", tipos)

    if tipo == "Histogram":
        col_x = st.selectbox("Variable (X):", df.columns)
        fig = px.histogram(df, x=col_x, color=target_col if target_col else None, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Bar chart":
        col_x = st.selectbox("Categorical variable (X):", df.columns)
        col_y = st.selectbox("Numeric variable (Y):", df.select_dtypes(include="number").columns)
        fig = px.bar(df, x=col_x, y=col_y, color=target_col if target_col else None, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Boxplot":
        col_x = st.selectbox("Categorical variable (X):", df.columns)
        col_y = st.selectbox("Numeric variable (Y):", df.select_dtypes(include="number").columns)
        fig = px.box(df, x=col_x, y=col_y, color=target_col if target_col else None)
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Scatterplot":
        col_x = st.selectbox("Numeric variable (X):", df.select_dtypes(include="number").columns)
        col_y = st.selectbox("Numeric variable (Y):", df.select_dtypes(include="number").columns)
        fig = px.scatter(df, x=col_x, y=col_y, color=target_col if target_col else None)
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Heatmap (correlation)":
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("There are not enough numerical variables to calculate correlation")

# Función principal de KPIs y gráficos
def ejecutar_eda_2(df):
    # KPIs
    mostrar_kpis(df, target_col=st.session_state.get("target_col", None))

    # Gráficos
    generador_graficos(df)