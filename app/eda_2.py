import streamlit as st
import pandas as pd
import plotly.express as px

# Mostrar KPIs
def mostrar_kpis(df, target_col=None):
    st.subheader("🔹 Indicadores Clave (KPI)")

    col1, col2, col3 = st.columns(3)

    # Shape
    col1.metric("Total filas", df.shape[0])
    col1.metric("Total columnas", df.shape[1])

    # Porcentaje de nulos
    total_celdas = df.shape[0] * df.shape[1]
    total_nulos = df.isnull().sum().sum()
    pct_nulos = (total_nulos / total_celdas) * 100
    col2.metric("% de valores nulos", f"{pct_nulos:.2f}%")

    # Promedio de una variable numérica a elección del usuario
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        col_selec = col3.selectbox("Selecciona columna numérica para KPI:", num_cols, key="kpi_col")
        promedio = df[col_selec].mean()
        col3.metric(f"Promedio de {col_selec}", f"{promedio:.2f}")
    else:
        col3.info("No hay columnas numéricas")

# Generar gráficos interactivos
def generador_graficos(df):
    st.subheader("Generador de gráficos interactivos")

    # Seleccionar variable objetivo
    target_col = st.selectbox("Selecciona variable objetivo (opcional):", [None] + list(df.columns))

    # Selección de tipo de gráfico
    tipos = ["Histograma", "Gráfico de barras", "Boxplot", "Scatterplot", "Heatmap (correlación)"]
    tipo = st.selectbox("Selecciona el tipo de gráfico:", tipos)

    if tipo == "Histograma":
        col_x = st.selectbox("Variable (X):", df.columns)
        fig = px.histogram(df, x=col_x, color=target_col if target_col else None, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Gráfico de barras":
        col_x = st.selectbox("Variable categórica (X):", df.columns)
        col_y = st.selectbox("Variable numérica (Y):", df.select_dtypes(include="number").columns)
        fig = px.bar(df, x=col_x, y=col_y, color=target_col if target_col else None, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Boxplot":
        col_x = st.selectbox("Variable categórica (X):", df.columns)
        col_y = st.selectbox("Variable numérica (Y):", df.select_dtypes(include="number").columns)
        fig = px.box(df, x=col_x, y=col_y, color=target_col if target_col else None)
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Scatterplot":
        col_x = st.selectbox("Variable numérica (X):", df.select_dtypes(include="number").columns)
        col_y = st.selectbox("Variable numérica (Y):", df.select_dtypes(include="number").columns)
        fig = px.scatter(df, x=col_x, y=col_y, color=target_col if target_col else None)
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Heatmap (correlación)":
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes variables numéricas para calcular correlación")

# Función principal de KPIs y gráficos
def ejecutar_eda_2(df):
    # KPIs
    mostrar_kpis(df)

    # Gráficos
    generador_graficos(df)