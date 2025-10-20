import streamlit as st
import pandas as pd

# Función para aplicar imputaciones sobre el dataset base
def aplicar_imputaciones(df, imputaciones):
    df_copy = df.copy()
    for col, strat, val in imputaciones:
        if strat == "Mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strat == "Median":
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strat == "Constant":
            df_copy[col] = df_copy[col].fillna(val)
        elif strat == "Delete rows":
            df_copy = df_copy.dropna(subset=[col])
        elif strat == "Mode":
            moda = df_copy[col].mode()[0]
            df_copy[col] = df_copy[col].fillna(moda)
    return df_copy

# Función para imputar datos nulos
def imputar_nulos(df):
    st.subheader("Treatment of null values")

    # Inicializar log de imputaciones
    if "imputaciones" not in st.session_state:
        st.session_state.imputaciones = []

    # Resumen de nulos
    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]

    if null_summary.empty:
        st.info("There are no null values in the current dataset.")
    else:
        st.write("Columns with null values:")
        st.dataframe(null_summary.rename("Missing Values"))

        # Seleccionar columna a imputar
        col_seleccionada = st.selectbox(
            "Select the column to be imputed:",
            options = null_summary.index
        )

        estrategia = None
        constante = None

        if col_seleccionada:
            if pd.api.types.is_numeric_dtype(df[col_seleccionada]):
                estrategia = st.radio(
                    f"Strategy for imputing {col_seleccionada} (numerical):",
                    ["Mean", "Median", "Mode", "Constant Value", "Delete rows"],
                    horizontal=True
                )
                if estrategia == "Valor constante":
                    constante = st.number_input(
                        f"Ingrese el valor con el que desea imputar {col_seleccionada}:"
                    )
            else:
                estrategia = st.radio(
                    f"Strategy for imputing {col_seleccionada} (categorical):",
                    ["Mode", "Constant Value", "Delete rows"],
                    horizontal=True
                )
                if estrategia == "Constant Value":
                    constante = st.text_input(
                        f"Enter the value to be imputed {col_seleccionada}:"
                    )

        # Botón para aplicar imputación
        if st.button("Apply imputation"):
            if estrategia and col_seleccionada:
                # Si ya existe imputación previa para esta columna, la reemplaza
                st.session_state.imputaciones = [
                    imp for imp in st.session_state.imputaciones if imp[0] != col_seleccionada
                ]
                st.session_state.imputaciones.append((col_seleccionada, estrategia, constante))
                st.success(f"Imputation saved: {col_seleccionada} -> {estrategia}")

    # Mostrar historial de imputaciones
    if st.session_state.imputaciones:
        st.sidebar.info("Imputations History:")
        for col, strat, val in st.session_state.imputaciones:
            detalle = f"{col} -> {strat}"
            if val not in [None, ""]:
                detalle += f" ({val})"
            st.sidebar.write(detalle)

# Función que muestra un resumen general
def mostrar_info(df):
    st.subheader("General Information (Updated Dataset)")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Null Count': df.isnull().sum().values,
        'Dtype': df.dtypes.values
    })
    st.dataframe(info_df)

# Función principal de EDA
def ejecutar_eda(df_original):

    # 1. Imputación de valores nulos
    df = imputar_nulos(df_original)

    # 2. Aplicar imputaciones al dataset original
    df = aplicar_imputaciones(df_original, st.session_state.get("imputaciones", []))

    # 3. Inicializar lista de columnas eliminadas en session_state
    if "eliminadas" not in st.session_state:
        st.session_state.eliminadas = []

    st.sidebar.subheader("Column Management")

    # 4. Selección de columnas a eliminar
    cols_a_eliminar = st.sidebar.multiselect(
        "Select columns to delete:",
        options=[col for col in df.columns if col not in st.session_state.eliminadas]
    )

    #if cols_a_eliminar:
    for col in cols_a_eliminar:
        if col not in st.session_state.eliminadas:
            st.session_state.eliminadas.append(col)

    # 2. Opción para recuperar columnas eliminadas
    cols_a_recuperar = st.sidebar.multiselect(
        "Select columns to recover:",
        options=st.session_state.eliminadas
    )

    #if cols_a_recuperar:
    for col in cols_a_recuperar:
        if col in st.session_state.eliminadas:
            st.session_state.eliminadas.remove(col)

    # Aplicar eliminaciones
    df_revised = df.drop(columns=st.session_state.eliminadas, errors="ignore")

    # Mostrar siempre qué columnas están eliminadas
    if st.session_state.eliminadas:
        st.sidebar.warning(f"Deleted columns: {', '.join(st.session_state.eliminadas)}")
    else:
        st.sidebar.info("There are no deleted columns")

    # Preview del dataset después de limpieza
    st.subheader("Data Preview after cleaning")
    st.write(df_revised.head(5))
    st.info(f"Dataset final shape: {df_revised.shape[0]} rows x {df_revised.shape[1]} columns")

    # Resumen general actualizado
    mostrar_info(df_revised)

    st.markdown("---")

    from app.eda_2 import ejecutar_eda_2
    ejecutar_eda_2(df_revised)

    st.markdown("---")

    from app.eda_target import ejecutar_eda_target
    ejecutar_eda_target(df_revised)

    # Retornamos el dataframe modificado
    return df_revised