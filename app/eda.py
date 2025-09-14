import streamlit as st
import pandas as pd
#from app.eda_2 import ejecutar_eda_2

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

def mostrar_conclusiones():
    st.markdown(
        """
        <div style="
            border-radius: 12px;
            padding: 20px 25px;
            margin-top: 30px;
            background-color: #1e1e1e;
            border: 1px solid #444;
        ">
            <h3 style="color:#93ACF6; margin-bottom:15px;">Conclusions of the EDA</h3>
            <ul style="color:#e0e0e0; font-size:16px; line-height:1.6;">
                <li>After imputation, the dataset reaches a sufficient structural quality level, making it reliable for churn prediction modeling.</li>
                <li>The distribution of the target variable <b>Churn</b> clearly shows the percentage of customers who cancel the service, providing a direct view of the problem and a solid basis for retention strategies.</li>
                <li>The average <b>MonthlyCharges</b> suggests a correlation between service cost and retention: customers with higher charges tend to show higher churn rates.</li>
                <li>The analysis of <b>tenure</b> and <b>contract type</b> shows that customers with monthly contracts present higher churn rates, while longer-term contracts foster greater loyalty. This supports the importance of promoting long-term plans.</li>
                <li>Although they represent only 16% of the sample, <b>SeniorCitizen</b> customers may show different churn behavior compared to non-seniors, suggesting that age could be a relevant predictor.</li>
                <li>Differences observed in <b>payment methods</b> and additional services (Internet, phone service, streaming) may influence customer retention, highlighting that billing preferences and service combinations are important churn factors.</li>
            </ul>
            <p style="margin-top:15px; color:#cccccc; font-size:15px;">
                Overall, the analysis confirms that <b>churn is multifactorial</b>, influenced by contract conditions, service-related factors, and demographic characteristics. 
                This provides a solid foundation for developing predictive models and effective loyalty programs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    

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

    from app.eda_2 import ejecutar_eda_2
    ejecutar_eda_2(df_revised)

    from app.eda_target import ejecutar_eda_target
    ejecutar_eda_target(df_revised)

    mostrar_conclusiones()

    # Retornamos el dataframe modificado
    return df_revised