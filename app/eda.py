import streamlit as st
import pandas as pd

# =====================================================
# 💅 ESTILO PERSONALIZADO (mismo tema que main.py)
# =====================================================
st.markdown("""
    <style>
        /* Contenedor principal */
        .main {
            background-color: #0e1117;
            color: #fafafa;
            font-family: "Inter", sans-serif;
        }
        h1, h2, h3, h4 {
            color: #00bcd4;
            font-weight: 700 !important;
        }
        p, label, .stMarkdown {
            color: #e0e0e0;
        }

        /* Subheaders con acento */
        div.block-container h2 {
            border-left: 5px solid #00bcd4;
            padding-left: 10px;
            margin-top: 1.5em;
        }

        /* Tarjetas elegantes */
        .stCard {
            background: #111827;
            padding: 1.2em 1.5em;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,.3);
            margin-bottom: 1.5em;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
        }

        /* Botones */
        div.stButton > button {
            background: linear-gradient(90deg, #00bcd4, #2196f3);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1.1em;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #26c6da, #42a5f5);
            transform: scale(1.03);
        }

        /* Tablas y dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            border: 1px solid #1f2937;
        }

        /* Mensajes */
        .stSuccess {
            background-color: #063d33;
            border-left: 6px solid #00e676;
            color: #d4f3e3;
        }
        .stInfo {
            background-color: #0b2239;
            border-left: 6px solid #42a5f5;
            color: #cfe8ff;
        }
        .stWarning {
            background-color: #332b00;
            border-left: 6px solid #ffeb3b;
            color: #fff8b0;
        }
    </style>
""", unsafe_allow_html=True)


# =====================================================
# ⚙️ FUNCIONES DE PROCESAMIENTO
# =====================================================

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


def imputar_nulos(df):
    st.subheader("🧩 Treatment of Null Values")

    # Inicializar log de imputaciones
    if "imputaciones" not in st.session_state:
        st.session_state.imputaciones = []

    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]

    if null_summary.empty:
        st.info("✅ There are no null values in the current dataset.")
    else:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.write("Columns with null values:")
        st.dataframe(null_summary.rename("Missing Values"))
        st.markdown("</div>", unsafe_allow_html=True)

        col_seleccionada = st.selectbox(
            "📌 Select the column to be imputed:",
            options=null_summary.index
        )

        estrategia = None
        constante = None

        if col_seleccionada:
            if pd.api.types.is_numeric_dtype(df[col_seleccionada]):
                estrategia = st.radio(
                    f"⚙️ Strategy for imputing `{col_seleccionada}` (numerical):",
                    ["Mean", "Median", "Mode", "Constant Value", "Delete rows"],
                    horizontal=True
                )
                if estrategia == "Constant Value":
                    constante = st.number_input(
                        f"Enter the constant value for `{col_seleccionada}`:"
                    )
            else:
                estrategia = st.radio(
                    f"⚙️ Strategy for imputing `{col_seleccionada}` (categorical):",
                    ["Mode", "Constant Value", "Delete rows"],
                    horizontal=True
                )
                if estrategia == "Constant Value":
                    constante = st.text_input(
                        f"Enter the constant value for `{col_seleccionada}`:"
                    )

        if st.button("💾 Apply Imputation"):
            if estrategia and col_seleccionada:
                st.session_state.imputaciones = [
                    imp for imp in st.session_state.imputaciones if imp[0] != col_seleccionada
                ]
                st.session_state.imputaciones.append((col_seleccionada, estrategia, constante))
                st.success(f"✅ Imputation saved: `{col_seleccionada}` → {estrategia}")

    # Mostrar historial de imputaciones
    if st.session_state.imputaciones:
        st.sidebar.markdown("### 🧮 Imputations History")
        for col, strat, val in st.session_state.imputaciones:
            detalle = f"• **{col}** → {strat}"
            if val not in [None, ""]:
                detalle += f" ({val})"
            st.sidebar.markdown(detalle)


def mostrar_info(df):
    st.subheader("📋 General Information (Updated Dataset)")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Null Count': df.isnull().sum().values,
        'Dtype': df.dtypes.values
    })
    st.dataframe(info_df)


# =====================================================
# 🎨 ESTRUCTURA PRINCIPAL DE EDA
# =====================================================

def ejecutar_eda(df_original):

    # 1️⃣ Imputación de valores nulos
    df = imputar_nulos(df_original)

    # 2️⃣ Aplicar imputaciones
    df = aplicar_imputaciones(df_original, st.session_state.get("imputaciones", []))

    # 3️⃣ Inicialización de columnas eliminadas
    if "eliminadas" not in st.session_state:
        st.session_state.eliminadas = []

    st.sidebar.subheader("🧱 Column Management")

    # 4️⃣ Eliminar columnas
    cols_a_eliminar = st.sidebar.multiselect(
        "Select columns to delete:",
        options=[col for col in df.columns if col not in st.session_state.eliminadas]
    )

    for col in cols_a_eliminar:
        if col not in st.session_state.eliminadas:
            st.session_state.eliminadas.append(col)

    # 5️⃣ Recuperar columnas
    cols_a_recuperar = st.sidebar.multiselect(
        "Select columns to recover:",
        options=st.session_state.eliminadas
    )

    for col in cols_a_recuperar:
        if col in st.session_state.eliminadas:
            st.session_state.eliminadas.remove(col)

    # 6️⃣ Aplicar eliminaciones
    df_revised = df.drop(columns=st.session_state.eliminadas, errors="ignore")

    # Mostrar resumen de columnas eliminadas
    if st.session_state.eliminadas:
        st.sidebar.warning(f"🗑️ Deleted columns: {', '.join(st.session_state.eliminadas)}")
    else:
        st.sidebar.info("No deleted columns.")

    # 7️⃣ Preview del dataset
    st.subheader("📊 Data Preview after Cleaning")
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.dataframe(df_revised.head(5))
    st.markdown("</div>", unsafe_allow_html=True)
    st.info(f"**Final Dataset Shape:** {df_revised.shape[0]} rows × {df_revised.shape[1]} columns")

    # 8️⃣ Información general
    mostrar_info(df_revised)

    st.markdown("---")

    # 9️⃣ Ejecutar módulos adicionales
    from app.eda_2 import ejecutar_eda_2
    ejecutar_eda_2(df_revised)

    st.markdown("---")

    from app.eda_target import ejecutar_eda_target
    ejecutar_eda_target(df_revised)

    return df_revised
