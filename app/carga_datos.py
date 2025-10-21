import streamlit as st
import pandas as pd
from app.utils import apply_style

def convertir_columnas_numericas(df):
    columnas_convertidas = []
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            nan_ratio = df[col].isna().mean()
            if nan_ratio > 0.4:
                df[col] = original
            else:
                columnas_convertidas.append(col)
        except Exception:
            pass
    return df, columnas_convertidas

def convertir_binarias_a_categoricas(df):
    columnas_convertidas = []
    for col in df.select_dtypes(include=["number"]).columns:
        valores = df[col].dropna().unique()
        if set(valores).issubset({0, 1}) and len(valores) <= 2:
            df[col] = df[col].map({0: "No", 1: "Yes"}).astype("object")
            columnas_convertidas.append(col)
    return df, columnas_convertidas

# =====================================================
# CARGA DE ARCHIVOS
# =====================================================

def cargar_archivo():
    apply_style()
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    archivo = st.file_uploader(
        "Drag & Drop or browse your CSV / Excel file below 👇",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if archivo is not None:
        st.success(f"✅ **File uploaded successfully:** `{archivo.name}`")

        try:
            with st.spinner("⏳ Reading and cleaning data..."):
                if archivo.name.endswith(".csv"):
                    df = pd.read_csv(archivo)
                else:
                    df = pd.read_excel(archivo)

                df, cols_num = convertir_columnas_numericas(df)
                df, cols_bin = convertir_binarias_a_categoricas(df)

            st.session_state.dataset_shape = (df.shape[0], df.shape[1])

            if cols_num:
                st.sidebar.info(f"🔢 Columns converted to numeric: {', '.join(cols_num)}")
            if cols_bin:
                st.sidebar.info(f"🟩 Binary columns converted to categorical: {', '.join(cols_bin)}")

            return df

        except Exception as e:
            st.error(f"❌ Error reading the file: **{e}**")

    else:
        st.info("📥 Please upload a file to continue.")

    return None

# =====================================================
# INFORMACIÓN GENERAL DEL DATASET
# =====================================================

def mostrar_info(df):
    apply_style()
    st.subheader("📋 General Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Dtype": df.dtypes.values
    })
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.dataframe(info_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🏷️ Categorical Columns — Unique Values")
    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0:
        summary_data = [(col, df[col].unique()) for col in cat_cols]
        summary_df = pd.DataFrame(summary_data, columns=["Column", "Unique Values"])
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No categorical columns found in the dataset.")

# =====================================================
# ESTADÍSTICAS DESCRIPTIVAS
# =====================================================

def mostrar_estadisticas(df):
    apply_style()
    st.subheader("📈 Descriptive Statistics")

    st.markdown("#### 🔢 Numerical Features")
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 🏷️ Categorical Features")
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.dataframe(df[cat_cols].describe().T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No categorical features found in this dataset.")

# =====================================================
# PREVISUALIZACIÓN DE DATOS (con Dataset Shape)
# =====================================================

def mostrar_preview(df):
    apply_style()
    st.subheader("👁️ Data Preview")
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "dataset_shape" in st.session_state:
        rows, cols = st.session_state.dataset_shape
        st.markdown(
            f"""
            <div style='background-color:#16213e; color:#e0e0e0; padding:12px 20px;
            border-radius:8px; border:1px solid #1f2937; margin-top:12px;'>
                <b>📊 Dataset Shape:</b> {rows:,} rows × {cols} columns
            </div>
            """,
            unsafe_allow_html=True
        )