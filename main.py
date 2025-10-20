import streamlit as st
import os
from app.carga_datos import cargar_archivo, mostrar_info, mostrar_estadisticas, mostrar_preview
from app.eda import ejecutar_eda
from app.utils import apply_style

# ==============================================
# CONFIGURACIÓN GLOBAL
# ==============================================
st.set_page_config(
    page_title="Telco Churn Analyzer",
    page_icon="📊",
    layout="wide"
)

# ==============================================
# APLICACIÓN PRINCIPAL
# ==============================================
def main():
    apply_style()

    st.markdown("""
    <div style="text-align:center; padding: 1.5em 0; border-bottom: 1px solid #2a2a40;">
        <h1 style="color:#00bcd4; margin-bottom:0;">📊 Telco Customer Churn Analyzer</h1>
        <p style="color:#a5a5a5;">End-to-End Machine Learning Web App with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("logo.png", width=140)
        st.markdown("### 🧭 Navigation")
        opciones = ["Data Loading", "EDA", "ML Model", "Dashboard", "Business Impact"]
        seleccion = st.radio("Select a tab:", opciones, index=0)
        st.markdown("---")
        st.caption("💡 Developed with Streamlit | Sancocho de Mono 2025")

    # ==============================================
    # DATA LOADING
    # ==============================================
    if seleccion == "Data Loading":
        st.subheader("📂 Upload Your Dataset")

        if "df_original" not in st.session_state:
            df = cargar_archivo()
            if df is not None:
                st.session_state.df_original = df.copy()
                st.session_state.df = df.copy()
                st.sidebar.success("✅ File uploaded successfully!")

        if "df_original" in st.session_state:
            st.markdown("### 👁️ Data Overview")
            opciones_submenu = ["Data Preview", "Data Information", "Descriptive Statistics"]
            seleccion_submenu = st.radio("Choose what to view:", opciones_submenu, horizontal=True)

            if seleccion_submenu == "Data Preview":
                mostrar_preview(st.session_state.df_original)
            elif seleccion_submenu == "Data Information":
                mostrar_info(st.session_state.df_original)
            elif seleccion_submenu == "Descriptive Statistics":
                mostrar_estadisticas(st.session_state.df_original)

    # ==============================================
    # EDA SECTION
    # ==============================================
    elif seleccion == "EDA":
        if 'df_original' in st.session_state:
            st.subheader("📊 Exploratory Data Analysis (EDA)")
            df_limpio = ejecutar_eda(st.session_state.df_original)
            st.session_state.df = df_limpio
        else:
            st.warning("⚠️ Please upload a file first to perform EDA.")

    # ==============================================
    # ML MODEL
    # ==============================================
    elif seleccion == "ML Model":
        from app.ml_page import prediction_page
        st.subheader("🤖 Machine Learning Model")
        if "df" in st.session_state:
            st.success("✅ Dataset is ready for model prediction.")
            st.dataframe(st.session_state.df.head())

            csv = st.session_state.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download cleaned dataset (CSV)",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

            prediction_page(st.session_state.df)
        else:
            st.warning("⚠️ Perform the EDA first to prepare the dataset.")

    # ==============================================
    # DASHBOARD
    # ==============================================
    elif seleccion == "Dashboard":
        from app.dashboard import dashboard_page
        st.subheader("📈 Model Dashboard")
        dashboard_page(st.session_state.get("df", None))

    # ==============================================
    # BUSINESS IMPACT
    # ==============================================
    elif seleccion == "Business Impact":
        from app.business_impact import business_impact_page
        st.subheader("💼 Business Impact Analysis")
        if "df" in st.session_state:
            business_impact_page(st.session_state.df)
        else:
            st.warning("⚠️ Please prepare the dataset in EDA first.")

# ==============================================
# RUN APP
# ==============================================
if __name__ == "__main__":
    main()