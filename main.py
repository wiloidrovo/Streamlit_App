import streamlit as st
import os
from app.carga_datos import cargar_archivo, mostrar_info, mostrar_estadisticas, mostrar_preview
from app.eda import ejecutar_eda

def main():
    # Título de la aplicación
    st.title("Data Analysis")

    # Sidebar para navegación entre pestañas
    st.sidebar.title("Navigation Menu")
    opciones = ["Data Loading", "EDA", "ML Model", "Dashboard", "Business Impact"]
    seleccion = st.sidebar.radio("Select a tab:", opciones)

    # Cargar Datos
    if seleccion == "Data Loading":
        st.subheader("Upload your data file")
        
        # Si no hemos cargado aún el archivo, mostramos  file_uploader
        if "df_original" not in st.session_state:
            df = cargar_archivo()
            if df is not None:
                st.session_state.df_original = df.copy()   # guardamos original
                st.session_state.df = df.copy()            # copia de trabajo
                st.sidebar.success("File uploaded successfully!")

        # Si ya existe el dataset en sesión, mostramos menú de visualización
        if "df_original" in st.session_state:
            
            # Menú interactivo dentro de la pestaña "Carga de Archivos"
            st.subheader("Display options")
            opciones_submenu = ["Data Preview", "Data Information", "Descriptive Statistics"]
            seleccion_submenu = st.radio("Choose what to view:", opciones_submenu, horizontal=True)  # Usamos `horizontal=True` para que quede en una fila

            # Mostrar la información general o estadísticas descriptivas
            if seleccion_submenu == "Data Preview":
                mostrar_preview(st.session_state.df_original)  # Muestra las primeras filas del archivo
            elif seleccion_submenu == "Data Information":
                mostrar_info(st.session_state.df_original)  # Mostrar la información general del dataframe
            elif seleccion_submenu == "Descriptive Statistics":
                mostrar_estadisticas(st.session_state.df_original)  # Mostrar estadísticas descriptivas

    # EDA
    elif seleccion == "EDA":
        #if 'df' in st.session_state:
        if 'df_original' in st.session_state:
            st.subheader("Exploratory Data Analysis (EDA)")

            df_limpio = ejecutar_eda(st.session_state.df_original)
            st.session_state.df = df_limpio # Dataset final actualizado para ML

            # Usamos df de trabajo, sin modificar el original
            #st.session_state.df = ejecutar_eda(st.session_state.df)
        else:
            st.warning("Please upload a file first.")

    # Machine Learning Model
    elif seleccion == "ML Model":
        st.subheader("Machine Learning Model")
        from app.ml_page import prediction_page
        if "df" in st.session_state:
            st.write("Dataset ready for Machine Learning Model:")
            st.write(st.session_state.df.head())
            st.info(f"Shape: {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")

            # Convertir el DataFrame a CSV en memoria
            csv = st.session_state.df.to_csv(index=False).encode("utf-8")

            # Botón para descargar el dataset final
            st.download_button(
                label="Download cleaned dataset (CSV)",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )
            prediction_page(st.session_state.df)
        else:
            st.warning("First, perform the EDA to prepare the dataset.")

    # Dashboard
    elif seleccion == "Dashboard":
        from app.dashboard import dashboard_page
        dashboard_page(st.session_state.get("df", None))

    # Business Impact
    elif seleccion == "Business Impact":
        from app.business_impact import business_impact_page
        if "df" in st.session_state:
            business_impact_page(st.session_state.df)
        else:
            st.warning("Please prepare the dataset in EDA first.")
        

if __name__ == "__main__":
    main()