import streamlit as st
from app.carga_datos import cargar_archivo, mostrar_info, mostrar_estadisticas, mostrar_preview
from app.eda import ejecutar_eda

def main():
    # Título de la aplicación
    st.title("Análisis de Datos - Telco Churn")

    # Sidebar para navegación entre pestañas
    st.sidebar.title("Navigation Menu")
    opciones = ["Carga de Datos", "Visualización de Datos", "EDA", "ML Model"]
    seleccion = st.sidebar.radio("Select a tab:", opciones)

    # Cargar Datos
    if seleccion == "Carga de Datos":
        st.subheader("Sube tu archivo de datos")
        
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
            st.subheader("Opciones de visualización")
            opciones_submenu = ["Data Preview", "Data Information", "Descriptive Statistics"]
            seleccion_submenu = st.radio("Selecciona qué ver:", opciones_submenu, horizontal=True)  # Usamos `horizontal=True` para que quede en una fila

            # Mostrar la información general o estadísticas descriptivas
            if seleccion_submenu == "Data Preview":
                mostrar_preview(st.session_state.df_original)  # Muestra las primeras filas del archivo
            elif seleccion_submenu == "Data Information":
                mostrar_info(st.session_state.df_original)  # Mostrar la información general del dataframe
            elif seleccion_submenu == "Descriptive Statistics":
                mostrar_estadisticas(st.session_state.df_original)  # Mostrar estadísticas descriptivas

    # Data Visualization
    elif seleccion == "Visualización de Datos":
        if 'df_original' in st.session_state:
            st.subheader("Vista previa del archivo cargado")
            st.write(st.session_state.df_original.head(10))
        else:
            st.warning("Please upload a file first.")

    # EDA
    elif seleccion == "EDA":
        #if 'df' in st.session_state:
        if 'df_original' in st.session_state:
            st.subheader("Análisis Exploratorio de Datos (EDA)")

            df_limpio = ejecutar_eda(st.session_state.df_original)
            st.session_state.df = df_limpio # Dataset final actualizado para ML

            # Usamos df de trabajo, sin modificar el original
            #st.session_state.df = ejecutar_eda(st.session_state.df)
        else:
            st.warning("Please upload a file first.")

    # Machine Learning Model
    elif seleccion == "ML Model":
        st.subheader("Machine Learning Model")

        if "df" in st.session_state:
            st.write("Dataset listo para ML:")
            st.write(st.session_state.df.head())
            st.info(f"Shape: {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")
        else:
            st.warning("Primero realiza el EDA para preparar el dataset.")

        # Aquí se colocará el modelo ML cuando esté listo
        

if __name__ == "__main__":
    main()
