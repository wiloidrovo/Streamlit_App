import streamlit as st
import pandas as pd

# Función auxiliar para convertir columnas object a numéricas cuando sea posible
def convertir_columnas_numericas(df):
    """
    Intenta convertir automáticamente columnas tipo 'object' a numéricas
    cuando sea posible. Si más del 40% de los valores se vuelven NaN,
    se revierte la conversión (ej. columnas de texto reales).
    Retorna el dataframe y la lista de columnas convertidas
    """
    columnas_convertidas = []
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")

            nan_ratio = df[col].isna().mean()
            if nan_ratio > 0.4:  
                # Si se pierden demasiados datos, revertimos
                df[col] = original
            else:
                columnas_convertidas.append(col)
        except Exception:
            # Si no se puede convertir, la dejamos igual
            pass
    return df, columnas_convertidas

def convertir_binarias_a_categoricas(df):
    """
    Convierte automáticamente columnas numéricas con valores únicos {0, 1} a categóricas ("No"/"Yes").
    Retorna el dataframe y la lista de columnas convertidas.
    """
    columnas_convertidas = []
    for col in df.select_dtypes(include=["number"]).columns:
        valores = df[col].dropna().unique()
        # Detectamos columnas binarias puras (solo 0 y 1)
        if set(valores).issubset({0, 1}) and len(valores) <= 2:
            df[col] = df[col].map({0: "No", 1: "Yes"}).astype("object")
            columnas_convertidas.append(col)
    return df, columnas_convertidas

# Función para cargar el archivo
def cargar_archivo():
    archivo = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if archivo is not None:
        # Mostrar nombre del archivo cargado
        st.write(f"**File uploaded successfully!** {archivo.name}")
        
        try:
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)

            df,cols_num = convertir_columnas_numericas(df)

            df, cols_bin = convertir_binarias_a_categoricas(df)

            if cols_num:
                st.sidebar.info(f"Columns converted to numeric: {', '.join(cols_num)}")
            if cols_bin:
                st.sidebar.info(f"Binary columns converted to categorical: {', '.join(cols_bin)}")

            return df

        except Exception as e:
            st.error(f"Error reading the file: {e}")
    return None

def mostrar_info(df):
    st.subheader('General Information')
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Null Count': df.isnull().sum().values,
        'Dtype': df.dtypes.values
    })
    st.dataframe(info_df)

    # Resumen de columnas categóricas con sus valores únicos
    st.subheader("Categorical Columns - Unique Values")
    cat_cols = df.select_dtypes(include='object').columns

    if len(cat_cols) > 0:
        data = [(col, df[col].unique()) for col in cat_cols]
        summary_df = pd.DataFrame(data, columns=['Column', 'Unique Values'])
        st.dataframe(summary_df)
    else:
        st.info("No categorical columns found.")


# Función para mostrar las estadísticas descriptivas
def mostrar_estadisticas(df):
    st.subheader('Descriptive Statistics')

    # Numerical Statistics
    st.markdown('**Numerical Features**')
    st.dataframe(df.describe().T)

    # Categorical Statistics
    st.markdown('**Categorical Features**')
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        st.dataframe(df[cat_cols].describe().T)
    else:
        st.info("No categorical features found.")

# Función para mostrar la vista previa de los datos (Data Preview)
def mostrar_preview(df):
    st.subheader('Data Preview')
    st.write(df.head(10))
