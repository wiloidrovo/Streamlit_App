# Streamlit EDA App

Esta aplicación interactiva desarrollada en **Python + Streamlit** permite realizar **Análisis Exploratorio de Datos (EDA)** y **predicciones con un modelo de Machine Learning** de manera intuitiva y flexible.  
El objetivo es **cargar datos, limpiarlos, analizarlos, visualizarlos y preparar un dataset final para la construcción y uso de modelos predictivos**.

---

## Características principales

### **Carga y Preprocesamiento**

- Subida de datasets en formato `.csv`.
- Imputación de valores nulos:
  - Media, Mediana, Moda
  - Valor constante
  - Eliminación de filas
- Eliminación y recuperación de columnas (persistente en `session_state`).
- Vista previa del dataset actualizado.
- Información general de columnas:
  - Conteo de valores nulos/no nulos.
  - Tipos de datos.

### **KPIs dinámicos**

- Total de filas y columnas.
- Porcentaje de valores nulos.
- Promedio de cualquier variable numérica seleccionada.
- Distribución de la variable objetivo con **gráfico circular** (si está definida).

### **Generador de Gráficos Interactivos**

- Histograma
- Gráfico de barras
- Boxplot
- Scatterplot
- Heatmap de correlaciones

### **EDA Automático sobre la Variable Objetivo**

- Comparación de cada variable frente al target definido.
- Boxplots para variables numéricas.
- Gráficos de barras apiladas (% de distribución) para categóricas.
- Histogramas segmentados.
- Tablas de distribución.
- Detección de desbalance de clases en el target.

---

### **Modelo de Machine Learning**

- Preprocesamiento con **Pipelines + ColumnTransformer**:
  - Escalado robusto en variables numéricas.
  - OneHotEncoding en categóricas.
- Entrenamiento y evaluación de **Decision Tree Classifier**.
- Exportación del pipeline + modelo en `models/decision_tree_pipeline.pkl`.

---

### **Predicción en Streamlit**

- Carga del modelo entrenado desde `models/`.
- Formulario dinámico con campos adaptados:
  - `number_input` para variables numéricas.
  - `selectbox` para categóricas.
- El input del usuario pasa automáticamente por el pipeline.
- Se muestra:
  - **Clase predicha (Cancelado / No Cancelado)**.
  - **Probabilidad de cancelación (%)**.

---

## Estructura del proyecto

```bash
Streamlit_App/
│
├── app/                          # Módulos principales
│   ├── carga_datos.py            # Carga y previsualización de datasets
│   ├── eda.py                    # Lógica principal del EDA (imputaciones, limpieza, columnas)
│   ├── eda_2.py                  # KPIs y generador de gráficos
│   ├── eda_target.py             # Análisis automático del target
│   ├── ml_page.py                # Página del modelo ML con formulario de predicción
│   ├── pipelines_transf.py       # Pipelines y transformadores personalizados
│   └── utils.py                  # Funciones auxiliares
│
├── models/                       # Modelos exportados
│   └── decision_tree_pipeline.pkl
│
├── booking_clean.csv             # Dataset limpio (ejemplo)
├── hotel_booking.csv             # Dataset original (ejemplo)
│
├── pipelines.ipynb               # Notebook de entrenamiento y exportación del modelo
├── main.py                       # Punto de entrada de la app Streamlit
├── requirements.txt              # Dependencias necesarias
└── README.md                     # Documentación del proyecto
```

## Ejecución

```bash
git clone https://github.com/wiloidrovo/Streamlit_App.git
cd tu-repositorio/Streamlit_App

pip install -r requirements.txt

streamlit run main.py
```
