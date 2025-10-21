# Streamlit Telco Churn App

Esta aplicación interactiva desarrollada en **Python + Streamlit** permite realizar **Análisis Exploratorio de Datos (EDA)**, **evaluar modelos de Machine Learning** y **analizar el impacto empresarial** del _Customer Churn_ (fuga de clientes).  
El proyecto está orientado a simular cómo una empresa de telecomunicaciones podría **predecir y reducir la pérdida de clientes** mediante inteligencia artificial.

---

## 🎯 Objetivo

El sistema permite cargar el dataset de **Telco Customer Churn**, realizar un análisis exploratorio interactivo, visualizar relaciones clave, generar un dataset limpio y usarlo para realizar inferencias con modelos previamente entrenados.  
Adicionalmente, el usuario puede comparar versiones de modelos, visualizar sus métricas, interpretar sus predicciones y evaluar el impacto financiero asociado al churn.

---

## 🚀 Características principales

### **1. Carga y Preprocesamiento de Datos**

- Subida de datasets en formato `.csv` o `.xlsx`.
- Conversión automática de columnas mal tipadas (numéricas y categóricas).
- Imputación de valores nulos:
  - Media, Mediana, Moda.
  - Valor constante o eliminación de filas.
- Eliminación y recuperación de columnas (persistente mediante `session_state`).
- Vista previa del dataset actualizado e información general:
  - Conteo de valores nulos/no nulos.
  - Tipos de datos detectados.

---

### **2. Exploratory Data Analysis (EDA)**

- Generación automática de KPIs:
  - Total de filas y columnas.
  - Porcentaje de valores nulos.
  - Promedio de variables numéricas.
  - Distribución de la variable objetivo (**Churn**).
- **Visualizaciones Interactivas**:
  - Histogramas, boxplots, gráficos de barras y dispersión.
  - Heatmap de correlaciones.
  - Análisis automático frente al target (segmentación de variables).
- Detección de **desbalance de clases** en el target.

---

### **3. Modelos de Machine Learning en Producción**

- Integración de **pipelines** y **transformadores personalizados** (`DataFramePreparer`, `CustomOneHotEncoder`).
- Modelos entrenados en Jupyter Notebook (`pipelines.ipynb`), exportados en formato `.pkl`.
- **Cinco modelos disponibles**:
  - Bagging (Random Forest)
  - Boosting (CatBoost)
  - Stacking Classifier
  - Voting Classifier
  - Decision Tree (baseline)
- Cada modelo tiene **dos versiones**:
  - `all` → entrenado con todas las características.
  - `top` → entrenado con las 15 mejores variables (según _mutual information_).
- Carga dinámica de modelos preentrenados en Streamlit (sin reentrenamiento).

---

### **4. Página de Inferencia (Prediction Page)**

- Formulario interactivo donde el usuario introduce características del cliente.
- El sistema transforma los datos automáticamente mediante el pipeline correspondiente.
- Predice:
  - Clase: `Churn` / `No Churn`.
  - Probabilidad asociada.
- Recomendación automática basada en nivel de riesgo:
  - 🔴 **Alto riesgo:** sugerir descuentos o incentivos de retención.
  - 🟡 **Riesgo medio:** realizar seguimiento o encuestas de satisfacción.
  - 🟢 **Bajo riesgo:** cliente estable.

---

### **5. Dashboard Analítico**

- Comparación de **métricas (F1, AUC, Accuracy, Precision, Recall)** entre modelos y versiones.
- **Matriz de confusión interactiva** (validación).
- **Visualización de importancia de características** (feature importance).
- Panel dinámico para explorar resultados y rendimiento de cada modelo.

---

### **6. Business Impact Analysis**

- Simulación del **impacto económico del churn**:
  - % de clientes en riesgo.
  - Revenue total en riesgo.
  - Potenciales ahorros con campañas de retención.
- Visualización de:
  - Distribución de probabilidades de churn.
  - Factores más influyentes en la pérdida de clientes.
  - Tabla de recomendaciones automáticas por cliente.
- Parámetros configurables:
  - Umbral de probabilidad de churn.
  - Efectividad de la campaña.
  - Costo por usuario retenido.

---

## 🧩 Estructura del Proyecto

```bash
Streamlit_App/
│
├── app/                          # Módulos principales de la aplicación
│   ├── carga_datos.py            # Carga y previsualización de datasets
│   ├── eda.py                    # Lógica principal del EDA (limpieza y transformaciones)
│   ├── eda_2.py                  # KPIs y visualizaciones interactivas
│   ├── eda_target.py             # Análisis automático respecto al target
│   ├── pipelines_transf.py       # Pipelines y transformadores personalizados
│   ├── ml_page.py                # Formulario de inferencia y predicción
│   ├── dashboard.py              # Dashboard con métricas, matrices y feature importances
│   ├── business_impact.py        # Análisis de impacto financiero del churn
│   └── utils.py                  # CSS para estilos
│
├── models/                       # Modelos y métricas exportadas
│   ├── *_all.pkl                 # Versiones con todas las variables
│   ├── *_top.pkl                 # Versiones con top features
│   └── model_metrics_summary.csv # Resumen global de métricas
│
├── pipelines.ipynb               # Notebook de entrenamiento y exportación de modelos
├── cleaned_dataset.csv           # Dataset limpio generado en la app
├── main.py                       # Punto de entrada principal de la aplicación Streamlit
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
