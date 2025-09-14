# Streamlit EDA & ML App

Esta aplicación interactiva desarrollada en **Python + Streamlit** permite realizar **Análisis Exploratorio de Datos (EDA)** de manera intuitiva y flexible.  
El objetivo es **cargar datos, limpiarlos, analizarlos y visualizar tendencias clave**, preparando el dataset para la construcción de modelos de Machine Learning.

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
- Detección de desbalance de clases en el target.

### **Conclusiones**

- Recuadro al final del análisis con **insights clave extraídos del dataset**, listos para documentar hallazgos o continuar con la fase de modelado.

---

## Estructura del proyecto

Streamlit_App/
│── app/
│ ├── eda.py # Lógica principal del EDA (imputaciones, limpieza, columnas)
│ ├── eda_2.py # KPIs y generador de gráficos
│ ├── eda_target.py # Análisis automático con respecto a la variable objetivo
│── main.py # Punto de entrada de la aplicación Streamlit
│── requirements.txt # Dependencias necesarias
│── README.md # Documentación del proyecto

## Ejecución

### Clonar el repositorio

```bash
git clone https://github.com/wiloidrovo/Streamlit_App.git
cd tu-repositorio/Streamlit_App

pip install -r requirements.txt

streamlit run main.py
```
