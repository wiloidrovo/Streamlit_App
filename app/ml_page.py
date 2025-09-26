import streamlit as st
import os
import pandas as pd
import joblib
from app.pipelines_transf import DataFramePreparer

def prediction_page(df_clean):
    st.subheader("Prediction")

    # Ruta del modelo
    model_path = os.path.join("models", "decision_tree_pipeline.pkl")

    # Cargar modelo entrenado
    try:
        bundle = joblib.load(model_path)
        prep = bundle["pipeline"]
        model = bundle["model"]
        st.sidebar.success("Model and pipeline loaded successfully")
    except Exception as e:
        st.error(f"Model or pipeline not found. Please train and export them first. ({e})")
        return

    # Crear formulario dinámico
    st.markdown("Fill in the details below to get a prediction:")

    user_input = {}
    with st.form("prediction_form"):
        for col in df_clean.columns:
            if col == "is_canceled":  # target → no pedirlo
                continue

            if pd.api.types.is_integer_dtype(df_clean[col]):
                # Entradas enteras
                user_input[col] = st.number_input(
                    f"{col}",
                    value=int(df_clean[col].median()),
                    step=1,
                    format="%d"
                )
            elif pd.api.types.is_float_dtype(df_clean[col]):
                # Entradas flotantes
                user_input[col] = st.number_input(
                    f"{col}",
                    value=float(df_clean[col].median())
                )
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                # Entradas de fecha
                user_input[col] = st.date_input(f"{col}")
            else:
                # Categóricas → selectbox con opciones únicas
                options = df_clean[col].dropna().unique().tolist()
                user_input[col] = st.selectbox(f"{col}", options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])

        # Preprocesar con pipeline
        input_prep = prep.transform(input_df)

        # Hacer predicción
        pred = model.predict(input_prep)[0]
        proba = model.predict_proba(input_prep)[0][1]

        st.markdown(f"### Prediction: {'Canceled' if pred == 1 else 'Not Canceled'}")
        st.markdown(f"**Probability of cancellation:** {proba * 100:.2f}%")
