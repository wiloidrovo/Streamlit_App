import streamlit as st
import os
import pandas as pd
import joblib


# ==============================================================
# 🧠 CUSTOMER CHURN PREDICTION PAGE
# ==============================================================

def prediction_page(df_clean):
    st.markdown(
        """
        <h1 style='text-align:center; color:#4B9CD3;'>📊 Customer Churn Prediction Dashboard</h1>
        <p style='text-align:center; font-size:18px;'>
            Use this interactive form to simulate a customer's profile and predict whether they are likely to churn.  
            Select your model and feature configuration below.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    available_models = [
        f for f in os.listdir(model_dir)
        if f.endswith(".pkl") and not f.startswith("model_metrics")
    ]

    if not available_models:
        st.error("❌ No trained models found in the `models/` directory.")
        return

    # Sidebar with model options
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_names = sorted(set(m.split("_")[0] for m in available_models))
        model_choice = st.selectbox("Select a model:", model_names)
        version_choice = st.radio("Select version:", ("all", "top"), horizontal=True)

    model_file = f"{model_choice}_{version_choice}.pkl"
    model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: `{model_file}`")
        return

    try:
        bundle = joblib.load(model_path)
        pipe = bundle["pipeline"]
        raw_features = bundle.get("raw_features", list(df_clean.columns))
        target_name = bundle.get("target_name", "Churn")

        st.sidebar.success(f"✅ Loaded: {model_choice.upper()} ({version_choice.upper()})")

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.markdown("---")

    # ----------------------------------------------------------
    # FORM SECTION
    # ----------------------------------------------------------
    st.markdown("<h3 style='color:#1E88E5;'>🧾 Enter Customer Details</h3>", unsafe_allow_html=True)

    user_input = {}

    # Responsive 2-column form
    with st.form("prediction_form"):
        cols = st.columns(2)
        i = 0
        for col_name in raw_features:
            if col_name == target_name or col_name not in df_clean.columns:
                continue

            serie = df_clean[col_name]
            with cols[i % 2]:  # alternate between 2 columns
                if pd.api.types.is_numeric_dtype(serie):
                    if pd.api.types.is_integer_dtype(serie):
                        user_input[col_name] = st.number_input(
                            f"**{col_name}**", value=int(serie.median()), step=1, format="%d"
                        )
                    else:
                        user_input[col_name] = st.number_input(
                            f"**{col_name}**", value=float(serie.median())
                        )
                else:
                    options = sorted([x for x in serie.dropna().unique().tolist() if x != ""])
                    if len(options) == 0:
                        user_input[col_name] = ""
                    elif len(options) == 2:
                        user_input[col_name] = st.radio(f"**{col_name}**", options, horizontal=True)
                    else:
                        user_input[col_name] = st.selectbox(f"**{col_name}**", options)
            i += 1

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 **Predict**", use_container_width=True)

    # ----------------------------------------------------------
    # PREDICTION RESULT
    # ----------------------------------------------------------
    if submitted:
        try:
            input_df = pd.DataFrame([user_input])
            pred = pipe.predict(input_df)[0]
            proba = (
                pipe.predict_proba(input_df)[0][1]
                if hasattr(pipe, "predict_proba")
                else None
            )

            st.markdown("---")
            st.markdown(
                "<h3 style='color:#43A047;'>🎯 Prediction Result</h3>",
                unsafe_allow_html=True
            )

            # Layout for prediction + probability side by side
            col1, col2 = st.columns([2, 1])
            with col1:
                if str(pred) == "Yes":
                    st.error("**Customer is likely to CHURN** 💔")
                else:
                    st.success("**Customer will NOT churn** 💚")

            with col2:
                if proba is not None:
                    st.metric("Churn Probability", f"{proba * 100:.2f}%")

            # Conditional recommendations
            if proba is not None:
                if proba > 0.7:
                    st.warning("⚠️ **High risk of churn detected.**\n\n💡 Suggested action: Offer a discount or loyalty incentive.")
                elif proba > 0.4:
                    st.info("🟡 **Medium churn risk.**\n\nConsider sending satisfaction surveys or retention offers.")
                else:
                    st.success("🟢 **Low churn risk.** Customer likely to remain loyal.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align:center; color:gray;'>Model used: <b>{model_choice.upper()}</b> | Version: <b>{version_choice.upper()}</b></p>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚙️ All models are pre-trained and loaded from /models. No retraining occurs inside this app.")
