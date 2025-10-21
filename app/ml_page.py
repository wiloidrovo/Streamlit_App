import streamlit as st
import os
import pandas as pd
import joblib
from app.utils import apply_style

apply_style()

def prediction_page(df_clean):
    st.markdown(
        """
        <div style='text-align:center; padding: 1.5em 0;'>
            <h1 style='color:#00bcd4; font-weight:700; letter-spacing:0.5px;'>
                🤖 Customer Churn Prediction Dashboard
            </h1>
            <p style='color:#b0bec5; font-size:17px; margin-top:8px;'>
                Simulate a customer’s profile and predict churn probability in real time.<br>
                Select your preferred <b>model</b> and <b>feature configuration</b> below.
            </p>
        </div>
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
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("<hr>", unsafe_allow_html=True)

        model_names = sorted(set(m.split("_")[0] for m in available_models))
        model_choice = st.selectbox("Select Model:", model_names)
        version_choice = st.radio("Select Version:", ("all", "top"), horizontal=True)

        st.markdown("<hr>", unsafe_allow_html=True)

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
    # FORM SECTION (Glass Card)
    # ----------------------------------------------------------
    st.markdown(
        """
        <div style='
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 188, 212, 0.25);
            border-radius: 12px;
            padding: 2em;
            margin-top: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(6px);
        '>
        <h3 style='color:#00bcd4; margin-bottom:10px;'>🧾 Enter Customer Details</h3>
        <p style='color:#9aa7b3; margin-top:-5px;'>Fill in the customer’s profile information to generate a prediction.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_input = {}

    with st.form("prediction_form"):
        cols = st.columns(2)
        i = 0
        for col_name in raw_features:
            if col_name == target_name or col_name not in df_clean.columns:
                continue

            serie = df_clean[col_name]
            with cols[i % 2]:
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
        submitted = st.form_submit_button("🔮 **Predict Now**", use_container_width=True)

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
                """
                <h3 style='color:#00e676; text-align:center;'>
                    🎯 Prediction Result
                </h3>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                if str(pred) == "Yes":
                    st.markdown(
                        """
                        <div style='
                            background: rgba(255, 0, 0, 0.07);
                            border-left: 6px solid #ef5350;
                            border-radius: 8px;
                            padding: 1.2em;
                            margin-top: 10px;
                        '>
                            <h4 style='color:#ef5350; margin:0;'>💔 Customer is likely to CHURN</h4>
                            <p style='color:#ef9a9a; margin:5px 0 0 0;'>Immediate attention required.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style='
                            background: rgba(0, 255, 127, 0.07);
                            border-left: 6px solid #00e676;
                            border-radius: 8px;
                            padding: 1.2em;
                            margin-top: 10px;
                        '>
                            <h4 style='color:#00e676; margin:0;'>💚 Customer will NOT churn</h4>
                            <p style='color:#b9f6ca; margin:5px 0 0 0;'>Customer likely to remain loyal.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col2:
                if proba is not None:
                    st.metric("Churn Probability", f"{proba * 100:.2f}%", delta=None)

            # Recommendations
            if proba is not None:
                if proba > 0.7:
                    st.warning(
                        "⚠️ **High risk of churn detected.**\n\n💡 Suggested action: Offer a discount or loyalty incentive."
                    )
                elif proba > 0.4:
                    st.info(
                        "🟡 **Medium churn risk.**\n\nConsider sending satisfaction surveys or retention offers."
                    )
                else:
                    st.success(
                        "🟢 **Low churn risk.** Customer likely to remain loyal."
                    )

            st.markdown(
                f"""
                <div style='text-align:center; margin-top:20px; color:#90a4ae;'>
                    <small>Model used: <b style='color:#00bcd4;'>{model_choice.upper()}</b> |
                    Version: <b style='color:#00bcd4;'>{version_choice.upper()}</b></small>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚙️ All models are pre-trained and loaded from /models. No retraining occurs inside this app.")