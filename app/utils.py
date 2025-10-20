import streamlit as st

def apply_style():
    """
    Aplica el tema visual global de la aplicación (fondo oscuro + azul turquesa)
    con botones, radios y selectores interactivos.
    """
    css = """
        <style>
            /* General */
            .main {
                background-color: #0e1117;
                color: #fafafa;
                font-family: "Inter", sans-serif;
            }
            h1, h2, h3, h4 {
                color: #00bcd4;
                font-weight: 700 !important;
            }
            p, label {
                color: #e0e0e0;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1a2e, #16213e);
                color: white;
            }
            div[data-testid="stSidebarNav"] span {
                color: #fff;
                font-weight: 600;
            }

            /* Buttons */
            div.stButton > button {
                background: linear-gradient(90deg, #00bcd4, #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.6em 1.2em;
                font-weight: 600;
                transition: all 0.2s ease-in-out;
            }
            div.stButton > button:hover {
                background: linear-gradient(90deg, #26c6da, #42a5f5);
                transform: scale(1.03);
            }

            /* Radio buttons y multiselects */
            div[role="radiogroup"] > label > div,
            div[data-baseweb="tag"] {
                background-color: #1e293b;
                color: #e0e0e0;
                border-radius: 6px;
                padding: 0.5em 1em;
                margin: 3px;
                border: 1px solid #1e293b;
                transition: 0.2s;
            }
            div[role="radiogroup"] > label > div:hover {
                background-color: #00bcd4;
                color: white;
            }
            div[role="radiogroup"] > label[data-selected="true"] > div,
            div[role="radiogroup"] > label > input:checked + div {
                background: linear-gradient(90deg, #00bcd4, #2196f3);
                color: #ffffff !important;
                border: 1px solid #42a5f5;
                box-shadow: 0 0 8px rgba(0,188,212,0.4);
            }

            /* Selectbox */
            div[data-baseweb="select"] > div {
                background-color: #1e293b !important;
                color: #fafafa !important;
                border-radius: 6px;
                border: 1px solid #2a2a40 !important;
            }
            div[data-baseweb="select"] svg {
                fill: #00bcd4 !important;
            }

            /* Dataframe */
            div[data-testid="stDataFrame"] {
                border-radius: 8px;
                border: 1px solid #1f2937;
            }

            /* Mensajes */
            .stSuccess {
                background-color: #063d33;
                border-left: 6px solid #00e676;
                color: #d4f3e3;
            }
            .stInfo {
                background-color: #0b2239;
                border-left: 6px solid #42a5f5;
                color: #cfe8ff;
            }
            .stWarning {
                background-color: #332b00;
                border-left: 6px solid #ffeb3b;
                color: #fff8b0;
            }

            /* Subheaders */
            div.block-container h2 {
                border-left: 5px solid #00bcd4;
                padding-left: 10px;
                margin-top: 1em;
            }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)