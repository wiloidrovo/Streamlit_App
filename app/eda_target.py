import streamlit as st
import pandas as pd
import plotly.express as px

def analizar_vs_target(df, target_col):
    st.subheader(f"Análisis automático respecto a la variable objetivo: `{target_col}`")

    if target_col not in df.columns:
        st.warning("La variable objetivo no está en el dataset.")
        return

    # Verificar que target no tenga demasiadas clases
    n_clases = df[target_col].nunique()
    if n_clases > 20:
        st.error(f"La variable objetivo tiene {n_clases} clases únicas, demasiadas para un análisis automático.")
        return

    for col in df.columns:
        if col == target_col:
            continue

        st.markdown(f"#### {col} vs {target_col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            # Boxplot para numéricas
            fig = px.box(
                df, x=target_col, y=col, color=target_col,
                title=f"Distribución de {col} según {target_col}"
            )

            # Añadir histograma adicional
            hist = px.histogram(
                df, x=col, color=target_col, barmode="overlay",
                title=f"Histograma de {col} según {target_col}"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(hist, use_container_width=True)

        else:
            n_unicos = df[col].nunique()

            if n_unicos == 2:
                # Variable binaria -> barras con porcentajes
                cross_tab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
                cross_tab = cross_tab.reset_index().melt(id_vars=col, var_name=target_col, value_name="Porcentaje")

                fig = px.bar(
                    cross_tab, x=col, y="Porcentaje", color=target_col,
                    text="Porcentaje", title=f"Distribución porcentual de {col} según {target_col}",
                    barmode="group"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")

            else:
                # Variable categórica con más clases
                cross_tab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
                cross_tab = cross_tab.reset_index().melt(id_vars=col, var_name=target_col, value_name="Porcentaje")

                fig = px.bar(
                    cross_tab, x=col, y="Porcentaje", color=target_col,
                    title=f"Distribución porcentual de {col} según {target_col}",
                    barmode="stack"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Mostrar tabla de conteos y porcentajes
            st.write("Tabla de distribución:")
            tabla = pd.crosstab(df[col], df[target_col], margins=True, normalize="index") * 100
            st.dataframe(tabla.round(2))

# Función principal
def ejecutar_eda_target(df):
    target_col = st.session_state.get("target_col", None)

    if not target_col:
        st.info("No se ha definido una variable objetivo aún.")
        return

    analizar_vs_target(df, target_col)