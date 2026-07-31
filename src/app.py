"""
Aplicación Streamlit para visualizar el monitoreo de data drift.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# Permite importar model_monitoring.py desde la carpeta src
RUTA_SRC = Path(__file__).resolve().parent
sys.path.append(str(RUTA_SRC))

from model_monitoring import detectar_data_drift


st.set_page_config(
    page_title="Monitoreo de Data Drift",
    page_icon="📊",
    layout="wide"
)

st.title("Monitoreo de Data Drift")
st.write(
    "Comparación entre un período histórico de referencia "
    "y el período actual de solicitudes de crédito."
)


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """Carga el dataset limpio utilizado por la aplicación."""

    posibles_rutas = [
        RUTA_SRC.parent / "data" / "processed" / "datos_limpios.csv",
        RUTA_SRC.parent / "data" / "processed" / "df_limpio.csv",
    ]

    for ruta in posibles_rutas:
        if ruta.exists():
            return pd.read_csv(ruta)

    raise FileNotFoundError(
        "No se encontró el dataset limpio en data/processed."
    )


try:
    df = cargar_datos()

    df["fecha_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"],
        errors="coerce"
    )

    df = (
        df.sort_values("fecha_prestamo")
        .reset_index(drop=True)
    )

    punto_corte = int(len(df) * 0.70)

    datos_referencia = df.iloc[:punto_corte].copy()
    datos_actuales = df.iloc[punto_corte:].copy()

    variables_numericas = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente"
    ]

    variables_categoricas = [
        "tipo_credito",
        "tipo_laboral"
    ]

    resultados = detectar_data_drift(
        datos_referencia=datos_referencia,
        datos_actuales=datos_actuales,
        variables_numericas=variables_numericas,
        variables_categoricas=variables_categoricas
    )

    cantidad_alertas = (
        resultados["Estado"]
        .isin(["Alerta", "Crítico"])
        .sum()
    )

    columna_1, columna_2, columna_3 = st.columns(3)

    columna_1.metric(
        "Registros de referencia",
        len(datos_referencia)
    )

    columna_2.metric(
        "Registros actuales",
        len(datos_actuales)
    )

    columna_3.metric(
        "Variables con alerta",
        int(cantidad_alertas)
    )

    if cantidad_alertas > 0:
        st.warning(
            "Se detectaron variables con posibles cambios de distribución. "
            "Se recomienda mantener el modelo bajo observación."
        )
    else:
        st.success(
            "No se detectaron cambios relevantes en las variables analizadas."
        )

    st.subheader("Resultados del monitoreo")

    st.dataframe(
        resultados.style.format({
            "KS": "{:.4f}",
            "PSI": "{:.4f}",
            "Chi-cuadrado": "{:.4f}",
            "p_valor": "{:.4f}"
        }, na_rep="-"),
        use_container_width=True
    )

    st.subheader("Comparación de una variable numérica")

    variable_seleccionada = st.selectbox(
        "Variable",
        variables_numericas
    )

    comparacion = pd.DataFrame({
        "Referencia": datos_referencia[
            variable_seleccionada
        ].reset_index(drop=True),
        "Actual": datos_actuales[
            variable_seleccionada
        ].reset_index(drop=True)
    })

    st.line_chart(comparacion)

    st.caption(
        "Para las variables numéricas se utilizan KS y PSI. "
        "Para las variables categóricas se utiliza chi-cuadrado."
    )

except FileNotFoundError as error:
    st.error(str(error))

except Exception as error:
    st.error(f"No fue posible ejecutar el monitoreo: {error}")