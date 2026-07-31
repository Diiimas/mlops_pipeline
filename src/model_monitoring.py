"""
Monitoreo y detección de data drift.

Este módulo compara un conjunto de datos de referencia con datos actuales
mediante pruebas estadísticas para variables numéricas y categóricas.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, ks_2samp


def calcular_ks(
    referencia: pd.Series,
    actual: pd.Series
) -> Dict[str, float]:
    """
    Calcula el test de Kolmogorov-Smirnov entre dos muestras numéricas.

    Un p-valor inferior a 0.05 indica evidencia estadística
    de un cambio en la distribución.
    """

    referencia_limpia = referencia.dropna()
    actual_limpia = actual.dropna()

    estadistico, p_valor = ks_2samp(
        referencia_limpia,
        actual_limpia
    )

    return {
        "KS": float(estadistico),
        "p_valor": float(p_valor)
    }


def calcular_psi(
    referencia: pd.Series,
    actual: pd.Series,
    bins: int = 10
) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos muestras numéricas.

    Interpretación:
    - PSI < 0.10: sin drift significativo.
    - PSI entre 0.10 y 0.25: alerta.
    - PSI > 0.25: drift crítico.
    """

    referencia_limpia = referencia.dropna().to_numpy()
    actual_limpia = actual.dropna().to_numpy()

    limites = np.quantile(
        referencia_limpia,
        np.linspace(0, 1, bins + 1)
    )

    limites = np.unique(limites)

    if len(limites) < 2:
        return 0.0

    limites[0] = -np.inf
    limites[-1] = np.inf

    referencia_conteos, _ = np.histogram(
        referencia_limpia,
        bins=limites
    )

    actual_conteos, _ = np.histogram(
        actual_limpia,
        bins=limites
    )

    referencia_proporciones = referencia_conteos / len(
        referencia_limpia
    )

    actual_proporciones = actual_conteos / len(
        actual_limpia
    )

    minimo = 1e-6

    referencia_proporciones = np.clip(
        referencia_proporciones,
        minimo,
        None
    )

    actual_proporciones = np.clip(
        actual_proporciones,
        minimo,
        None
    )

    psi = np.sum(
        (actual_proporciones - referencia_proporciones)
        * np.log(
            actual_proporciones / referencia_proporciones
        )
    )

    return float(psi)


def calcular_chi_cuadrado(
    referencia: pd.Series,
    actual: pd.Series
) -> Dict[str, float]:
    """
    Aplica la prueba chi-cuadrado para comparar la distribución
    de una variable categórica entre los datos de referencia y actuales.

    Un p-valor inferior a 0.05 indica evidencia estadística
    de un cambio en la distribución.
    """

    referencia_limpia = referencia.dropna().astype(str)
    actual_limpia = actual.dropna().astype(str)

    categorias = sorted(
        set(referencia_limpia.unique())
        | set(actual_limpia.unique())
    )

    tabla_contingencia = pd.DataFrame({
        "Referencia": referencia_limpia
        .value_counts()
        .reindex(categorias, fill_value=0),

        "Actual": actual_limpia
        .value_counts()
        .reindex(categorias, fill_value=0)
    }).T

    estadistico, p_valor, _, _ = chi2_contingency(
        tabla_contingencia
    )

    return {
        "Chi-cuadrado": float(estadistico),
        "p_valor": float(p_valor)
    }


def detectar_data_drift(
    datos_referencia: pd.DataFrame,
    datos_actuales: pd.DataFrame,
    variables_numericas: List[str],
    variables_categoricas: List[str]
) -> pd.DataFrame:
    """
    Evalúa el data drift de variables numéricas y categóricas
    y devuelve una tabla consolidada de resultados.
    """

    resultados = []

    # Evaluar variables numéricas con KS y PSI
    for variable in variables_numericas:
        resultado_ks = calcular_ks(
            datos_referencia[variable],
            datos_actuales[variable]
        )

        psi = calcular_psi(
            datos_referencia[variable],
            datos_actuales[variable]
        )

        if psi > 0.25:
            estado = "Crítico"
        elif psi >= 0.10:
            estado = "Alerta"
        else:
            estado = "Sin drift"

        resultados.append({
            "Variable": variable,
            "Tipo": "Numérica",
            "KS": resultado_ks["KS"],
            "PSI": psi,
            "Chi-cuadrado": np.nan,
            "p_valor": resultado_ks["p_valor"],
            "Estado": estado
        })

    # Evaluar variables categóricas con chi-cuadrado
    for variable in variables_categoricas:
        resultado_chi = calcular_chi_cuadrado(
            datos_referencia[variable],
            datos_actuales[variable]
        )

        estado = (
            "Alerta"
            if resultado_chi["p_valor"] < 0.05
            else "Sin drift"
        )

        resultados.append({
            "Variable": variable,
            "Tipo": "Categórica",
            "KS": np.nan,
            "PSI": np.nan,
            "Chi-cuadrado": resultado_chi["Chi-cuadrado"],
            "p_valor": resultado_chi["p_valor"],
            "Estado": estado
        })

    return pd.DataFrame(resultados)