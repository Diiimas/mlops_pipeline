"""
Módulo de ingeniería de características y preprocesamiento.

Este archivo contiene las funciones necesarias para preparar los datos
antes del entrenamiento, evitando fuga de información entre los conjuntos
de entrenamiento y prueba.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler
)

# Configuración general

TARGET_ORIGINAL = "Pago_atiempo"
TARGET_MODELO = "incumplimiento"

# Variable excluida por posible fuga de información
COLUMNAS_CON_RIESGO_LEAKAGE = [
    "puntaje"
]

# Columnas que no deben utilizarse como predictoras
COLUMNAS_EXCLUIDAS = [
    TARGET_ORIGINAL,
    TARGET_MODELO,
    *COLUMNAS_CON_RIESGO_LEAKAGE
]
def separar_variables_objetivo(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las variables predictoras y construye el objetivo de modelado.

    La variable incumplimiento toma el valor 1 cuando el cliente
    no pagó a tiempo y 0 cuando sí pagó a tiempo.

    Parameters
    ----------
    df : pd.DataFrame
        Conjunto de datos limpio.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Matriz de variables predictoras y variable objetivo.
    """

    if TARGET_ORIGINAL not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo '{TARGET_ORIGINAL}'."
        )

    datos = df.copy()

    valores_objetivo_validos = {0, 1}

    valores_encontrados = set(
        datos[TARGET_ORIGINAL]
        .dropna()
        .unique()
    )

    if not valores_encontrados.issubset(
        valores_objetivo_validos
    ):
        raise ValueError(
            f"'{TARGET_ORIGINAL}' debe contener únicamente 0 y 1."
        )

    if datos[TARGET_ORIGINAL].isna().any():
        raise ValueError(
            f"'{TARGET_ORIGINAL}' contiene valores faltantes."
        )

    y = (
        1 - datos[TARGET_ORIGINAL].astype(int)
    ).rename(TARGET_MODELO)

    columnas_a_eliminar = [
        columna
        for columna in COLUMNAS_EXCLUIDAS
        if columna in datos.columns
    ]

    X = datos.drop(
        columns=columnas_a_eliminar
    )

    return X, y
def identificar_tipos_columnas(
    X: pd.DataFrame
) -> Tuple[list[str], list[str]]:
    """
    Identifica las columnas numéricas y categóricas del conjunto predictor.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de variables predictoras.

    Returns
    -------
    Tuple[list[str], list[str]]
        Lista de columnas numéricas y lista de columnas categóricas.
    """

    if X.empty:
        raise ValueError(
            "La matriz de variables predictoras está vacía."
        )

    columnas_numericas = (
        X.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    columnas_categoricas = (
        X.select_dtypes(
            include=["object","string", "category", "bool"]
        )
        .columns
        .tolist()
    )

    columnas_no_clasificadas = [
        columna
        for columna in X.columns
        if columna not in columnas_numericas
        and columna not in columnas_categoricas
    ]

    if columnas_no_clasificadas:
        raise ValueError(
            "Existen columnas con tipos no contemplados: "
            f"{columnas_no_clasificadas}"
        )

    return columnas_numericas, columnas_categoricas
def construir_preprocesador(
    columnas_numericas: list[str],
    columnas_categoricas: list[str]
) -> ColumnTransformer:
    """
    Construye el preprocesador para variables numéricas y categóricas.

    Las variables numéricas se imputan mediante la mediana y se escalan
    con RobustScaler. Las variables categóricas se imputan con la categoría
    más frecuente y se codifican mediante One-Hot Encoding.

    Parameters
    ----------
    columnas_numericas : list[str]
        Nombres de las columnas numéricas.

    columnas_categoricas : list[str]
        Nombres de las columnas categóricas.

    Returns
    -------
    ColumnTransformer
        Preprocesador preparado para ajustarse con los datos de entrenamiento.
    """

    if not columnas_numericas and not columnas_categoricas:
        raise ValueError(
            "No se proporcionaron columnas para construir el preprocesador."
        )

    pipeline_numerico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False
                )
            ),
            (
                "escalador",
                RobustScaler()
            )
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "codificador",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformadores = []

    if columnas_numericas:
        transformadores.append(
            (
                "numericas",
                pipeline_numerico,
                columnas_numericas
            )
        )

    if columnas_categoricas:
        transformadores.append(
            (
                "categoricas",
                pipeline_categorico,
                columnas_categoricas
            )
        )

    preprocesador = ColumnTransformer(
        transformers=transformadores,
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocesador
# Columna cuyo valor cero representa información no disponible

COLUMNA_CERO_COMO_FALTANTE = "puntaje_datacredito"
COLUMNA_FECHA = "fecha_prestamo"


def convertir_cero_datacredito_en_faltante(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Convierte en valor faltante los ceros de puntaje_datacredito.

    Esta regla se aplica únicamente a puntaje_datacredito. Los ceros
    presentes en variables de saldo se conservan como valores válidos.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de variables predictoras.

    Returns
    -------
    pd.DataFrame
        Copia de la matriz con los ceros de puntaje_datacredito
        reemplazados por valores faltantes.
    """

    datos_transformados = X.copy()

    if COLUMNA_CERO_COMO_FALTANTE in datos_transformados.columns:
        datos_transformados[
            COLUMNA_CERO_COMO_FALTANTE
        ] = datos_transformados[
            COLUMNA_CERO_COMO_FALTANTE
        ].replace(0, np.nan)

    return datos_transformados
def crear_caracteristicas_temporales(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Extrae características útiles de la fecha del préstamo.

    La fecha original se elimina después de generar el año, mes,
    día de la semana y hora del préstamo.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de variables predictoras.

    Returns
    -------
    pd.DataFrame
        Copia de la matriz con las características temporales creadas.
    """

    datos_transformados = X.copy()

    if COLUMNA_FECHA not in datos_transformados.columns:
        return datos_transformados

    fecha = pd.to_datetime(
        datos_transformados[COLUMNA_FECHA],
        errors="coerce"
    )

    datos_transformados["prestamo_anio"] = fecha.dt.year
    datos_transformados["prestamo_mes"] = fecha.dt.month
    datos_transformados["prestamo_dia_semana"] = fecha.dt.dayofweek
    datos_transformados["prestamo_hora"] = fecha.dt.hour

    datos_transformados = datos_transformados.drop(
        columns=COLUMNA_FECHA
    )

    return datos_transformados
def aplicar_reglas_especificas(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Aplica las reglas específicas previas al preprocesamiento.
    """

    datos_transformados = (
        convertir_cero_datacredito_en_faltante(X)
    )

    datos_transformados = (
        crear_caracteristicas_temporales(datos_transformados)
    )

    return datos_transformados


def construir_pipeline_preprocesamiento(
    columnas_numericas: list[str],
    columnas_categoricas: list[str]
) -> Pipeline:
    """
    Construye el pipeline completo de preprocesamiento.

    Primero aplica las reglas específicas y luego realiza la imputación,
    el escalado y la codificación de las variables.
    """

    preprocesador = construir_preprocesador(
        columnas_numericas=columnas_numericas,
        columnas_categoricas=columnas_categoricas
    )

    transformador_reglas = FunctionTransformer(
        func=aplicar_reglas_especificas,
        validate=False
    )

    pipeline_preprocesamiento = Pipeline(
        steps=[
            (
                "reglas_especificas",
                transformador_reglas
            ),
            (
                "preprocesador",
                preprocesador
            )
        ]
    )

    return pipeline_preprocesamiento