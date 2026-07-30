"""
Entrenamiento y evaluación de modelos supervisados.

Este módulo centraliza la construcción, el entrenamiento y la evaluación
de modelos de clasificación destinados a predecir el incumplimiento
de pago.
"""

from typing import Dict, Tuple

import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from xgboost import XGBClassifier


def construir_modelos(
    random_state: int = 42
) -> Dict[str, ClassifierMixin]:
    """
    Construye los modelos de clasificación que se utilizarán
    en la comparación inicial.

    Parameters
    ----------
    random_state : int, default=42
        Semilla utilizada para garantizar la reproducibilidad
        de los modelos que incluyen componentes aleatorios.

    Returns
    -------
    Dict[str, ClassifierMixin]
        Diccionario que relaciona el nombre de cada modelo con
        su estimador sin entrenar.
    """

    modelos = {
        "Regresión logística": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }

    return modelos


def calcular_scale_pos_weight(
    y_train: pd.Series
) -> float:
    """
    Calcula el peso de la clase positiva para XGBoost.

    El valor se obtiene dividiendo la cantidad de observaciones de la
    clase negativa por la cantidad de observaciones de la clase positiva.

    Parameters
    ----------
    y_train : pd.Series
        Variable objetivo del conjunto de entrenamiento.

    Returns
    -------
    float
        Peso correspondiente a la clase positiva.
    """

    cantidad_negativos = (y_train == 0).sum()
    cantidad_positivos = (y_train == 1).sum()

    if cantidad_positivos == 0:
        raise ValueError(
            "y_train no contiene observaciones de la clase positiva."
        )

    scale_pos_weight = cantidad_negativos / cantidad_positivos

    return float(scale_pos_weight)


def entrenar_evaluar_modelo(
    modelo: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[ClassifierMixin, Dict[str, float]]:
    """
    Entrena un modelo de clasificación y calcula sus métricas.

    En el caso de XGBoost, calcula automáticamente el parámetro
    scale_pos_weight utilizando únicamente la distribución del
    conjunto de entrenamiento.

    Parameters
    ----------
    modelo : ClassifierMixin
        Modelo de clasificación sin entrenar.

    X_train : pd.DataFrame
        Variables predictoras del conjunto de entrenamiento.

    y_train : pd.Series
        Variable objetivo del conjunto de entrenamiento.

    X_test : pd.DataFrame
        Variables predictoras del conjunto de evaluación.

    y_test : pd.Series
        Variable objetivo del conjunto de evaluación.

    Returns
    -------
    Tuple[ClassifierMixin, Dict[str, float]]
        Modelo entrenado y diccionario con las métricas obtenidas.
    """

    # Configurar el balance de clases específico de XGBoost
    if isinstance(modelo, XGBClassifier):
        modelo.set_params(
            scale_pos_weight=calcular_scale_pos_weight(y_train)
        )

    # Entrenar el modelo exclusivamente con el conjunto de entrenamiento
    modelo.fit(X_train, y_train)

    # Obtener clases predichas y probabilidades de la clase positiva
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Calcular métricas de evaluación
    metricas = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return modelo, metricas


def comparar_modelos(
    modelos: Dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, ClassifierMixin], pd.DataFrame]:
    """
    Entrena y evalúa varios modelos utilizando los mismos conjuntos de datos.

    Parameters
    ----------
    modelos : Dict[str, ClassifierMixin]
        Diccionario con los nombres y estimadores sin entrenar.

    X_train : pd.DataFrame
        Variables predictoras del conjunto de entrenamiento.

    y_train : pd.Series
        Variable objetivo del conjunto de entrenamiento.

    X_test : pd.DataFrame
        Variables predictoras del conjunto de evaluación.

    y_test : pd.Series
        Variable objetivo del conjunto de evaluación.

    Returns
    -------
    Tuple[Dict[str, ClassifierMixin], pd.DataFrame]
        Diccionario con los modelos entrenados y tabla comparativa
        ordenada de mayor a menor ROC-AUC.
    """

    modelos_entrenados = {}
    resultados = []

    for nombre_modelo, modelo in modelos.items():
        modelo_entrenado, metricas = entrenar_evaluar_modelo(
            modelo=modelo,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        modelos_entrenados[nombre_modelo] = modelo_entrenado

        resultados.append({
            "Modelo": nombre_modelo,
            **metricas
        })

    tabla_resultados = pd.DataFrame(resultados)

    tabla_resultados = (
        tabla_resultados
        .sort_values(
            by="ROC-AUC",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return modelos_entrenados, tabla_resultados


def obtener_evaluacion_detallada(
    modelo: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera la matriz de confusión y el reporte de clasificación
    de un modelo previamente entrenado.

    Parameters
    ----------
    modelo : ClassifierMixin
        Modelo de clasificación previamente entrenado.

    X_test : pd.DataFrame
        Variables predictoras del conjunto de evaluación.

    y_test : pd.Series
        Variable objetivo real del conjunto de evaluación.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Matriz de confusión y reporte de clasificación
        expresados como DataFrames.
    """

    y_pred = modelo.predict(X_test)

    matriz = confusion_matrix(
        y_test,
        y_pred
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=["Real: no incumple", "Real: incumple"],
        columns=["Predicho: no incumple", "Predicho: incumple"]
    )

    reporte = classification_report(
        y_test,
        y_pred,
        target_names=["No incumple", "Incumple"],
        output_dict=True,
        zero_division=0
    )

    reporte_df = (
        pd.DataFrame(reporte)
        .transpose()
    )

    return matriz_df, reporte_df