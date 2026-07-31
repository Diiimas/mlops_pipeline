# MLOps Pipeline: predicción y monitoreo de riesgo crediticio

## Descripción del proyecto

Este proyecto desarrolla un flujo de Machine Learning para analizar solicitudes de crédito y predecir el riesgo de incumplimiento de los clientes.

El proceso incluye:

- Análisis exploratorio y limpieza de datos.
- Ingeniería y selección de características.
- Entrenamiento y evaluación de modelos supervisados.
- Selección del modelo con mejor desempeño.
- Monitoreo estadístico de data drift.
- Visualización de los resultados mediante una aplicación en Streamlit.

## Caso de negocio

Las instituciones financieras necesitan estimar el riesgo asociado a una solicitud de crédito antes de aprobarla. Un modelo predictivo permite identificar patrones relacionados con el incumplimiento y apoyar la toma de decisiones.

Sin embargo, las características de los solicitantes y de los créditos pueden cambiar con el tiempo. Estos cambios pueden reducir el desempeño de un modelo que fue entrenado con información histórica.

Por este motivo, el proyecto incorpora un sistema de monitoreo que compara un período histórico de referencia con un período más reciente y detecta cambios en la distribución de las variables.

## Estructura del proyecto

```text
mlops_pipeline/
├── data/
│   └── processed/
│       └── datos_limpios.csv
├── src/
│   ├── app.py
│   ├── cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── model_training_evaluation.py
├── .gitignore
├── README.md
└── requirements.txt