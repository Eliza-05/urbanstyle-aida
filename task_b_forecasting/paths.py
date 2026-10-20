"""Tarea B (Elizabeth): carpetas de salida en outputs/task_b/ (ver outputs/task_b/README.md)."""
from common import config

OUT_DIR = config.OUTPUTS_DIR / "task_b"
EXPORT_DIR = OUT_DIR / "export"      # predicciones exportadas (bonus CSV)
METRICS_DIR = OUT_DIR / "metrics"    # métricas en prueba, bootstrap, residuos
TRAINING_DIR = OUT_DIR / "training"  # curvas de pérdida, búsqueda de hiperparámetros, semillas
EDA_DIR = OUT_DIR / "eda"            # análisis exploratorio
FIGURES_DIR = OUT_DIR / "figures"    # figuras para el informe y capturas de pantalla

PREDICTIONS_CSV = EXPORT_DIR / "predictions.csv"              # estándar: coma separadora, punto decimal
PREDICTIONS_EXCEL_CSV = EXPORT_DIR / "predictions_excel.csv"  # Excel en español: ";" y coma decimal
NEXT_WEEK_CSV = EXPORT_DIR / "next_week.csv"
METRICS_CSV = METRICS_DIR / "metrics.csv"
BOOTSTRAP_CSV = METRICS_DIR / "bootstrap.csv"
