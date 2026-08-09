#!/usr/bin/env python3
"""One-time script: register the pre-MLflow smoke results as baseline runs.

Run ONCE after MLflow is installed. Safe to keep in git as documentation
of where the baseline numbers came from (smoke training, 10 epochs, 2026-08).
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("war-damage-smoke")

for name, m50, m5095 in [("yolo12n", 0.601, 0.382), ("yolo26n", 0.624, 0.397)]:
    with mlflow.start_run(run_name=f"baseline-{name}"):
        mlflow.log_param("model", name)
        mlflow.log_param("epochs", 10)
        mlflow.log_param("source", "pre-mlflow smoke run 2026-08")
        mlflow.log_metric("mAP50", m50)
        mlflow.log_metric("mAP50_95", m5095)