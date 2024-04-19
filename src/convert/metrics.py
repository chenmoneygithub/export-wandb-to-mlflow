from src.config import MLFLOW_MAXIMUM_METRICS_PER_BATCH

from mlflow.entities import Metric

EXCLUDE_METRICS = ["_timestamp", "_step", "_run_time"]


def convert_wandb_experiment_metrics_to_mlflow(wandb_run, mlflow_client, mlflow_run_id):
    metric_history = wandb_run.scan_history()
    mlflow_metrics = []
    for _, row in enumerate(metric_history):
        timestamp = int(row["_timestamp"] * 1000)
        step = int(row["_step"])
        current_row_metrics = []
        for k, v in row.items():
            if k not in EXCLUDE_METRICS and isinstance(v, (int, float)):
                # Metrics must be either int or float.
                # There are other types such as str, dict or None, we should skip them.
                current_row_metrics.append(Metric(k.replace(".", "/"), v, timestamp, step))
        if (len(mlflow_metrics) + len(current_row_metrics)) >= MLFLOW_MAXIMUM_METRICS_PER_BATCH:
            mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics)
            mlflow_metrics = current_row_metrics
        else:
            mlflow_metrics.extend(current_row_metrics)

    # Clear up leftovers.
    mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics)
