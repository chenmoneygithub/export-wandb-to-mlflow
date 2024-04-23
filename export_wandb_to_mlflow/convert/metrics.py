from mlflow.entities import Metric

from export_wandb_to_mlflow.config import MLFLOW_MAXIMUM_METRICS_PER_BATCH

EXCLUDE_METRICS = ["_timestamp", "_step", "_run_time"]


def get_single_step_metrics(wandb_run):
    """Get the metrics that are logged only once.

    This function returns the metrics that are logged only once in the Wandb run, which is
    important because wandb returns a random step for these only-logged-runs metrics. When logging
    to MLflow, we want to render them as bar plots, which requires us not to set the step field.

    Args:
        wandb_run: The Wandb run object.
    """
    sample_history = wandb_run.history()

    # Find columns with exactly one non-None value.
    single_non_none = sample_history.notna().sum() == 1

    # Get the columns that have exactly one non-None value.
    return single_non_none[single_non_none].index.tolist()


def convert_wandb_experiment_metrics_to_mlflow(wandb_run, mlflow_client, mlflow_run_id):
    metric_history = wandb_run.scan_history()
    mlflow_metrics = []
    single_step_metrics = get_single_step_metrics(wandb_run)
    for _, row in enumerate(metric_history):
        timestamp = int(row["_timestamp"] * 1000)
        step = int(row["_step"])
        current_row_metrics = []

        for k, v in row.items():
            if k not in EXCLUDE_METRICS and isinstance(v, (int, float)):
                # Metrics must be either int or float.
                # There are other types such as str, dict or None, we should skip them.
                if k in single_step_metrics:
                    current_row_metrics.append(Metric(k.replace(".", "/"), v, timestamp, step=0))
                else:
                    current_row_metrics.append(Metric(k.replace(".", "/"), v, timestamp, step))
        if (len(mlflow_metrics) + len(current_row_metrics)) >= MLFLOW_MAXIMUM_METRICS_PER_BATCH:
            mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)
            mlflow_metrics = current_row_metrics
        else:
            mlflow_metrics.extend(current_row_metrics)

    # Clear up leftovers.
    mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)
