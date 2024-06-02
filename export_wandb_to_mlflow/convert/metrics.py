from mlflow.entities import Metric

from export_wandb_to_mlflow.config import MLFLOW_MAXIMUM_METRICS_PER_BATCH
from export_wandb_to_mlflow.dry_run_utils import log_metrics_dry_run

DEFAULT_EXCLUDE_METRICS = ["_timestamp", "_step", "_run_time"]


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


def _convert_wandb_metrics_to_mlflow_from_file(
    wandb_run,
    mlflow_client,
    mlflow_run,
    exclude_metrics=None,
):
    mlflow_metrics = []
    mlflow_run_id = mlflow_run.info.run_id
    for metrics in wandb_run.read_metrics():
        if exclude_metrics and metrics[0].key in exclude_metrics:
            # Skip the metrics that are in the exclude list.
            continue
        if (len(mlflow_metrics) + len(metrics)) >= MLFLOW_MAXIMUM_METRICS_PER_BATCH:
            space_left = MLFLOW_MAXIMUM_METRICS_PER_BATCH - len(mlflow_metrics)
            mlflow_metrics.extend(metrics[:space_left])
            mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)
            mlflow_metrics = metrics[space_left:]
        else:
            mlflow_metrics.extend(metrics)

    # Clear up the leftovers.
    mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)


def _convert_wandb_metrics_to_mlflow_from_server(
    wandb_run,
    mlflow_client,
    mlflow_run,
    exclude_metrics=None,
    dry_run=False,
    dry_run_save_dir=None,
):
    if not dry_run:
        mlflow_run_id = mlflow_run.info.run_id
    metric_history = wandb_run.scan_history()
    mlflow_metrics = []
    single_step_metrics = get_single_step_metrics(wandb_run)
    exclude_metrics = (exclude_metrics or []) + DEFAULT_EXCLUDE_METRICS

    batch_count = 0

    if dry_run:
        metrics_path = dry_run_save_dir / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        file_handlers = {}

    for _, row in enumerate(metric_history):
        timestamp = int(row["_timestamp"] * 1000)
        step = int(row["_step"])
        current_row_metrics = []

        for k, v in row.items():
            if k not in exclude_metrics and isinstance(v, (int, float)):
                # Metrics must be either int or float.
                # There are other types such as str, dict or None, we should skip them.
                if k in single_step_metrics:
                    current_row_metrics.append(Metric(k.replace(".", "/"), v, timestamp, step=0))
                else:
                    current_row_metrics.append(Metric(k.replace(".", "/"), v, timestamp, step))
        if (len(mlflow_metrics) + len(current_row_metrics)) >= MLFLOW_MAXIMUM_METRICS_PER_BATCH:
            batch_count += 1
            if dry_run:
                log_metrics_dry_run(mlflow_metrics, metrics_path, file_handlers)
            else:
                # Clear up leftovers.
                mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)
            mlflow_metrics = current_row_metrics
        else:
            mlflow_metrics.extend(current_row_metrics)

    if dry_run:
        log_metrics_dry_run(mlflow_metrics, metrics_path, file_handlers)
        for handler in file_handlers.values():
            # Close all file handlers.
            handler.close()
    else:
        # Clear up leftovers.
        mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)


def convert_wandb_metrics_to_mlflow(
    wandb_run,
    mlflow_client,
    mlflow_run,
    exclude_metrics=None,
    dry_run=False,
    resume_from_dry_run=False,
    dry_run_save_dir=None,
):
    """Convert Wandb experiment metrics to MLflow.

    This function converts Wandb experiment metrics for the given `wandb_run` to MLflow experiment
    metrics with the same metrics name. All logging happens asynchronously. This function has 3
    modes:
        1. Normal mode (default): read metrics from wandb server and directly write to MLflow.
        2. Dry run mode (dry_run=True): read metrics from wandb server and write to files.
        3. Resume from dry run mode (resume_from_dry_run=True): read metrics from files and write to
            MLflow.

    Args:
        wandb_run (wandb.sdk.wandb_run.Run): The Wandb run object.
        mlflow_client (mlflow.client.MlflowClient): The MLflow client.
        mlflow_run (mlflow.entities.Run): The MLflow run.
        exclude_metrics (List[str]): The list of metrics to exclude from migration.
        dry_run (bool): Whether to run in dry run mode, i.e., write metrics to files.
        resume_from_dry_run (bool): Whether to resume from dry run mode, i.e., read metrics from
            files and write to Mlflow. This argument is only valid when `dry_run` is False.
        dry_run_save_dir (pathlib.Path): The directory to the Mlflow run to save metrics.
    """
    if resume_from_dry_run:
        _convert_wandb_metrics_to_mlflow_from_file(
            wandb_run,
            mlflow_client,
            mlflow_run,
            exclude_metrics,
        )
    else:
        _convert_wandb_metrics_to_mlflow_from_server(
            wandb_run,
            mlflow_client,
            mlflow_run,
            exclude_metrics,
            dry_run,
            dry_run_save_dir,
        )
