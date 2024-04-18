import wandb
import mlflow
import json
from functools import partial
import re
import os
from contextlib import contextmanager
import uuid

wandb.login()
api = wandb.Api()


global_system_metrics_logging_counts = 0
global_training_metrics_logging_counts = 0


def convert_wandb_config_to_mlflow_params(run):
    converted_config = {
        k: json.dumps(v) if isinstance(v, dict) else v for k, v in run.config.items()
    }
    mlflow.log_params(converted_config, synchronous=False)


gpu_metrics_mapping = {
    "system.gpu.(\d+).memory$": "system/gpu_{i}_utilization_percentage",
    "system.gpu.(\d+).memoryAllocated$": "system/gpu_{i}_memory_usage_percentage",
    "system.gpu.(\d+).memoryAllocatedBytes": "system/gpu_{i}_memory_usage_megabytes",
}


def convert_gpu_metrics_to_mlflow(row, step):
    metrics = []

    for k, v in row.items():
        for wandb_key, mlflow_key in gpu_metrics_mapping.items():
            match_result = re.search(wandb_key, k)

            # Check if there was a match
            if match_result and v is not None:
                gpu_index = match_result.group(1)
                mlflow_key = mlflow_key.format(i=gpu_index)
                if "memoryAllocatedBytes" in wandb_key:
                    metrics.append(
                        mlflow.entities.Metric(
                            mlflow_key, round(v * 1.0 / 1e6, 2), step, step
                        )
                    )  # Convert bytes to MB.
                else:
                    metrics.append(mlflow.entities.Metric(mlflow_key, v, step, step))
    return metrics


def convert_bytes_to_mb(row, key):
    return None if row.get(key, None) is None else round(row[key] / 1000000.0, 2)


def convert_gb_to_mb(row, key):
    return None if row.get(key, None) is None else round(row[key] * 1000.0, 2)


def convert_wandb_system_metrics_to_mlflow(run, mlflow_client, mlflow_run_id):
    system_metrics = run.history(stream="system")

    system_metrics_mapping = {
        "system/cpu_utilization_percentage": "system.cpu",
        "system/disk_usage_megabytes": partial(
            convert_gb_to_mb, key="system.disk.\\.usageGB"
        ),
        "system/disk_usage_percentage": "system.disk.\\.usagePercent",
        "system/system_memory_usage_megabytes": "system.proc.memory.rssMB",
        "system/system_memory_usage_percentage": "system.memory",
        "system/network_receive_megabytes": partial(
            convert_bytes_to_mb, key="system.network.recv"
        ),
        "system/network_transmit_megabytes": partial(
            convert_bytes_to_mb, key="system.network.sent"
        ),
    }

    mlflow_system_metrics = []
    global global_system_metrics_logging_counts

    for index, row in system_metrics.iterrows():
        row_dict = row.to_dict()

        gpu_metrics = convert_gpu_metrics_to_mlflow(row_dict, step=index)
        current_row_metrics = []
        for mlflow_key, wandb_handler in system_metrics_mapping.items():
            val = (
                wandb_handler(row)
                if callable(wandb_handler)
                else row_dict.get(wandb_handler, None)
            )
            if val:
                current_row_metrics.append(
                    mlflow.entities.Metric(mlflow_key, val, index, index)
                )

        if (
            len(mlflow_system_metrics) + len(gpu_metrics) + len(current_row_metrics)
        ) >= 1000:
            global_system_metrics_logging_counts += 1
            mlflow_client.log_batch(
                mlflow_run_id, metrics=mlflow_system_metrics, synchronous=False
            )
            mlflow_system_metrics = gpu_metrics + current_row_metrics
        else:
            mlflow_system_metrics.extend(gpu_metrics)
            mlflow_system_metrics.extend(current_row_metrics)

    global_system_metrics_logging_counts += 1
    # Clear up leftovers.
    mlflow_client.log_batch(
        mlflow_run_id, metrics=mlflow_system_metrics, synchronous=False
    )


def convert_wandb_experiment_metrics_to_mlflow(run, mlflow_client, mlflow_run_id):
    metric_history = run.scan_history()
    mlflow_metrics = []
    global global_training_metrics_logging_counts
    for i, row in enumerate(metric_history):
        timestamp = int(row["_timestamp"] * 1000)
        current_row_metrics = []
        for k, v in row.items():
            if k not in ["_timestamp", "_step", "_run_time"] and v:
                if isinstance(v, (int, float)):
                    # Metrics must be either int or float.
                    # There are other types such as str or dict, we should skip them.
                    current_row_metrics.append(
                        mlflow.entities.Metric(
                            k.replace(".", "/"), v, timestamp, int(row["_step"])
                        )
                    )
        if (len(mlflow_metrics) + len(current_row_metrics)) >= 1000:
            global_training_metrics_logging_counts += 1
            mlflow_client.log_batch(
                mlflow_run_id, metrics=mlflow_metrics, synchronous=False
            )
            mlflow_metrics = current_row_metrics
        else:
            mlflow_metrics.extend(current_row_metrics)

    # Clear up leftovers.
    global_training_metrics_logging_counts += 1
    mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics, synchronous=False)


def set_mlflow_experiment(project):
    project_name = project.name

    mlflow_experiment = mlflow.get_experiment_by_name(f"/{project_name}")
    project_id = getattr(project, "id", None)
    if mlflow_experiment:
        if mlflow_experiment.tags.get("migrate_from_wandb_project", None):
            # id matching means this is the right mlflow experiment, so we reuse it.
            mlflow.set_experiment(f"/{project_name}")
        else:
            # The name has already been used on an unrelated MLflow experiment,
            # so we set the experiment name with a random suffix.
            mlflow.set_experiment(f"/{project_name}_{uuid.uuid4().hex[:6]}")
            mlflow.set_experiment_tag("migrate_from_wandb_project", "True")
            mlflow.set_experiment_tag("wandb_project_name", project.name)
            mlflow.set_experiment_tag("wandb_project_id", project_id)
    else:
        mlflow.set_experiment(f"/{project_name}")
        mlflow.set_experiment_tag("migrate_from_wandb_project", "True")
        mlflow.set_experiment_tag("wandb_project_name", project.name)
        mlflow.set_experiment_tag("wandb_project_id", project_id)


@contextmanager
def create_mlflow_parent_run(wandb_run, group_to_run_id=None):
    if not hasattr(wandb_run, "group"):
        yield None
    else:
        group = wandb_run.group
        if group in group_to_run_id:
            # Resume the parent run (wandb group equivalent).
            run = mlflow.start_run(run_id=group_to_run_id[group])
        else:
            run = mlflow.start_run(run_name=group)
            group_to_run_id[group] = run.info.run_id
        try:
            yield run
        finally:
            mlflow.end_run()


def main(_):
    os.environ["MLFLOW_VERBOSE"] = "True"
    project_names = ["shared-moe-Apr3"]

    wandb.login()
    mlflow.login()

    api = wandb.Api()

    for i, project_name in enumerate(project_names):
        project = api.project(name=project_name, entity="mosaic-ml")
        set_mlflow_experiment(project)
        runs = api.runs(path=f"mosaic-ml/{project.name}")

        group_to_run_id = {}

        for run in runs:
            with create_mlflow_parent_run(run, group_to_run_id) as parent_run:
                with mlflow.start_run(
                    run_name=run.name, nested=parent_run is not None
                ) as mlflow_run:
                    client = mlflow.MlflowClient()
                    convert_wandb_config_to_mlflow_params(run)
                    convert_wandb_system_metrics_to_mlflow(
                        run, client, mlflow_run.info.run_id
                    )
                    convert_wandb_experiment_metrics_to_mlflow(
                        run, client, mlflow_run.info.run_id
                    )

                print(
                    "TOTAL SYSTEM METRICS IN ASYNC QUEUE: ",
                    global_system_metrics_logging_counts,
                )
                print(
                    "TOTAL TRAINING METRICS IN ASYNC QUEUE: ",
                    global_training_metrics_logging_counts,
                )

        # Clear the async logging queue per 5 projects to avoid dead threadpool.
        mlflow.flush_async_logging()


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("preset")
    app.run(main)
