import os
import threading
import time

import mlflow
import wandb
from absl import app, flags, logging
from mlflow.tracking import _get_store

from convert.metrics import convert_wandb_experiment_metrics_to_mlflow
from convert.params import convert_wandb_config_to_mlflow_params
from convert.system_metrics import convert_wandb_system_metrics_to_mlflow
from mlflow_utils import create_mlflow_parent_run, set_mlflow_experiment

flags.DEFINE_string(
    "wandb_project_name",
    None,
    "The name of the wandb project to migrate to MLflow.",
)

flags.DEFINE_string(
    "mlflow_experiment_name",
    None,
    "The name of the MLflow experiment to match the `wandb_project_name`.",
)

flags.DEFINE_bool(
    "verbose",
    False,
    "Whether to enable verbose logging.",
)

flags.DEFINE_bool(
    "use_nested_run",
    False,
    "Whether to use nested run to represent wandb group.",
)

FLAGS = flags.FLAGS


def logging_async_pool_info(stop_event):
    def logging_func(stop_event):
        store = _get_store()
        async_queue = store._async_logging_queue

        while not stop_event.is_set():
            try:
                logging.info(
                    "Number of metrics batch waiting to be logged: "
                    f"{async_queue._batch_logging_worker_threadpool._work_queue.qsize()}"
                )
            except AttributeError:
                logging.error("Failed to get the number of metrics batch waiting to be logged.")
            time.sleep(30)

    # Create the daemon thread
    thread = threading.Thread(target=logging_func, args=(stop_event,), daemon=True)
    thread.start()


def run(_):
    start_time = time.time()

    os.environ["MLFLOW_VERBOSE"] = str(FLAGS.verbose)
    project_name = FLAGS.wandb_project_name

    wandb.login()
    mlflow.login()

    api = wandb.Api()

    wandb_project = api.project(name=project_name, entity="mosaic-ml")
    set_mlflow_experiment(wandb_project, FLAGS.mlflow_experiment_name)
    runs = api.runs(path=f"mosaic-ml/{wandb_project.name}")

    group_to_run_id = {}

    async_pool_logging_stop_event = threading.Event()
    logging_async_pool_info(async_pool_logging_stop_event)

    for run in runs:
        with create_mlflow_parent_run(run, group_to_run_id, FLAGS.use_nested_run) as parent_run:
            with mlflow.start_run(run_name=run.name, nested=parent_run is not None) as mlflow_run:
                logging.info(f"Processing run: {run.name}")
                if getattr(run, "group", None):
                    # Add the wandb group to the mlflow run as a tag.
                    mlflow.set_tag("run_group", run.group)
                mlflow.set_tag("wandb_run_name", run.name)
                client = mlflow.MlflowClient()
                convert_wandb_config_to_mlflow_params(run)
                convert_wandb_system_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                convert_wandb_experiment_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                logging.info(f"Finished processing run: {run.name}")

    logging.info("Waiting for all data to be logged, please be patient, patient and patient...")
    # Clear the async logging queue per 5 projects to avoid dead threadpool.
    mlflow.flush_async_logging()

    async_pool_logging_stop_event.set()

    end_time = time.time()
    logging.info(f"Migration of {project_name} completed in {end_time - start_time:.2f} seconds.")


def main():
    app.run(run)


if __name__ == "__main__":
    main()
