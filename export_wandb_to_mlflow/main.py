import os
import threading
import time

import mlflow
import wandb
from absl import app, flags, logging
from mlflow.tracking import _get_store

from export_wandb_to_mlflow.convert.metrics import \
    convert_wandb_experiment_metrics_to_mlflow
from export_wandb_to_mlflow.convert.params import \
    convert_wandb_config_to_mlflow_params
from export_wandb_to_mlflow.convert.system_metrics import \
    convert_wandb_system_metrics_to_mlflow
from export_wandb_to_mlflow.mlflow_utils import (create_mlflow_parent_run,
                                                 set_mlflow_experiment)

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


def setup_logging(log_dir):
    if log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        logging.get_absl_handler().use_absl_log_file("export_wandb_to_mlflow", FLAGS.log_dir)

    # Set verbosity level (defaults to INFO)
    logging.set_verbosity(logging.INFO)
    # Set the stderr threshold (to see all logs in the terminal)
    logging.set_stderrthreshold(logging.INFO)


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


def run(
    wandb_project_name,
    mlflow_experiment_name=None,
    verbose=False,
    use_nested_run=False,
    log_dir=None,
):
    setup_logging(log_dir)
    start_time = time.time()

    os.environ["MLFLOW_VERBOSE"] = str(verbose)
    project_name = wandb_project_name

    api = wandb.Api()

    wandb_project = api.project(name=project_name, entity="mosaic-ml")
    set_mlflow_experiment(wandb_project, mlflow_experiment_name)
    runs = api.runs(path=f"mosaic-ml/{wandb_project.name}")

    group_to_run_id = {}

    async_pool_logging_stop_event = threading.Event()
    logging_async_pool_info(async_pool_logging_stop_event)

    for run in runs:
        logging.info(f"Starting processing wandb run: {run.name}.")
        with create_mlflow_parent_run(run, group_to_run_id, use_nested_run) as parent_run:
            with mlflow.start_run(run_name=run.name, nested=parent_run is not None) as mlflow_run:
                logging.info(
                    f"Created Mlflow run: {mlflow_run.info.run_name} with id "
                    f"{mlflow_run.info.run_id}."
                )
                if getattr(run, "group", None):
                    # Add the wandb group to the mlflow run as a tag.
                    mlflow.set_tag("run_group", run.group)
                mlflow.set_tag("wandb_run_name", run.name)
                client = mlflow.MlflowClient()
                convert_wandb_config_to_mlflow_params(run)
                convert_wandb_system_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                convert_wandb_experiment_metrics_to_mlflow(run, client, mlflow_run.info.run_id)

                logging.info(
                    "Done processing wandb data, now waiting for all data to be logged to MLflow "
                    f"for run: {run.name}."
                )
                # Clear the async logging queue per 5 projects to avoid dead threadpool.
                mlflow.flush_async_logging()
                # Set a tag to indicate that the migration is complete.
                mlflow.set_tag("wandb_migration_complete", True)
                logging.info(f"Finished processing run: {run.name}! Moving to the next run...")

    # Stop logging the async pool info.
    async_pool_logging_stop_event.set()

    end_time = time.time()
    logging.info(f"Migration of {project_name} completed in {end_time - start_time:.2f} seconds.")


def launch(_):
    run(
        FLAGS.wandb_project_name,
        FLAGS.mlflow_experiment_name,
        FLAGS.verbose,
        FLAGS.use_nested_run,
        FLAGS.log_dir,
    )


def main():
    wandb.login()
    mlflow.login()
    app.run(launch)


if __name__ == "__main__":
    main()
