import os
import threading
import time

import mlflow
import wandb
from absl import app, flags, logging
from mlflow.tracking import _get_store

from export_wandb_to_mlflow.convert.metrics import convert_wandb_experiment_metrics_to_mlflow
from export_wandb_to_mlflow.convert.params import convert_wandb_config_to_mlflow_params
from export_wandb_to_mlflow.convert.system_metrics import convert_wandb_system_metrics_to_mlflow
from export_wandb_to_mlflow.mlflow_utils import create_mlflow_parent_run, set_mlflow_experiment
from export_wandb_to_mlflow.remedy.get_crash_state import CrashHandler

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

flags.DEFINE_list(
    "wandb_run_names",
    [],
    "The list of run names to migrate to MLflow. If specified, only the runs with the given names "
    "will be migrated, otherwise all runs in the project will be migrated.",
)

flags.DEFINE_list(
    "exclude_metrics",
    [],
    "The list metrics to exclude from migration, regex is supported.",
)

flags.DEFINE_bool(
    "resume_from_crash",
    False,
    "Indicate if this job is resuming from a crash. Migration script could crash in the middle, "
    "setting this flag to True will skip migrating runs already complete, delete then remigrate "
    "the run crashing in the middle, and migrate all other runs.",
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
    wandb_run_names=None,
    exclude_metrics=None,
    resume_from_crash=False,
):
    setup_logging(log_dir)
    start_time = time.time()

    os.environ["MLFLOW_VERBOSE"] = str(verbose)
    project_name = wandb_project_name

    api = wandb.Api()

    wandb_project = api.project(name=project_name, entity="mosaic-ml")
    if resume_from_crash:
        crash_handler = CrashHandler(wandb_project_name)
        crash_handler.delete_crashed_runs_and_get_finished_runs()
    else:
        set_mlflow_experiment(wandb_project, mlflow_experiment_name)

    runs = api.runs(path=f"mosaic-ml/{wandb_project.name}")

    group_to_run_id = {}

    async_pool_logging_stop_event = threading.Event()
    logging_async_pool_info(async_pool_logging_stop_event)

    for run in runs:
        if wandb_run_names and run.name not in wandb_run_names:
            # Only migrate the runs specified in `wandb_run_names` if it is not empty.
            continue
        if resume_from_crash and run.id in crash_handler.finished_wandb_run_id:
            # Skip the run that has been finished.
            logging.info(
                f"Skipping wandb run {run.name} of id {run.id} because it's already done.'"
            )
            continue

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
                mlflow.set_tags(
                    {
                        "wandb_run_name": run.name,
                        "wandb_run_id": run.id,
                    }
                )
                client = mlflow.MlflowClient()
                convert_wandb_config_to_mlflow_params(run)
                convert_wandb_system_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                convert_wandb_experiment_metrics_to_mlflow(
                    run,
                    client,
                    mlflow_run.info.run_id,
                    exclude_metrics=exclude_metrics,
                )

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
        FLAGS.wandb_run_names,
        FLAGS.exclude_metrics,
        FLAGS.resume_from_crash,
    )


def main():
    wandb.login()
    mlflow.login()
    app.run(launch)


if __name__ == "__main__":
    main()
