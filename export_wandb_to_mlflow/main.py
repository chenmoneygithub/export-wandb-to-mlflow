import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import mlflow
import wandb
from absl import app, flags, logging
from mlflow.tracking import _get_store

from export_wandb_to_mlflow.convert.metrics import convert_wandb_metrics_to_mlflow
from export_wandb_to_mlflow.convert.params import convert_wandb_config_to_mlflow_params
from export_wandb_to_mlflow.convert.system_metrics import convert_wandb_system_metrics_to_mlflow
from export_wandb_to_mlflow.dry_run_utils import RunReadHandler
from export_wandb_to_mlflow.mlflow_utils import (
    set_mlflow_experiment,
    set_mlflow_tags,
    start_mlflow_run,
)
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
    "will be migrated, otherwise all runs in the wandb project will be migrated. Support regex.",
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

flags.DEFINE_bool(
    "dry_run",
    False,
    "If True, the script will write wandb data to files instead of MLflow server. This is useful "
    "when you want to keep the data locally before migrating to MLflow.",
)

flags.DEFINE_bool(
    "resume_from_dry_run",
    False,
    "If True, the script will read the data from the dry run directory and migrate it to MLflow."
    "This is useful when you ran the script in dry run mode before to save data to some disk.",
)

flags.DEFINE_string(
    "dry_run_save_dir",
    None,
    "The directory path to save the data in dry run mode. If None, the current directory will be "
    "used.",
)

flags.DEFINE_integer(
    "dry_run_thread_pool_size",
    None,
    "The size of threadpool to use in dry run mode. If None, the program doesn't use "
    "multithreading in the dry run mode.",
)


FLAGS = flags.FLAGS


def _setup_absl_logging(log_dir):
    """Configure absl logging."""
    if log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        logging.get_absl_handler().use_absl_log_file("export_wandb_to_mlflow", FLAGS.log_dir)

    # Set verbosity level (defaults to INFO)
    logging.set_verbosity(logging.INFO)
    # Set the stderr threshold (to see all logs in the terminal)
    logging.set_stderrthreshold(logging.INFO)


def _logging_async_pool_info(stop_event):
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


def _validate_flags(
    dry_run=False,
    resume_from_dry_run=False,
    dry_run_save_dir=None,
):
    if dry_run and resume_from_dry_run:
        raise ValueError(
            "The `dry_run` and `resume_from_dry_run` flags cannot be both set to True."
        )
    if dry_run and not dry_run_save_dir:
        raise ValueError("The `dry_run_save_dir` must be specified when `dry_run` is True.")

    if resume_from_dry_run and not dry_run_save_dir:
        raise ValueError(
            "The `dry_run_save_dir` must be specified when `resume_from_dry_run` is True."
        )


def migrate_data(run, mlflow_experiment, exclude_metrics, dry_run, resume_from_dry_run):
    logging.info(f"Starting processing wandb run: {run.name}.")
    with start_mlflow_run(
        run, dry_run=dry_run, dry_run_experiment_dir=mlflow_experiment
    ) as mlflow_run:
        if dry_run:
            logging.info(
                f"Migrating wandb run {run.name} to MLflow in dry run mode. The saved data "
                f"is stored in {str(mlflow_run)}."
            )
        else:
            logging.info(
                f"Created Mlflow run: {mlflow_run.info.run_name} with id "
                f"{mlflow_run.info.run_id}."
            )
        tags = {}
        if getattr(run, "group", None):
            # Add the wandb group to the mlflow run as a tag.
            tags["run_group"] = run.group

        tags["wandb_run_name"] = run.name
        tags["wandb_run_id"] = run.id
        set_mlflow_tags(tags, dry_run=dry_run, dry_run_save_dir=mlflow_run)

        # client = None if dry_run else mlflow.MlflowClient()
        client = mlflow.MlflowClient()
        convert_wandb_config_to_mlflow_params(
            run,
            dry_run=dry_run,
            resume_from_dry_run=resume_from_dry_run,
            dry_run_save_dir=mlflow_run,
        )
        logging.info(f"Starting converting system metrics for {run.name} (id: {run.id})...")
        convert_wandb_system_metrics_to_mlflow(
            run,
            client,
            mlflow_run,
            dry_run=dry_run,
            resume_from_dry_run=resume_from_dry_run,
            dry_run_save_dir=mlflow_run,
        )
        logging.info(f"Starting converting metrics for {run.name} (id: {run.id})...")
        convert_wandb_metrics_to_mlflow(
            run,
            client,
            mlflow_run,
            exclude_metrics=exclude_metrics,
            dry_run=dry_run,
            resume_from_dry_run=resume_from_dry_run,
            dry_run_save_dir=mlflow_run,
        )
        if not dry_run:
            logging.info(
                "Done processing wandb data, now waiting for all data to be logged to MLflow "
                f"for run: {run.name}."
            )
            mlflow.flush_async_logging()

        # Set a tag to indicate that the migration is complete.
        set_mlflow_tags(
            {"wandb_migration_complete": True},
            dry_run=dry_run,
            dry_run_save_dir=mlflow_run,
        )
        logging.info(f"Finished processing run: {run.name}! Moving to the next run...")


def should_skip_run(run_name, target_wandb_run_names):
    if not target_wandb_run_names:
        return False
    # Only migrate the runs specified in `wandb_run_names` if it is not empty.
    for target_run_name in target_wandb_run_names:
        if re.match(target_run_name, run_name):
            return False
    logging.info(
        f"Skipping run {run_name} because it's not in the target runs to migrate. "
        f"Target run names (regex supported): {target_wandb_run_names}."
    )
    return True


def run(
    wandb_project_name,
    mlflow_experiment_name=None,
    verbose=False,
    log_dir=None,
    wandb_run_names=None,
    exclude_metrics=None,
    resume_from_crash=False,
    dry_run=False,
    resume_from_dry_run=False,
    dry_run_save_dir=None,
    dry_run_thread_pool_size=None,
):
    """Main function to migrate wandb data to mlflow."""
    _validate_flags(dry_run, resume_from_dry_run, dry_run_save_dir)
    _setup_absl_logging(log_dir)
    start_time = time.time()

    os.environ["MLFLOW_VERBOSE"] = str(verbose)

    if resume_from_dry_run:
        # When `resume_from_dry_run` is True, we need to read the data from the dry run directory.
        # instead of wandb server.
        if not dry_run_save_dir:
            raise ValueError(
                "The `dry_run_save_dir` must be specified when `resume_from_dry_run` is True."
            )
        mlflow_experiment_name = mlflow_experiment_name or wandb_project_name
        mlflow_experiment_path = Path(dry_run_save_dir) / mlflow_experiment_name
        runs = []
        for child_dir in mlflow_experiment_path.iterdir():
            # iterdir() only lists the top-level contents
            if not child_dir.is_dir():
                continue
            runs.append(RunReadHandler(child_dir))
    else:
        # When `resume_from_dry_run` is False, we need to read the data from wandb server.
        api = wandb.Api()
        wandb_project = api.project(name=wandb_project_name, entity="mosaic-ml")
        runs = api.runs(path=f"mosaic-ml/{wandb_project.name}")

    mlflow_experiment = None
    if resume_from_crash:
        # If we are resuming from a crash, we need to delete the crashed runs and get the finished
        # runs. We also need to set the mlflow experiment to the one that was used before the crash.
        crash_handler = CrashHandler(
            wandb_project_name,
            dry_run=dry_run,
            dry_run_save_dir=dry_run_save_dir,
        )
        mlflow_experiment = crash_handler.mlflow_experiment
        crash_handler.delete_crashed_runs_and_get_finished_runs()
    else:
        # If we are not resuming from a crash, we need to create a new mlflow experiment.
        mlflow_experiment = set_mlflow_experiment(
            wandb_project_name,
            mlflow_experiment_name,
            dry_run=dry_run,
            dry_run_save_dir=dry_run_save_dir,
        )
    if dry_run and dry_run_thread_pool_size:
        # In dry run mode, we can concurrently process request to speed up the process.
        executor = ThreadPoolExecutor(max_workers=dry_run_thread_pool_size)

    if not dry_run:
        # If not in dry run mode, start logging the MLflow async logging pool information.
        async_pool_logging_stop_event = threading.Event()
        _logging_async_pool_info(async_pool_logging_stop_event)

    for run in runs:
        if should_skip_run(run.name, wandb_run_names):
            continue
        if resume_from_crash and run.id in crash_handler.finished_wandb_run_ids:
            # Skip the run that has been finished.
            logging.info(
                f"Skipping wandb run {run.name} of id {run.id} because it's already done.'"
            )
            continue

        if dry_run and dry_run_thread_pool_size:
            # In dry run mode, we can concurrently process request to speed up the process.
            executor.submit(
                migrate_data,
                run,
                mlflow_experiment,
                exclude_metrics,
                dry_run,
                resume_from_dry_run,
            )
        else:
            # If not in dry run mode, process the runs sequentially because MLflow server
            # cannot handle too many requests at the same time.
            migrate_data(run, mlflow_experiment, exclude_metrics, dry_run, resume_from_dry_run)

    if not dry_run:
        # Stop logging the MLflow async pool info.
        async_pool_logging_stop_event.set()

    if dry_run and dry_run_thread_pool_size:
        executor.shutdown(wait=True)

    end_time = time.time()
    logging.info(
        f"Migration of {wandb_project_name} completed in {end_time - start_time:.2f} seconds."
    )


def launch(_):
    if not FLAGS.resume_from_dry_run:
        wandb.login()
    if not FLAGS.dry_run:
        mlflow.login()
    run(
        FLAGS.wandb_project_name,
        FLAGS.mlflow_experiment_name,
        FLAGS.verbose,
        FLAGS.log_dir,
        FLAGS.wandb_run_names,
        FLAGS.exclude_metrics,
        FLAGS.resume_from_crash,
        FLAGS.dry_run,
        FLAGS.resume_from_dry_run,
        FLAGS.dry_run_save_dir,
        FLAGS.dry_run_thread_pool_size,
    )


def main():
    flags.mark_flag_as_required("wandb_project_name")
    app.run(launch)


if __name__ == "__main__":
    main()
