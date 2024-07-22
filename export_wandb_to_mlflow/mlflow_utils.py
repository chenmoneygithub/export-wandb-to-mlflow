import re
import uuid
from contextlib import contextmanager
from pathlib import Path

import mlflow
from absl import logging

from export_wandb_to_mlflow.dry_run_utils import set_tags_dry_run


def _set_mlflow_experiment_on_tracking_server(
    wandb_project_name,
    mlflow_experiment_name=None,
    skip_existing_runs=False,
    dual_writing_mlflow_experiment_id=None,
):
    """Set MLflow experiment on the tracking server."""
    mlflow_experiment_name = mlflow_experiment_name or wandb_project_name
    if dual_writing_mlflow_experiment_id:
        mlflow.set_experiment(experiment_id=dual_writing_mlflow_experiment_id)
        mlflow.set_experiment_tags(
            {
                "migrate_from_wandb_project": "True",
                "dual_write_mlflow_wandb": "True",
                "wandb_project_name": wandb_project_name,
            }
        )
        return dual_writing_mlflow_experiment_id

    # Note that in Databricks workspace, MLflow experiment name is prefixed with `/`.
    mlflow_experiment = mlflow.get_experiment_by_name(f"/{mlflow_experiment_name}")
    if mlflow_experiment:
        if mlflow_experiment.tags.get("migrate_from_wandb_project", None):
            # Having tag `migrate_from_wandb_project` indicates that the MLflow experiment is
            # created from wandb project, so we reuse it.
            mlflow.set_experiment(f"/{mlflow_experiment_name}")
        elif skip_existing_runs:
            # When `skip_existing_runs` is True, we reuse the existing MLflow experiment, and add
            # a few tags to indicate that this experiment has associated wandb project. (either
            # migrated from wandb or dual writed to wandb and mlflow)
            mlflow.set_experiment(f"/{mlflow_experiment_name}")
            mlflow.set_experiment_tags(
                {
                    "migrate_from_wandb_project": "True",
                    "wandb_project_name": wandb_project_name,
                }
            )
        else:
            # The name has already been used on an MLflow experiment that is not created from wandb
            # migration, so we set the experiment name with a random suffix.
            mlflow.set_experiment(f"/{mlflow_experiment_name}_{uuid.uuid4().hex[:6]}")
            mlflow.set_experiment_tags(
                {
                    "migrate_from_wandb_project": "True",
                    "wandb_project_name": wandb_project_name,
                }
            )
    else:
        mlflow_experiment = mlflow.set_experiment(f"/{mlflow_experiment_name}")
        mlflow.set_experiment_tag("migrate_from_wandb_project", "True")
        mlflow.set_experiment_tag("wandb_project_name", wandb_project_name)
    return mlflow_experiment.experiment_id


def set_mlflow_experiment(
    wandb_project_name,
    mlflow_experiment_name=None,
    dry_run=False,
    dry_run_save_dir=None,
    skip_existing_runs=False,
    dual_writing_mlflow_experiment_id=None,
):
    """Set MLflow experiment based on the Wandb project.

    Create or reuse an MLflow experiment based on the Wandb project. If `mlflow_experiment_name` is
    set, an MLflow experiment of name `mlflow_experiment_name` will be created. Otherwise we will
    create MLflow experiment "/my-wandb-project" (prefixed with "/") to match wandb project with
    name "my-wandb-project". If an MLflow experiment with the same name already exists, and it is
    created from wandb project, as indicated by the tag `migrate_from_wandb_project`, we reuse it.
    Otherwise, we create a new MLflow experiment with a random suffix.

    When a new MLflow experiment is created, it will be tagged with `migrate_from_wandb_project`
    and the wandb project's name in tag `wandb_project_name`.

    This function also supports dry run mode. In dry run mode, the function will create a directory
    to save the data instead of creating an MLflow experiment.

    Args:
        wandb_project_name (str): The name of the Wandb project.
        mlflow_experiment_name (str, optional): The name of the MLflow experiment. If None, the
            experiment name will be the same as the wandb project name.
        dry_run (bool): Whether to run in dry run mode.
        dry_run_save_dir (str, optional): The directory to save the data in dry run mode. If None,
            the current directory will be used.
        skip_existing_runs (bool): Whether to skip existing runs.
        dual_writing_mlflow_experiment_id (str, optional): The experiment ID of the MLflow
            experiment that is being dual written to.
    """
    if dry_run:
        current_path = Path.cwd()
        dry_run_save_dir = Path(dry_run_save_dir) if dry_run_save_dir else current_path
        if not dry_run_save_dir.exists():
            raise ValueError(
                f"Directory '{dry_run_save_dir}' does not exist. Please create it first."
            )
        mlflow_experiment_name = mlflow_experiment_name or wandb_project_name
        experiment_path = dry_run_save_dir / mlflow_experiment_name
        if experiment_path.exists():
            if skip_existing_runs:
                return experiment_path
            else:
                raise ValueError(
                    f"The experiment path {experiment_path} already exists, please remove it first "
                    "or set `resume_from_crash=True` if you are resuming from a previous crash."
                )
        experiment_path.mkdir()
        experiment_tags = {
            "migrate_from_wandb_project": True,
            "wandb_project_name": wandb_project_name,
        }
        # Set experiment tags.
        set_tags_dry_run(experiment_tags, experiment_path)
        return experiment_path
    else:
        return _set_mlflow_experiment_on_tracking_server(
            wandb_project_name,
            mlflow_experiment_name,
            skip_existing_runs,
            dual_writing_mlflow_experiment_id,
        )


def set_mlflow_tags(tags, dry_run=False, dry_run_save_dir=None):
    """Function to set mlflow tags, support dry run mode."""
    if dry_run:
        # Write the tags to file.
        set_tags_dry_run(tags, dry_run_save_dir)
    else:
        mlflow.set_tags(tags)


@contextmanager
def start_mlflow_run(wandb_run, dry_run=False, dry_run_experiment_dir=None):
    """Start MLflow run based on the Wandb run.

    Dry run mode is supported. In dry run mode, the function will create a directory to save the
    data instead of writing data to the MLflow tracking server.

    Args:
        wandb_run (wandb.sdk.wandb_run.Run): The Wandb run object.
        dry_run (bool): Whether to run in dry run mode.
        dry_run_experiment_dir (pathlib.Path): The directory to save the data in dry run mode.

    Yields:
        An MLflow run object or a directory path in dry run mode.
    """
    if not dry_run:
        with mlflow.start_run(run_name=wandb_run.name) as run:
            yield run

    else:
        # Dry run mode should use `wandb_run.id` instead of `wandb_run.name` as the directory name
        # to avoid duplicates.
        run_dir = dry_run_experiment_dir / wandb_run.id
        if run_dir.exists():
            raise ValueError(
                f"Run directory {run_dir} already exists, please remove it first. If you are "
                "resuming from a previous crash, please set `resume_from_crash=True`."
            )
        run_dir.mkdir()
        yield run_dir


def get_existing_runs(mlflow_experiment):
    """Get existing runs in the MLflow experiment.

    This function is used to get existing runs in the MLflow experiment. It returns a list of wandb
    run ids that already exist in dry run directory or MLflow experiment.

    Args:
        mlflow_experiment (str or pathlib.Path): The MLflow experiment ID or path to the experiment
            directory created in the dry run mode.

    Returns:
        List[str]: A list of wandb run ids that already exist in the MLflow experiment.
    """
    if isinstance(mlflow_experiment, Path):
        existing_runs = []
        for child_dir in mlflow_experiment.iterdir():
            # iterdir() only lists the top-level contents
            if not child_dir.is_dir():
                continue
            existing_runs.append(str(child_dir.relative_to(mlflow_experiment)))
    else:
        fetched_runs = mlflow.search_runs(experiment_ids=[mlflow_experiment])
        existing_runs = (
            fetched_runs["tags.wandb_run_id"].to_list()
            if "tags.wandb_run_id" in fetched_runs.columns
            else []
        )
    return existing_runs


def should_skip_run(
    run,
    target_wandb_run_names=None,
    existing_runs=None,
    skip_existing_runs=False,
    skip_dual_writing_runs=False,
):
    """Check if the wandb run should be skipped.

    There are three conditions to skip a run:
        1. The run already exists in the MLflow experiment and user specifies to skip existing runs.
        2. The run is already existing in MLflow due to dual writing and user specifies to skip dual
            writing runs. This is different from 1, because existing runs can be created from
            migration or other reasons.
        3. The run is not in the target runs to migrate, if users specify target runs.

    Args:
        run (wandb.sdk.wandb_run.Run | CrashHandler): The Wandb run object or CrashHandler object
            managing the run in dry run /resume from dry run mode.
        target_wandb_run_names (List[str]): The list of target runs to migrate.
        existing_runs (List[str]): The list of existing runs in the MLflow experiment.
        skip_existing_runs (bool): Whether to skip existing runs.
        skip_dual_writing_runs (bool): Whether to skip dual writing runs.

    Returns:
        bool: Whether to skip the run.
    """
    run_name = run.name
    if skip_existing_runs:
        if run.id in existing_runs:
            logging.info(
                f"Skipping run {run_name} because it already exists and you set "
                "`skip_existing_runs=True`."
            )
            return True
    if skip_dual_writing_runs:
        if "mlflow_experiment_id" in run.config or "mlflow" in run.config.get("loggers", {}):
            logging.info(
                f"Skipping run {run_name} because it's already existing in MLflow due to dual "
                "writing and you set `skip_existing_runs=True`."
            )
            return True

    if not target_wandb_run_names:
        return False
    if len(target_wandb_run_names) > 0:
        # Only migrate the runs specified in `wandb_run_names` if it is not empty.
        for target_run_name in target_wandb_run_names:
            if re.match(target_run_name, run.name):
                return False
    logging.info(
        f"Skipping run {run_name} because it's not in the target runs to migrate. "
        f"Target run names (regex supported): {target_wandb_run_names}."
    )
    return True
