from pathlib import Path

import mlflow
from absl import logging

from export_wandb_to_mlflow.dry_run_utils import read_tags


class CrashHandler:
    """Class that handles the crash state of the migration job.

    Args:
        wandb_project_name (str): The name of the Wandb project.
    """

    def __init__(self, wandb_project_name, dry_run=False, dry_run_save_dir=None):
        self.wandb_project_name = wandb_project_name
        self.dry_run = dry_run
        self.dry_run_save_dir = Path(dry_run_save_dir)

    def find_corresponding_mlflow_experiment(self):
        """Get the corresponding MLflow experiment based on the Wandb project."""
        if self.dry_run:
            # If the dry run is enabled, we search for the experiment in file system.
            if not self.dry_run_save_dir.is_dir():
                raise ValueError(
                    "Cannot find corresponding MLflow experiment in file system while you set "
                    "`resume_from_crash=True` and `dry_run=True`. Please double check your "
                    "`wandb_project_name` or set `resume_from_crash=False`."
                )
            return self.dry_run_save_dir

        # In non-dry-run mode, we search for the experiment in MLflow, and return the experiment.
        mlflow_experiment = mlflow.get_experiment_by_name(f"/{self.wandb_project_name}")
        if (
            mlflow_experiment is None
            or mlflow_experiment.tags.get("migrate_from_wandb_project", None) is None
        ):
            raise ValueError(
                "Cannot find corresponding MLflow experiment while you set "
                "`resume_from_crash=True`. Please double check your `wandb_project_name` or set "
                "`resume_from_crash=False`."
            )
        mlflow.set_experiment(f"/{self.wandb_project_name}")
        return mlflow_experiment

    def delete_crashed_runs_and_get_finished_runs(self):
        """Delete the crashed runs and get the finished runs.

        This method deletes the runs that are not finished due to the previous crash. It also
        fetches already finished runs and store the information in `self.finished_wandb_run_ids`.
        """
        mlflow_experiment = self.find_corresponding_mlflow_experiment()

        if self.dry_run:
            finished_runs = []
            for child_dir in mlflow_experiment.iterdir():
                # iterdir() only lists the top-level contents
                if not child_dir.is_dir():
                    continue
                # Find a run, then we check if it is finished by reading the tags.
                tags = read_tags(child_dir)
                if tags.get("wandb_migration_complete") == "True":
                    finished_runs.append(str(child_dir.relative_to(mlflow_experiment)))
                else:
                    # If the run is not finished, we delete it.
                    child_dir.rmdir()
            self.finished_wandb_run_ids = finished_runs
        else:
            runs = mlflow.search_runs(experiment_ids=mlflow_experiment.experiment_id)
            finished_runs = mlflow.search_runs(
                experiment_ids=mlflow_experiment.experiment_id,
                filter_string="tags.wandb_migration_complete='True'",
            )
            all_run_ids = runs.run_id.to_list()
            finished_run_ids = finished_runs.run_id.to_list()
            for run_id in all_run_ids:
                if run_id not in finished_run_ids:
                    logging.info(
                        f"Deleting run {run_id} because the previous migration did not finish."
                    )
                    mlflow.delete_run(run_id)

            self.finished_wandb_run_ids = finished_runs.loc[:, "tags.wandb_run_id"].tolist()
