import mlflow
from absl import logging


class CrashHandler:
    def __init__(self, wandb_project_name):
        self.wandb_project_name = wandb_project_name

    def find_corresponding_mlflow_experiment(self):
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
        mlflow_experiment = self.find_corresponding_mlflow_experiment()

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

        self.finished_wandb_run_id = finished_runs.loc[:, "tags.wandb_run_id"].tolist()
