import wandb
import mlflow
import os
from src.convert.metrics import convert_wandb_experiment_metrics_to_mlflow
from src.convert.params import convert_wandb_config_to_mlflow_params
from src.convert.system_metrics import convert_wandb_system_metrics_to_mlflow

from mlflow_utils import set_mlflow_experiment, create_mlflow_parent_run

# TODO: add flag to control which project to migrate, and control logging.


def export_wandb_projet_to_mlflow(wandb_project_name, wandb_entity):
    api = wandb.Api()
    project = api.project(name=wandb_project_name, entity=wandb_entity)
    set_mlflow_experiment(project)
    runs = api.runs(path=f"mosaic-ml/{project.name}")

    wandb_group_to_mlflow_parent_run_id = {}

    for run in runs:
        with create_mlflow_parent_run(run, wandb_group_to_mlflow_parent_run_id) as parent_run:
            with mlflow.start_run(run_name=run.name, nested=parent_run is not None) as mlflow_run:
                client = mlflow.MlflowClient()
                convert_wandb_config_to_mlflow_params(run)
                convert_wandb_system_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                convert_wandb_experiment_metrics_to_mlflow(run, client, mlflow_run.info.run_id)

    # Clear the async logging queue per 5 projects to avoid dead threadpool.
    mlflow.flush_async_logging()


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
                    convert_wandb_system_metrics_to_mlflow(run, client, mlflow_run.info.run_id)
                    convert_wandb_experiment_metrics_to_mlflow(run, client, mlflow_run.info.run_id)

        # Clear the async logging queue per 5 projects to avoid dead threadpool.
        mlflow.flush_async_logging()


if __name__ == "__main__":
    mlflow.config.enable_async_logging()
