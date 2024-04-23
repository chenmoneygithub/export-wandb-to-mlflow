import uuid
from contextlib import contextmanager

import mlflow


def set_mlflow_experiment(wandb_project, mlflow_experiment_name=None):
    """Set MLflow experiment based on the Wandb project.

    Create or reuse an MLflow experiment based on the Wandb project. If `mlflow_experiment_name` is
    set, an MLflow experiment of name `mlflow_experiment_name` will be created. Otherwise we will
    create MLflow experiment "/my-wandb-project" (prefixed with "/") to match wandb project with
    name "my-wandb-project". If an MLflow experiment with the same name already exists, and it is
    created from wandb project, as indicated by the tag `migrate_from_wandb_project`, we reuse it.
    Otherwise, we create a new MLflow experiment with a random suffix.

    When a new MLflow experiment is created, it will be tagged with `migrate_from_wandb_project`
    and the wandb project's name in tag `wandb_project_name`.

    Args:
        wandb_project (wandb.sdk.wandb_project.Project): Wandb project object.
        mlflow_experiment_name (str, optional): The user select name for the MLflow experiment.
            if None, we will automatically find a name based on the wandb project name.
    """
    wandb_project_name = wandb_project.name

    if mlflow_experiment_name:
        mlflow.set_experiment(mlflow_experiment_name)
        return

    # Note that in Databricks workspace, MLflow experiment name is prefixed with `/`.
    mlflow_experiment = mlflow.get_experiment_by_name(f"/{wandb_project_name}")
    if mlflow_experiment:
        if mlflow_experiment.tags.get("migrate_from_wandb_project", None):
            # id matching means this is the right mlflow experiment, so we reuse it.
            mlflow.set_experiment(f"/{wandb_project_name}")
        else:
            # The name has already been used on an MLflow experiment that is not created from wandb
            # migration, so we set the experiment name with a random suffix.
            mlflow.set_experiment(f"/{wandb_project_name}_{uuid.uuid4().hex[:6]}")
            mlflow.set_experiment_tag("migrate_from_wandb_project", "True")
            mlflow.set_experiment_tag("wandb_project_name", wandb_project_name)
    else:
        mlflow.set_experiment(f"/{wandb_project_name}")
        mlflow.set_experiment_tag("migrate_from_wandb_project", "True")
        mlflow.set_experiment_tag("wandb_project_name", wandb_project_name)


@contextmanager
def create_mlflow_parent_run(wandb_run, group_to_run_id=None, enable=False):
    """Create MLflow parent run based on Wandb run's group.

    This function function is a context manager that creates an MLflow parent run based on the
    Wandb run's group. The parent run is an equivalent to wandb's `group`. If the parent run already
    exists for the given `group`, we resume it. The parent run is created with the same name as the
    wandb group.
    """
    if not enable or not hasattr(wandb_run, "group"):
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
