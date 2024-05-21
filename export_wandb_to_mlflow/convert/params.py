import json

import mlflow

from export_wandb_to_mlflow.dry_run_utils import log_params_dry_run


def convert_wandb_config_to_mlflow_params(wandb_run, dry_run=False, dry_run_save_dir=None):
    """Convert Wandb config to MLflow params.

    This function converts wandb config for the given `wandb_run` to MLflow params and log to
    MLflow. For nested wandb configs such as {key: some_dict}, `some_dict` is converted to a json
    string before logging to MLflow for UI consistency.

    Args:
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run object.
    """
    converted_config = {
        k: json.dumps(v) if isinstance(v, dict) else v for k, v in wandb_run.config.items()
    }
    if dry_run:
        log_params_dry_run(converted_config, dry_run_save_dir)
    else:
        mlflow.log_params(converted_config, synchronous=False)
