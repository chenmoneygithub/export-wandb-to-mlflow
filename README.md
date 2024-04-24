# export-wandb-to-mlflow

export-wandb-to-mlflow is a tool for people to easily export WandB projects to MLflow.
Basically the mapping between WandB and MLflow is:

- WandB projects <=> MLflow experiments
- WandB runs <=> MLflow runs

To use the tool, first clone and install the package:

```bash
git clone https://github.com/chenmoneygithub/export-wandb-to-mlflow.git
cd export-wandb-to-mlflow.git
pip install -e "."
```

Then run to kick off the data migration:

```bash
w2m --wandb_project_name="{your-wandb-project-name}" [Options]
```

Available options are listed below:

| Option                 | Type   | Explanation                                                                                   |
| ---------------------- | ------ | --------------------------------------------------------------------------------------------- |
| mlflow_experiment_name | String | Use the designated name as the MLflow experiment name, instead of automatically creating one. |
| verbose                | Bool   | If verbose logging is enabled.                                                                |
| use_nested_run         | Bool   | If using nested MLflow run to represent wandb group.                                          |

If `mlflow_experiment_name` is not set, an MLflow experiment of the same name as WandB experiment will be created, and all MLflow
runs will be created using the corresponding WandB runs' names. If WandB runs are grouped, the group name will be logged to MLflow
as a tag `run_group`, and users can use this tag to group MLflow runs on MLflow UI.

## Limitations

For now this tool only supports migrating WandB config (params) and metrics. Artifacts migration need to be done manually.
