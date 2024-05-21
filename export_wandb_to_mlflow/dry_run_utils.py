"""Utility functions for dry run mode."""

import csv
import json


def log_params_dry_run(params, dry_run_save_dir):
    """Log parameters to a file in dry run mode.

    This function logs the parameters to a file in dry run mode. The file is saved in the directory
    specified by `dry_run_save_dir`.

    Args:
        params (dict): The parameters to be logged.
        dry_run_save_dir (pathlib.Path): The directory to save the parameters.
    """
    file_path = dry_run_save_dir / "params.json"
    with file_path.open(mode="w") as f:
        json.dump(params, f)


def log_metrics_dry_run(metrics, dry_run_save_dir, index=0):
    """Log metrics to a file in dry run mode.

    This function logs the metrics to a file in dry run mode. The file is saved in the directory
    specified by `dry_run_save_dir`.

    Args:
        metrics (List[mlflow.entities.Metric]): The metrics to be logged.
        dry_run_save_dir (pathlib.Path): The directory to save the metrics.
    """
    file_path = dry_run_save_dir / f"metrics_batch_{index}.csv"
    with file_path.open(mode="a") as f:
        for metric in metrics:
            f.write(f"{metric.key}, {metric.value}, {metric.timestamp}, {metric.step}\n")


def set_tags_dry_run(tags, dry_run_save_dir):
    """Set tags in dry run mode.

    This function sets tags in dry run mode. The tags are saved to a file in the directory specified
    by `dry_run_save_dir`.

    Args:
        tags (dict): The tags to be set.
        dry_run_save_dir (pathlib.Path): The directory to save the tags.
    """
    # Write the tags to file.
    tags_path = dry_run_save_dir / "tags.csv"

    with tags_path.open("a") as f:
        for key, value in tags.items():
            f.write(f"{key}, {value}\n")


def read_tags(tag_path):
    """Read Mlflow tags from a file.

    This function reads tags from a file and returns them as a dictionary.

    Args:
        tag_path (pathlib.Path): The path to the file containing the tags.

    Returns:
        A dict containing mlflow tags.
    """
    tags = {}
    with tag_path.open(mode="r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            tags[row[0]] = row[1]
    return tags
