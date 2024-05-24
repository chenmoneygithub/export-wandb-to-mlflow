"""Utility functions for dry run mode."""

import csv
import json
import os
from collections import deque
from pathlib import Path

from mlflow.entities import Metric

from export_wandb_to_mlflow.config import MLFLOW_MAXIMUM_METRICS_PER_BATCH


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


def log_metrics_dry_run(metrics, dry_run_save_dir, file_handlers):
    """Log metrics to a file in dry run mode.

    This function logs the metrics to a file in dry run mode. The file is saved in the directory
    specified by `dry_run_save_dir`.

    Args:
        metrics (List[mlflow.entities.Metric]): The metrics to be logged.
        dry_run_save_dir (pathlib.Path): The directory to save the metrics.
        file_handlers (Dict[str, File]): A dictionary containing file handlers for each metric file.
    """
    for metric in metrics:
        metric_path = dry_run_save_dir / f"{metric.key}.csv"
        if str(metric_path) not in file_handlers:
            # Haven't create this metric file yet.
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            file_handlers[str(metric_path)] = metric_path.open(mode="a")
        file_handlers[str(metric_path)].write(
            f"{metric.value}, {metric.timestamp}, {metric.step}\n"
        )


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
            # Strip out the leading and trailing spaces.
            tags[row[0]] = row[1].strip()
    return tags


class RunReadHandler:
    def __init__(self, run_path):
        self.run_path = run_path

        self._tags_path = run_path / "tags.csv"
        self.tags = self.read_tags()
        self.id = self.tags["wandb_run_id"]
        self.name = self.tags["wandb_run_name"]
        if "run_group" in self.tags:
            self.group = self.tags["run_group"]

        self._metrics_path = run_path / "metrics"
        self._system_metrics_path = run_path / "system_metrics"
        self._params_path = run_path / "params.json"

    def read_params(self):
        """Read Mlflow parameters from a file.

        This function reads parameters from a file and returns them as a dictionary.

        Returns:
            A dict containing mlflow parameters.
        """
        params = {}
        with self._params_path.open(mode="r") as file:
            params = json.load(file)
        return params

    def read_tags(self):
        return read_tags(self._tags_path)

    def read_metrics(self):
        """Read metrics and return a iterable generator of metrics."""
        yield from self._read_metrics(self._metrics_path)

    def read_system_metrics(self):
        """Read system metrics and return a iterable generator."""
        yield from self._read_metrics(self._system_metrics_path)

    def _cast_str_to_number(self, value):
        try:
            return int(value)
        except ValueError:
            return float(value)

    def _read_metrics(self, metrics_path):
        """Read Mlflow metrics from a file.

        This function reads metrics from a file and returns them as a generator of Metric objects.
        Users can iterate over the generator to get the metrics. To speed up the database i/o, we
        are reading the metrics in batches and in a circular form.

        Args:
            metrics_path (pathlib.Path): The path to the directory containing the metrics.

        Returns:
            A generator of List of `mlflow.entities.Metric`.
        """
        file_handlers_queue = deque()
        for root, dirs, files in os.walk(metrics_path):
            for file in files:
                if not file.endswith(".csv"):
                    continue

                csv_path = Path(os.path.join(root, file))
                relative_path = csv_path.relative_to(Path(metrics_path))
                key = os.path.splitext(relative_path)[0]

                # Get file handlers for every metric file.
                file_handler = csv_path.open(mode="r", newline="")
                file_handlers_queue.append((key, file_handler))

        def get_metrics_batch(file_handler):
            reader = csv.reader(file_handler, delimiter=",")
            metrics = []
            for row in reader:
                value = self._cast_str_to_number(row[0])
                timestamp = int(row[1].strip())
                step = int(row[2].strip())
                metrics.append(Metric(key, value, timestamp, step))
                if len(metrics) >= MLFLOW_MAXIMUM_METRICS_PER_BATCH:
                    return metrics, False

            return metrics, True

        while file_handlers_queue:
            key, file_handler = file_handlers_queue.popleft()

            metrics, finished = get_metrics_batch(file_handler)
            yield metrics
            if finished:
                file_handler.close()
            else:
                file_handlers_queue.append((key, file_handler))
