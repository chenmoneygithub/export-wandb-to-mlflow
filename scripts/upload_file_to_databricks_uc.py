# Script to upload local files to databricks UC with multithreading.
# This script is way faster than the databricks CLI `databricks fs cp` command.

from databricks.sdk import WorkspaceClient

from absl import flags, app, logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import os

flags.DEFINE_string(
    "path",
    None,
    "The path to the local directory that contains wandb project data",
)

flags.DEFINE_integer(
    "upload_thread_pool_size",
    10,
    "thread pool size for uploading runs to databricks uc",
)

UC_PREFIX = "/Volumes/main/chenmoney/wandb_migrations/"
LOCAL_PREFIX = "./migrations/"
FLAGS = flags.FLAGS


def process_one_run(run_path):
    run_name = os.path.basename(run_path)
    logging.info(f"Uploading run to UC: {run_name}")
    w = WorkspaceClient()
    for root, dirs, files in os.walk(run_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = Path(file_path).relative_to(Path(LOCAL_PREFIX))
            uc_path = UC_PREFIX + str(relative_path)

            with open(file_path, "rb") as f:
                w.files.upload(uc_path, f)

    logging.info(f"Finished uploading run to UC: {run_name}")


def main(_):
    executor = ThreadPoolExecutor(max_workers=FLAGS.upload_thread_pool_size)
    local_project_path = LOCAL_PREFIX + FLAGS.path
    w = WorkspaceClient()

    logging.info(f"Starting to upload {local_project_path} to databricks uc...")
    start_time = time.time()

    entries = os.listdir(local_project_path)
    for entry in entries:
        path = os.path.join(local_project_path, entry)
        if os.path.isfile(path):
            relative_path = Path(path).relative_to(Path(LOCAL_PREFIX))
            uc_path = UC_PREFIX + str(relative_path)
            with open(path, "rb") as f:
                w.files.upload(uc_path, f)
        else:
            executor.submit(process_one_run, path)

    executor.shutdown(wait=True)

    end_time = time.time()
    logging.info(
        f"Finished uploading {local_project_path} to databricks uc in {end_time - start_time} "
        "seconds."
    )


if __name__ == "__main__":
    app.run(main)
