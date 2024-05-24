MLFLOW_MAXIMUM_METRICS_PER_BATCH = 1000


def set_mlflow_maximum_metrics_per_batch(value):
    global MLFLOW_MAXIMUM_METRICS_PER_BATCH
    MLFLOW_MAXIMUM_METRICS_PER_BATCH = value
