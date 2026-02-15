import pandas as pd
import yaml
import os
import json
import time
from datetime import datetime


def run_monitoring():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_dir = config["deployment"]["model_dir"]
    metadata_name = config["deployment"]["metadata_name"]
    log_name = config["monitoring"]["api_log"]
    drift_threshold = config["monitoring"]["drift_threshold"]
    production_dir = config["data"]["production_dir"]
    target_col = config["data"]["target_column"]
    monitor_log = config["monitoring"]["monitor_log"]

    log_path = os.path.join(production_dir, log_name)

    if not os.path.exists(log_path):
        print("Production log not found.")
        return

    prod_df = pd.read_csv(log_path)

    production_accuracy = (prod_df["actual"] == prod_df["prediction"]).mean()

    metadata_path = os.path.join(model_dir, metadata_name)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    training_accuracy = metadata["accuracy"]
    accuracy_drop = training_accuracy - production_accuracy 
    print("Training Accuracy:", training_accuracy) 
    print("Production Accuracy:", production_accuracy) 
    print("Accuracy Drop:", accuracy_drop)
    status = "STABLE"

    if accuracy_drop > drift_threshold:
        status = "DRIFT_DETECTED"
        print("Drift detected")
    else:
        print("Model performance stable.")

    monitor_log_path = os.path.join(model_dir, monitor_log)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now(),
        "training_accuracy": training_accuracy,
        "production_accuracy": production_accuracy,
        "accuracy_drop": accuracy_drop,
        "status": status
    }])

    if os.path.exists(monitor_log_path):
        log_entry.to_csv(monitor_log_path, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(monitor_log_path, index=False)


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    interval = config["monitoring"]["seconds"]

    print(f"periodic monitoring (every {interval} seconds)...")

    while True:
        run_monitoring()
        time.sleep(interval)
