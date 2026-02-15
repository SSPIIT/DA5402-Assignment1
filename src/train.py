import os
import yaml
import json
import joblib
import subprocess
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_dir = config["data"]["processed_dir"]
train_file = config["versions"]["train"]
target_column = config["data"]["target_column"]
model_params = config["model"]
model_dir = config["deployment"]["model_dir"]
model_name = config["deployment"]["model_name"]
metadata_name = config["deployment"]["metadata_name"]
metadata_log_name = config["deployment"]["metadata_log_name"]
train_path = os.path.join(processed_dir, train_file)

df = pd.read_csv(train_path)

X = df.drop(target_column, axis=1)
y = df[target_column]

model = RandomForestClassifier(
    n_estimators=model_params["n_estimators"],
    max_depth=model_params["max_depth"],
    random_state=model_params["random_state"]
)

model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"Training Accuracy: {accuracy}")

os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_name)
joblib.dump(model, model_path)

try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode("utf-8").strip()
except:
    git_hash = "not_available"

metadata = {
    "project": config["project"]["name"],
    "author": config["project"]["author"],
    "training_date": str(datetime.now()),
    "dataset_used": train_path,
    "git_commit_hash": git_hash,
    "accuracy": accuracy,
    "model_params": model_params
}

metadata_path = os.path.join(model_dir, metadata_name)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

metadata_log_path = os.path.join(model_dir, metadata_log_name)

with open(metadata_log_path, "a") as f:
    f.write(json.dumps(metadata) + "\n")
print("Model and metadata saved")
