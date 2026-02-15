import requests
import pandas as pd
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["data"]["target_column"]
port = config["deployment"]["port"]

production_dir = config["data"]["production_dir"]
drift_file = config["versions"]["drift"]

drift_path = os.path.join(production_dir, drift_file)

df = pd.read_csv(drift_path)

results = []

for _, row in df.iterrows():
    payload = {
        "features": row.drop(target_col).tolist()
    }

    response = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json=payload
    )

    response_data = response.json()
    prediction = response_data["prediction"]

    results.append({
        "actual": row[target_col],
        "prediction": prediction
    })

log_df = pd.DataFrame(results)
log_file = config["monitoring"]["api_log"]
log_path = os.path.join(production_dir, log_file)
log_df.to_csv(log_path, index=False)

print("predictions logged.")
