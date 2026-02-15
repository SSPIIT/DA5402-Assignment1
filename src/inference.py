import joblib
import yaml
import os
import csv
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_dir = config["deployment"]["model_dir"]
model_name = config["deployment"]["model_name"]
deployment_log_path = config["deployment"]["deployment_log"]
model_path = os.path.join(model_dir, model_name)
model = joblib.load(model_path)

if not os.path.exists(deployment_log_path):
    with open(deployment_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model_version"])

with open(deployment_log_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([datetime.now(), model_name])

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}

@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != 11:
        return {
            "error": f"Expected 11 features, got {len(data.features)}"
        }

    prediction = model.predict([data.features])

    return {
        "model_version": model_name,
        "prediction": int(prediction[0])
    }

