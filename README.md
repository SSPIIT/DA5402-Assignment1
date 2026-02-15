Name: Swara
Roll No: MM22B045

Manual MLOps – Predictive Maintenance

This project implements a complete manual MLOps pipeline for a Predictive Maintenance system using Python, Git, CSV files, and FastAPI. No automated MLOps tools such as MLflow, DVC, or Airflow were used.

Project Structure:

The project contains separate folders for data, models, and source code.
The data folder includes raw, processed, and production datasets.
The models folder stores the trained model and metadata files.
The src folder contains scripts for data preparation, training, deployment, drift simulation, production inference, and monitoring.
All paths and parameters are controlled through config.yaml.
Deployment history is stored in deployment_log.csv.

Phases Implemented:

Phase A – Data Management
Manual data versioning was implemented. All file paths and parameters are read from config.yaml. No hardcoded values were used.

Phase B – Model Registry
The trained model is saved in the models folder. Metadata such as training date and accuracy is stored in metadata.json. Model versions are logged for reproducibility.

Phase C – Deployment
The model is wrapped using FastAPI and exposed through a prediction endpoint. Deployment activity is recorded in a log file.

Phase D – Monitoring and Drift
Drift is simulated using a separate script. Production data is sent to the API and predictions are logged. A monitoring script calculates production error rate and compares it with training error. If the error increase crosses a threshold, retraining is required.

How to Run:

Install dependencies using pip install -r requirements.txt

Train the model using python3 src/train.py

Start the API using uvicorn src.inference:app --port 5000 --reload
then it runs on http://127.0.0.1:5000

Simulate drift using python3 src/create_drift.py

Run monitoring using python3 src/monitor.py

