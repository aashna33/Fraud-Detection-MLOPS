# Fraud-Detection-MLOPS
End-to-End IEEE-CIS Credit Card Fraud Detection Pipeline using LightGBM, MLflow, FastAPI, and Docker.

🚀 Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline for detecting fraudulent online transactions using the IEEE-CIS Fraud Detection dataset.

It covers every stage from data preprocessing → model training → experiment tracking → deployment.


🧠 Key Features

✅ Preprocessing: Missing value handling, label encoding for categorical variables


✅ Model Training: RandomForest, XGBoost, and LightGBM comparison


✅ Model Tracking: MLflow for metrics (AUC, Precision, Recall, F1)


✅ Evaluation: ROC Curve visualization


✅ Deployment: REST API with FastAPI


✅ Containerization: Dockerized inference service



🚀 Run the complete project


python src/train.py       # Train & log models


uvicorn src.app:app --reload   # Run API locally


docker build -t fraud-api . && docker run -p 8000:8000 fraud-api   # Docker deploy


