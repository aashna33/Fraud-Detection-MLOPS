# Fraud-Detection-MLOPS
End-to-End IEEE-CIS Credit Card Fraud Detection Pipeline using LightGBM, MLflow, FastAPI, and Docker.

🚀 Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline for detecting fraudulent online transactions using the IEEE-CIS Fraud Detection dataset.

It covers every stage from data preprocessing → model training → experiment tracking → deployment.

🧩 Project Structure
📦 mlops_project/
│
├── src/
│   ├── preprocess.py         # Loads and preprocesses the dataset
│   ├── train.py              # Trains RandomForest, XGBoost, and LightGBM + logs to MLflow
│   ├── visualize.py          # Plots ROC Curve
│   ├── app.py                # FastAPI app for model serving
│
├── models/
│   └── lightgbm_model.txt    # Saved trained LightGBM model
│
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│
├── Dockerfile                # For containerizing the FastAPI app
├── requirements.txt          # All dependencies
└── README.md

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
