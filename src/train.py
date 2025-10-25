from preprocess import load_data, preprocess
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import os 

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    
    print(f"\nTraining {model_name} ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"{model_name} - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.sklearn.log_model(model, "model")


def main():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    
    mlflow.set_experiment("fraud-detection")


    rf = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced_subsample",
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    train_and_log_model("RandomForest", rf, X_train, X_test, y_train, y_test)

  
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=25,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    train_and_log_model("XGBoost", xgb, X_train, X_test, y_train, y_test)
    


    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=25,
        random_state=42
    )
    train_and_log_model("LightGBM", lgbm, X_train, X_test, y_train, y_test)
    #os.makedirs("models", exist_ok=True)   
    lgbm.fit(X_train, y_train)             
    lgbm.booster_.save_model("models/lightgbm_model.txt")
    print(" LightGBM model saved successfully to models/lightgbm_model.txt")

    print("\nAll models logged to MLflow! Use 'mlflow ui' to compare results.")


if __name__ == "__main__":
    main()

