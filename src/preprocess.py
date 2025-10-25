#PREPROCESSING THE DATA

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(sample_size=200000):
    print("Loading data...")

    train_data=pd.read_csv("data/train_transaction.csv")
    identity=pd.read_csv("data/train_identity.csv")

    print(f" Train data shape: {train_data.shape}")
    print(f" identity shape: {identity.shape}")

    #MERGE 
    df=train_data.merge(identity, how="left", on="TransactionID")
    print(f" Merged shape: {df.shape}")
    print(f"Fraud ratio: {df['isFraud'].mean():.4f}")
    return df

def preprocess(df: pd.DataFrame):
  
    print("\n Starting preprocessing...")

    
    df = df.fillna(-999)

    
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"Encoding {len(cat_cols)} categorical columns...")

    le = LabelEncoder()
    for col in cat_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f" Skipping {col} due to error: {e}")

    
    y = df["isFraud"]
    X = df.drop("isFraud", axis=1)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Fraud ratio in Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print(" Running preprocessing script...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    print(" Preprocessing completed.")

