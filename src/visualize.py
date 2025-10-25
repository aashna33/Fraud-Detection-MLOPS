import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Loading limited data for ROC visualization...")


trans = pd.read_csv("data/train_transaction.csv", nrows=100000)
iden = pd.read_csv("data/train_identity.csv")
df = trans.merge(iden, on="TransactionID", how="left")

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Fraud ratio: {df['isFraud'].mean():.4f}")


df = df.fillna(-999)
le = LabelEncoder()
for col in df.select_dtypes("object").columns:
    try:
        df[col] = le.fit_transform(df[col].astype(str))
    except Exception:
        continue

y = df["isFraud"]
X = df.drop("isFraud", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


print("Training LightGBM model (sample)...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=25,
    random_state=42
)
model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM Fraud Detection (100K Sample)')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig("roc_curve.png", dpi=300)
plt.show()

print(f"ROC curve saved as 'roc_curve.png' (AUC = {roc_auc:.4f})")
