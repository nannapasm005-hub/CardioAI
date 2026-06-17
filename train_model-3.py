"""
train_model.py - ใช้กับ Kaggle Heart Failure Prediction Dataset
Download: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
วางไฟล์ heart.csv ไว้ในโฟลเดอร์เดียวกันก่อนรัน
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("📦 กำลังโหลด Heart Failure Dataset...")
df = pd.read_csv("heart.csv")
print(f"✅ Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}")

# แยก X, y
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
print(f"   Positive cases: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")

# แปลง string columns → numeric
for col in X.columns:
    if X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"✅ Features ready: {X.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("\n🔧 กำลัง Train Model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["ไม่มีความเสี่ยง", "มีความเสี่ยง"]))

joblib.dump(model, "model.pkl")
print("💾 บันทึก model.pkl เรียบร้อยแล้ว!")
print("\n🚀 ต่อไป: python train_model_quiz.py แล้ว uvicorn main:app --reload --port 8000")
