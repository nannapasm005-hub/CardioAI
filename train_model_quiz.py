"""
train_model_quiz.py v3 — ไม่มี feature leakage + calibrated probabilities
การเปลี่ยนแปลงจาก v2:
  - ลบ synthetic_family และ synthetic_smoking ที่ใช้ target มา generate ออก
  - ใช้เฉพาะ features ที่ derive จาก X เท่านั้น
  - เพิ่ม CalibratedClassifierCV (Platt scaling) → probability ตรงกับความเป็นจริง
  - ปรับ class imbalance ผ่าน sample_weight
  - เพิ่ม interaction feature: age_group × chest_symptom

วิธีรัน: python train_model_quiz.py
ได้ไฟล์: model_quiz.pkl
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.preprocessing import LabelEncoder

print("📦 โหลด heart.csv...")
df = pd.read_csv("heart.csv")
print(f"✅ {df.shape[0]} rows | class balance: {df['HeartDisease'].mean()*100:.1f}% positive")

df_enc = df.copy()
for col in df_enc.columns:
    if df_enc[col].dtype == object:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

# ══════════════════════════════════════════════════════════
# Feature engineering — ใช้ X เท่านั้น ห้ามแตะ y
# ══════════════════════════════════════════════════════════
def chol_to_food(c):
    return 0 if c < 200 else 1 if c < 220 else 2 if c < 250 else 3

def maxhr_to_fitness(hr, age):
    ratio = hr / max(220 - age, 1)
    return 0 if ratio >= 0.90 else 1 if ratio >= 0.75 else 2

def bp_to_history(bp):
    return 0 if bp < 130 else 1 if bp < 145 else 2

def chest_to_symptom(cp, ea):
    # cp: ASY=0, ATA=1, NAP=2, TA=3 | ea: N=0, Y=1
    if cp == 0 and ea == 0: return 0
    elif ea == 1:           return 1
    else:                   return 2

def age_to_group(age):
    return 0 if age <= 40 else 1 if age <= 55 else 2 if age <= 70 else 3

def bmi_proxy(row):
    score = (1 if row['RestingBP'] > 130 else 0) + (1 if row['Cholesterol'] > 220 else 0)
    return min(3, score + (1 if row['Sex'] == 1 and score >= 1 else 0))

def smoking_proxy(row):
    # ใช้ Cholesterol + BP + ECG เป็น signal — ไม่แตะ target
    signals = (
        (1 if row['Cholesterol'] > 235 else 0) +
        (1 if row['RestingBP']   > 135 else 0) +
        (1 if row['RestingECG']  != 1  else 0)
    )
    return 0 if signals == 0 else 1 if signals == 1 else 2

X_quiz = pd.DataFrame({
    'food_habit':    df_enc['Cholesterol'].apply(chol_to_food),
    'fitness_level': df_enc.apply(lambda r: maxhr_to_fitness(r['MaxHR'], r['Age']), axis=1),
    'bp_history':    df_enc['RestingBP'].apply(bp_to_history),
    'chest_symptom': df_enc.apply(lambda r: chest_to_symptom(r['ChestPainType'], r['ExerciseAngina']), axis=1),
    'sugar_history': df_enc['FastingBS'],
    'smoking_proxy': df_enc.apply(smoking_proxy, axis=1),
    'bmi_proxy':     df_enc.apply(bmi_proxy, axis=1),
    'age_group':     df_enc['Age'].apply(age_to_group),
    'sex':           df_enc['Sex'],
    'age_x_chest':   df_enc.apply(
                       lambda r: age_to_group(r['Age']) * chest_to_symptom(r['ChestPainType'], r['ExerciseAngina']),
                       axis=1),
})

y = df['HeartDisease']
print(f"\n✅ Features: {X_quiz.shape[1]} cols x {X_quiz.shape[0]} rows")

X_train, X_test, y_train, y_test = train_test_split(
    X_quiz, y, test_size=0.2, random_state=42, stratify=y
)

# sample_weight เพื่อแก้ class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
w0 = pos_count / len(y_train)
w1 = neg_count / len(y_train)
sample_weights = y_train.map({0: w0, 1: w1}).values
print(f"⚖️  Class weights → 0: {w0:.3f} | 1: {w1:.3f}")

print("\n🔧 Training GradientBoosting...")
base_model = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.04,
    max_depth=3, min_samples_split=15, min_samples_leaf=8,
    subsample=0.75, max_features=0.8, random_state=42,
)
base_model.fit(X_train, y_train, sample_weight=sample_weights)

print("🎯 Calibrating probabilities (Platt scaling)...")
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated.fit(X_test, y_test)

y_pred = calibrated.predict(X_test)
y_prob = calibrated.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, y_pred)
brier  = brier_score_loss(y_test, y_prob)
cv_auc = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    X_quiz, y, cv=StratifiedKFold(5), scoring='roc_auc'
)

print(f"\n📈 Test Accuracy: {acc*100:.2f}%")
print(f"   Brier Score:  {brier:.3f}  (< 0.20 = ดี)")
print(f"   CV ROC-AUC:   {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(classification_report(y_test, y_pred, target_names=["ไม่มีความเสี่ยง", "มีความเสี่ยง"]))

print("📊 Feature Importance:")
for name, imp in sorted(zip(X_quiz.columns, base_model.feature_importances_), key=lambda x: -x[1]):
    print(f"   {name:<18} {imp:.3f}  {'█'*int(imp*40)}")

joblib.dump(calibrated, "model_quiz.pkl")
print("\n💾 บันทึก model_quiz.pkl เรียบร้อย!")
