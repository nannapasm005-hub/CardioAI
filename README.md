# 🫀 CardioAI

> ระบบประเมินความเสี่ยงโรคหัวใจเบื้องต้นด้วย **Machine Learning + LLM ภาษาไทย**
> ผู้ใช้กรอกข้อมูลสุขภาพ → โมเดลทำนายระดับความเสี่ยง → "หมอเอไอ" อธิบายผลและให้คำแนะนำเป็นภาษาไทย

`FastAPI` · `scikit-learn` · `Typhoon LLM` · `Docker`

---

## 📌 ภาพรวม

CardioAI ช่วยให้คนทั่วไปประเมินความเสี่ยงโรคหัวใจเบื้องต้นได้ด้วยตัวเอง โดยรองรับผู้ใช้ 2 กลุ่มที่มีข้อมูลต่างกัน:

- **Lab mode** — สำหรับคนที่มีผลตรวจสุขภาพ (ความดัน, คอเลสเตอรอล, ECG ฯลฯ) ใช้ข้อมูลคลินิกจริงทำนายแม่นยำ
- **Quiz mode** — สำหรับคนที่ไม่มีผลแล็บ ตอบแบบสอบถามพฤติกรรม (อาหาร, ออกกำลังกาย, ประวัติครอบครัว) แล้วประเมินจาก proxy
- **หมอเอไอ** — แชตบอตภาษาไทยที่อ้างอิงผลประเมินของผู้ใช้ ตอบคำถามต่อยอดและเตือนให้พบแพทย์เมื่อจำเป็น

## 🧠 โมเดล Machine Learning

ใช้ dataset [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (918 เคส) แล้ว train แยก 2 โมเดลตามลักษณะ input:

| | Lab mode | Quiz mode |
|---|----------|-----------|
| Algorithm | Random Forest | Gradient Boosting + Platt calibration |
| Input | 11 ฟีเจอร์คลินิก | คำตอบแบบสอบถาม → proxy features |
| ผลลัพธ์ | Accuracy ~89% | Accuracy ~83%, Brier 0.13, ROC-AUC 0.86 |

**จุดที่ตั้งใจออกแบบ:**
- Lab model จัดการ class imbalance ด้วย `class_weight="balanced"` และคุม overfitting ด้วย `max_depth`, `min_samples_leaf`
- Quiz model ใช้ **calibrated probability** (Platt scaling) เพื่อให้ค่าความเสี่ยงที่แสดงสะท้อนความน่าจะเป็นจริง ไม่ใช่แค่ลำดับ
- Feature engineering ของ Quiz mode derive ฟีเจอร์จากข้อมูลคลินิกล้วน **ไม่แตะ target** เพื่อเลี่ยง data leakage

## 🏗️ สถาปัตยกรรม

```
Browser (real.html)
   │  POST /predict         ─►  FastAPI ─►  Random Forest        ─►  risk score
   │  POST /predict_quiz    ─►  FastAPI ─►  Gradient Boosting    ─►  risk score
   │  POST /chat            ─►  FastAPI ─►  Typhoon LLM          ─►  คำอธิบายภาษาไทย
   ▼
ผลประเมิน + คำแนะนำ
```

**การตัดสินใจเชิงออกแบบที่สำคัญ:**
- **คีย์ LLM อยู่ฝั่ง backend เท่านั้น** — frontend เรียกผ่าน `/chat` ไม่เคยเห็นคีย์ และคีย์อ่านจาก environment variable (ไม่ฝังในโค้ด)
- **Input validation ระดับคลินิก** — ใช้ Pydantic `Field` กำหนดช่วงค่าที่เป็นไปได้ (เช่น Age 0–120, Cholesterol 0–700) และ `Literal` คุม enum → ส่งค่าผิดได้ HTTP 422 พร้อมเหตุผล
- **Config แยกจากโค้ด** — คีย์, รุ่นโมเดล, และ CORS origins ตั้งผ่าน env ทำให้ย้าย environment ได้โดยไม่แก้โค้ด
- โมเดลถูกตรวจสอบความสอดคล้องของ label encoding ด้วย test เพื่อกันโค้ดพังเงียบๆ เมื่อ library อัปเดต

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.11), เสิร์ฟทั้ง REST API และหน้าเว็บ
- **ML:** scikit-learn (Random Forest, Gradient Boosting, CalibratedClassifierCV)
- **LLM:** Typhoon (โมเดลภาษาไทย) ผ่าน backend proxy
- **Frontend:** HTML/CSS/JS (single page)
- **Deploy:** Docker + docker-compose

## 🚀 รันในเครื่อง

```bash
# 1. ติดตั้ง dependencies (แนะนำใช้ venv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. ตั้งค่าคีย์
cp .env.example .env          # แก้ TYPHOON_API_KEY เป็นคีย์ของคุณ

# 3. รัน (โมเดลถูก commit มาแล้ว ใช้งานได้เลย)
uvicorn main:app --reload --port 8000
```

เปิด http://localhost:8000

> อยาก train โมเดลใหม่เอง: วาง `heart.csv` (จาก Kaggle) แล้วรัน `python train_model-3.py && python train_model_quiz.py`

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## ✅ Tests

```bash
pip install -r requirements-dev.txt
pytest
```

ครอบคลุมทุก endpoint (happy path + edge case) และทดสอบความถูกต้องของ label encoding — รันได้จาก clean clone โดยไม่ต้องมี dataset จริง (mock โมเดล + LLM)

## ⚠️ ข้อจำกัด

ผลลัพธ์เป็นการประเมินเบื้องต้นเพื่อการศึกษาเท่านั้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้**

## 🌍 SDG

**SDG 3 — Good Health and Well-Being:** ส่งเสริมการดูแลสุขภาพเชิงป้องกัน ให้คนเข้าถึงการประเมินความเสี่ยงเบื้องต้นได้ง่ายขึ้น
