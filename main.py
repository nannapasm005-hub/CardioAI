"""
main.py — FastAPI backend สำหรับ CardioAI
วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ model.pkl แล้วรัน:
  uvicorn main:app --reload --port 8000

ต้องตั้งค่า environment variable TYPHOON_API_KEY ก่อนรัน (ดู .env.example)
"""

import os
from typing import List, Literal, Optional

import httpx
import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# โหลด .env (ถ้ามี) เข้า environment — production ใช้ env จริงจาก container/host
load_dotenv()

# ── อ่าน secret จาก environment เท่านั้น ห้ามฮาร์ดโค้ด ──
# ถ้าไม่มีค่า ให้ fail ทันทีตอน startup พร้อมข้อความชัดเจน
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY")
if not TYPHOON_API_KEY:
    raise RuntimeError(
        "TYPHOON_API_KEY ไม่ได้ถูกตั้งค่า — คัดลอก .env.example เป็น .env แล้วใส่คีย์ของคุณ "
        "หรือ export TYPHOON_API_KEY ก่อนรัน uvicorn"
    )

TYPHOON_API_URL = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_MODEL = os.environ.get("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")

app = FastAPI(title="CardioAI API")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── CORS ──
# TODO(production): ตั้ง ALLOWED_ORIGINS เป็นโดเมนจริง เช่น "https://cardioai.example.com"
#   (คั่นหลายโดเมนด้วย comma) — ยังไม่รู้โดเมน ณ ตอนนี้ จึง default "*" สำหรับ dev เท่านั้น
#   ⚠️ ห้ามใช้ "*" บน production
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── โหลดโมเดล ──
model      = joblib.load("model.pkl")       # Lab mode  (clinical features)
model_quiz = joblib.load("model_quiz.pkl")  # Quiz mode (indirect features)

# ── LabelEncoder mappings (ต้องตรงกับที่ train ไว้) ──
# LabelEncoder เรียง alphabetically เสมอ
ENCODINGS = {
    "Sex":           {"F": 0, "M": 1},
    "ChestPainType": {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3},
    "RestingECG":    {"LVH": 0, "Normal": 1, "ST": 2},
    "ExerciseAngina":{"N": 0, "Y": 1},
    "ST_Slope":      {"Down": 0, "Flat": 1, "Up": 2},
}

# ── Input schema ──
# ใช้ Literal กับ field ที่เป็น enum (reject ค่านอกชุดอัตโนมัติ → 422)
# และ Field(ge/le) กำหนด range ที่สมเหตุสมผลทางคลินิก
class PatientInput(BaseModel):
    Age: int = Field(ge=0, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ASY", "ATA", "NAP", "TA"]
    RestingBP: int = Field(ge=0, le=300)
    Cholesterol: int = Field(ge=0, le=700)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["LVH", "Normal", "ST"]
    MaxHR: int = Field(ge=0, le=250)
    ExerciseAngina: Literal["N", "Y"]
    Oldpeak: float = Field(ge=-5, le=10)
    ST_Slope: Literal["Down", "Flat", "Up"]

class QuizInput(BaseModel):
    Age: int = Field(ge=0, le=120)
    Sex: Literal["M", "F"]
    food_habit: int    = Field(ge=0, le=3)
    fitness_level: int = Field(ge=0, le=2)
    bp_history: int    = Field(ge=0, le=2)
    chest_symptom: int = Field(ge=0, le=2)
    sugar_history: int = Field(ge=0, le=1)
    smoking: int        = Field(default=0, ge=0, le=2)  # ใช้คำนวณ smoking_proxy ใน backend
    family_history: int = Field(default=0, ge=0, le=2)  # เก็บไว้ใน patientContext
    bmi_category: int   = Field(default=1, ge=0, le=3)  # ใช้คำนวณ bmi_proxy ใน backend

# ── Chat schema ──
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class PatientContext(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    bp: Optional[str] = None
    chol: Optional[str] = None
    maxhr: Optional[str] = None
    riskScore: Optional[int] = None
    riskLevel: Optional[str] = None

class ChatInput(BaseModel):
    message: str
    history: List[ChatMessage] = []
    patient_context: Optional[PatientContext] = None


def build_system_prompt(ctx: Optional[PatientContext]) -> str:
    """ย้ายมาจาก buildSystemPrompt() ใน real.html — สร้าง prompt ฝั่ง backend
    โดยรับ patient context จาก request body แทนการ hardcode ในหน้าเว็บ"""
    c = ctx or PatientContext()
    age   = c.age   if c.age   is not None else "ไม่ทราบ"
    sex   = c.sex   if c.sex   is not None else "ไม่ทราบ"
    bp    = c.bp    if c.bp    is not None else "ไม่ทราบ"
    chol  = c.chol  if c.chol  is not None else "ไม่ทราบ"
    maxhr = c.maxhr if c.maxhr is not None else "ไม่ทราบ"
    level = c.riskLevel if c.riskLevel is not None else "ไม่ทราบ"
    score = c.riskScore if c.riskScore is not None else "—"

    return f"""คุณคือ "หมอเอไอ" ผู้ช่วยสุขภาพ AI ของระบบ CardioAI ที่พูดภาษาไทยเท่านั้น

ข้อมูลผู้ป่วยที่คุณกำลังดูแล:
- อายุ: {age} | เพศ: {sex}
- ความดันโลหิต: {bp}
- คอเลสเตอรอล: {chol}
- Max Heart Rate: {maxhr}
- ผลประเมิน AI: ความเสี่ยง{level} {score}%

กฎการตอบ:
- ตอบเป็นภาษาไทยเสมอ สุภาพ เป็นมิตร ใช้คำลงท้าย "ครับ"
- ตอบกระชับ ชัดเจน ไม่เกิน 4-5 ประโยค
- อ้างอิงข้อมูลของผู้ป่วยคนนี้ในคำตอบเสมอ
- ถ้าถามเรื่องรุนแรง/ฉุกเฉิน ให้แนะนำพบแพทย์ทันที
- ห้ามวินิจฉัยโรคหรือสั่งยา บอกเสมอว่านี่เป็นข้อมูลเบื้องต้น"""


@app.post("/predict")
def predict(data: PatientInput):
    # ใช้ DataFrame เพื่อให้ชื่อ column ตรงกับตอน train
    X = pd.DataFrame([{
        "Age":            data.Age,
        "Sex":            ENCODINGS["Sex"][data.Sex],
        "ChestPainType":  ENCODINGS["ChestPainType"][data.ChestPainType],
        "RestingBP":      data.RestingBP,
        "Cholesterol":    data.Cholesterol,
        "FastingBS":      data.FastingBS,
        "RestingECG":     ENCODINGS["RestingECG"][data.RestingECG],
        "MaxHR":          data.MaxHR,
        "ExerciseAngina": ENCODINGS["ExerciseAngina"][data.ExerciseAngina],
        "Oldpeak":        data.Oldpeak,
        "ST_Slope":       ENCODINGS["ST_Slope"][data.ST_Slope],
    }])
    prob = float(model.predict_proba(X)[0][1])
    score = round(prob * 100)

    if prob >= 0.7:
        level = "สูง"
        level_en = "high"
        emoji = "🔴"
        advice = "ความเสี่ยงสูงมาก ควรพบแพทย์โดยเร็วที่สุด"
    elif prob >= 0.4:
        level = "ปานกลาง"
        level_en = "medium"
        emoji = "🟡"
        advice = "มีปัจจัยเสี่ยงที่ควรระวัง แนะนำพบแพทย์ภายใน 1–2 เดือน"
    else:
        level = "ต่ำ"
        level_en = "low"
        emoji = "🟢"
        advice = "ความเสี่ยงต่ำ ดูแลสุขภาพต่อเนื่องและตรวจสุขภาพประจำปี"

    return {
        "risk_score": score,
        "risk_level": level,
        "risk_level_en": level_en,
        "emoji": emoji,
        "advice": advice,
        "probability": round(prob, 4),
    }

@app.post("/predict_quiz")
def predict_quiz(data: QuizInput):
    age_group = 0 if data.Age <= 40 else 1 if data.Age <= 55 else 2 if data.Age <= 70 else 3
    sex_enc   = 1 if data.Sex == "M" else 0

    # แปลง quiz answers → model features (ตรงกับ train_model_quiz.py v3)
    smoking_proxy = min(2, (1 if data.smoking >= 1 else 0) + (1 if data.smoking == 2 else 0))
    bmi_proxy     = min(3, (data.bmi_category - 1) if data.bmi_category > 1 else 0)
    age_x_chest   = age_group * data.chest_symptom

    X = pd.DataFrame([{
        "food_habit":    data.food_habit,
        "fitness_level": data.fitness_level,
        "bp_history":    data.bp_history,
        "chest_symptom": data.chest_symptom,
        "sugar_history": data.sugar_history,
        "smoking_proxy": smoking_proxy,
        "bmi_proxy":     bmi_proxy,
        "age_group":     age_group,
        "sex":           sex_enc,
        "age_x_chest":   age_x_chest,
    }])

    prob  = float(model_quiz.predict_proba(X)[0][1])
    score = round(prob * 100)

    if prob >= 0.7:
        level, emoji = "สูง", "🔴"
        advice = "ความเสี่ยงสูง ควรพบแพทย์และตรวจเลือดโดยเร็ว"
    elif prob >= 0.4:
        level, emoji = "ปานกลาง", "🟡"
        advice = "มีปัจจัยเสี่ยง แนะนำพบแพทย์และตรวจสุขภาพประจำปี"
    else:
        level, emoji = "ต่ำ", "🟢"
        advice = "ความเสี่ยงต่ำ ดูแลสุขภาพและออกกำลังกายสม่ำเสมอ"

    return {
        "risk_score": score, "risk_level": level,
        "emoji": emoji, "advice": advice,
        "probability": round(prob, 4), "mode": "quiz"
    }


@app.post("/chat")
async def chat(data: ChatInput):
    """Proxy ไปยัง Typhoon LLM ฝั่ง backend — คีย์ไม่หลุดไป client
    รับ {message, history, patient_context} แล้วประกอบ messages + system prompt เอง"""
    messages = [{"role": "system", "content": build_system_prompt(data.patient_context)}]
    messages += [m.model_dump() for m in data.history]
    messages.append({"role": "user", "content": data.message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                TYPHOON_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TYPHOON_API_KEY}",
                },
                json={
                    "model": TYPHOON_MODEL,
                    "max_tokens": 5000,
                    "messages": messages,
                },
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"เชื่อมต่อ LLM ไม่ได้: {e}")

    if resp.status_code != 200:
        # ไม่ส่ง body ดิบกลับไป client เพื่อกันข้อมูล upstream รั่ว
        raise HTTPException(
            status_code=502,
            detail=f"LLM upstream error (HTTP {resp.status_code})",
        )

    try:
        reply = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="รูปแบบ response จาก LLM ไม่ถูกต้อง")

    return {"reply": reply}


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest Heart Disease"}
