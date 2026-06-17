"""
test_api.py — happy path + edge case ต่อ endpoint
รัน: pytest -q
"""

VALID_PATIENT = {
    "Age": 58, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
    "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150,
    "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat",
}

VALID_QUIZ = {
    "Age": 60, "Sex": "M", "food_habit": 2, "fitness_level": 1, "bp_history": 1,
    "chest_symptom": 2, "sugar_history": 1, "smoking": 2, "family_history": 1,
    "bmi_category": 2,
}


# ── /health ──────────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /predict ─────────────────────────────────────────────
def test_predict_happy(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    # response format ที่ renderResult() ใน frontend คาดหวัง
    for key in ("risk_score", "risk_level", "emoji", "advice", "probability"):
        assert key in body
    assert 0 <= body["risk_score"] <= 100


def test_predict_bad_enum_returns_422(client):
    bad = {**VALID_PATIENT, "Sex": "X"}        # ไม่ใช่ M/F
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range_returns_422(client):
    bad = {**VALID_PATIENT, "Cholesterol": 9999}   # เกิน 700
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


# ── /predict_quiz ────────────────────────────────────────
def test_predict_quiz_happy(client):
    r = client.post("/predict_quiz", json=VALID_QUIZ)
    assert r.status_code == 200
    body = r.json()
    for key in ("risk_score", "risk_level", "emoji", "advice"):
        assert key in body
    assert body.get("mode") == "quiz"


def test_predict_quiz_out_of_range_returns_422(client):
    bad = {**VALID_QUIZ, "food_habit": 9}      # 0-3 เท่านั้น
    r = client.post("/predict_quiz", json=bad)
    assert r.status_code == 422


# ── /chat ────────────────────────────────────────────────
def test_chat_happy(client, mock_typhoon):
    mock_typhoon(status=200)
    r = client.post("/chat", json={
        "message": "ผลของผมน่ากังวลไหม",
        "history": [],
        "patient_context": {"age": "58 ปี", "sex": "ชาย", "riskScore": 85, "riskLevel": "สูง"},
    })
    assert r.status_code == 200
    assert "reply" in r.json()
    assert r.json()["reply"]


def test_chat_upstream_error_returns_502(client, mock_typhoon):
    mock_typhoon(status=401)                   # คีย์ผิด/หมดอายุ ฝั่ง upstream
    r = client.post("/chat", json={"message": "hi", "history": []})
    assert r.status_code == 502
    # ต้องไม่ leak body ดิบของ upstream
    assert "fake-body" not in r.text


def test_chat_bad_history_role_returns_422(client):
    r = client.post("/chat", json={
        "message": "hi",
        "history": [{"role": "system", "content": "x"}],   # อนุญาตแค่ user/assistant
    })
    assert r.status_code == 422


# ── ENCODINGS ตรงกับ LabelEncoder จริง ─────────────────────
def test_encodings_match_label_encoder(app_main):
    """กันโค้ดพังเงียบๆ ถ้า sklearn เปลี่ยนวิธีเรียง:
    LabelEncoder เรียง alphabetically — ENCODINGS ใน main.py ต้องตรงกับผลนั้น"""
    from sklearn.preprocessing import LabelEncoder

    # ค่าที่เป็นไปได้ของแต่ละคอลัมน์ (ตรงกับ heart.csv)
    raw_values = {
        "Sex":            ["M", "F", "M", "F"],
        "ChestPainType":  ["TA", "ATA", "NAP", "ASY"],
        "RestingECG":     ["Normal", "ST", "LVH"],
        "ExerciseAngina": ["Y", "N"],
        "ST_Slope":       ["Up", "Flat", "Down"],
    }
    for col, values in raw_values.items():
        le = LabelEncoder().fit(values)
        expected = {label: int(code) for label, code in zip(le.classes_, le.transform(le.classes_))}
        assert app_main.ENCODINGS[col] == expected, (
            f"ENCODINGS['{col}'] ไม่ตรงกับ LabelEncoder: {app_main.ENCODINGS[col]} != {expected}"
        )
