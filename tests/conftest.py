"""
conftest.py — pytest fixtures สำหรับ CardioAI

ทดสอบได้จาก clean clone โดยไม่ต้องมี heart.csv หรือ .pkl จริง:
  - stub joblib.load ด้วยโมเดลปลอม (predict_proba คืนค่าคงที่)
  - mock การเรียก Typhoon API (httpx) ไม่ให้ยิงเน็ตจริง
"""
import os

import numpy as np
import pytest

# ต้องมี key (main.py raise ตอน startup ถ้าไม่มี) — ใช้ค่า dummy ใน test
os.environ.setdefault("TYPHOON_API_KEY", "test-key")


class _FakeModel:
    """โมเดลปลอมแทน sklearn — predict_proba คืน prob คงที่"""
    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.array([[1.0 - self.prob, self.prob]] * n)


@pytest.fixture(scope="session")
def app_main():
    # ต้อง patch joblib.load ก่อน import main (โหลดโมเดลตอน import)
    import joblib
    joblib.load = lambda path: _FakeModel(0.5 if "quiz" in str(path) else 0.85)
    import main
    return main


@pytest.fixture
def client(app_main):
    from fastapi.testclient import TestClient
    return TestClient(app_main.app)


@pytest.fixture
def mock_typhoon(app_main, monkeypatch):
    """mock httpx.AsyncClient ที่ /chat ใช้ — ตั้ง status/payload ได้ต่อ test"""
    state = {
        "status": 200,
        "json": {"choices": [{"message": {"content": "สวัสดีครับ นี่เป็นข้อมูลเบื้องต้น"}}]},
    }

    class FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = "fake-body"

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResp(state["status"], state["json"])

    monkeypatch.setattr(app_main.httpx, "AsyncClient", FakeClient)

    def setter(status=200, payload=None):
        state["status"] = status
        if payload is not None:
            state["json"] = payload

    return setter
