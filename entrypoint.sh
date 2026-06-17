#!/usr/bin/env sh
# entrypoint.sh — เช็คไฟล์โมเดลก่อน start server
# โมเดล (.pkl) เป็น generated artifact ถูก gitignore ไว้ จึงไม่มาจาก clone
set -e

missing=""
[ -f model.pkl ]      || missing="$missing model.pkl"
[ -f model_quiz.pkl ] || missing="$missing model_quiz.pkl"

if [ -n "$missing" ]; then
  echo "❌ ไม่พบไฟล์โมเดล:$missing" >&2
  echo "" >&2
  echo "โมเดลเป็น generated artifact (ถูก gitignore) จึงไม่มาพร้อม repo" >&2
  echo "วิธีแก้ — train ก่อนแล้วค่อย build/run:" >&2
  echo "  1) วาง heart.csv (จาก Kaggle: fedesoriano/heart-failure-prediction) ในโฟลเดอร์นี้" >&2
  echo "  2) python train_model-3.py        # -> model.pkl" >&2
  echo "  3) python train_model_quiz.py     # -> model_quiz.pkl" >&2
  echo "  4) build ใหม่ (docker compose up --build) หรือ mount ไฟล์ .pkl เข้า /app" >&2
  exit 1
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
