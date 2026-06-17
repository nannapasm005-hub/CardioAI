FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (ดู .dockerignore — ไม่รวม .env, .git ฯลฯ)
# จะ copy .pkl / heart.csv เข้ามาด้วยถ้ามีในโฟลเดอร์ build context
COPY . .

# ถ้ามี heart.csv และยังไม่มีโมเดล → train ตอน build ให้เลย
# ถ้าไม่มี heart.csv ก็ข้าม แล้วไปเช็คอีกที (fail ชัดๆ) ที่ entrypoint ตอน start
RUN if [ -f heart.csv ]; then \
        [ -f model.pkl ]      || python train_model-3.py ; \
        [ -f model_quiz.pkl ] || python train_model_quiz.py ; \
    else \
        echo "⚠️  ไม่พบ heart.csv ตอน build — จะเช็คโมเดลอีกครั้งตอน start (ดู entrypoint.sh)" ; \
    fi

# Serve HTML ด้วย FastAPI static files
RUN mkdir -p static && cp real.html static/index.html

# คีย์ TYPHOON_API_KEY ส่งผ่าน environment ตอน run เท่านั้น (ไม่ baked เข้า image)
EXPOSE 8000

RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
