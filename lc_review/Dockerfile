FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run 기본 포트 8080 사용
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
