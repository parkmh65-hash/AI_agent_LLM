FROM python:3.10-slim

작업 디렉토리 설정 및 패키지 설치

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

소스 코드 복사

COPY main.py .

Cloud Run의 기본 포트인 8080을 통해 uvicorn 서버 실행

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
