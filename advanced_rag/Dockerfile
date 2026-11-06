# ==============================================================================
# Google Cloud Run 배포용 Dockerfile
# FAISS + Multi-Query + Hybrid Search + RAG Fusion 백엔드 서버
# ==============================================================================
FROM python:3.11-slim

# 시스템 패키지 업데이트 및 FAISS 구동에 필요한 기본 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY main.py .

# Cloud Run 기본 포트 환경변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 컨테이너 실행 시 Uvicorn ASGI 서버 구동
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

