FROM python:3.11-slim

WORKDIR /app

# system deps kept minimal; libgomp needed by xgboost/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code + artifacts needed at serving time
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000
ENV MODEL_DIR=/app/models MODEL_NAME=best_model THRESHOLD=0.5

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
