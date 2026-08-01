# Scoring service only. No jupyter, matplotlib, xgboost.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# libgomp1 is LightGBM's OpenMP runtime. Omit it and the image builds fine, then
# fails at container start with a cryptic shared-object error.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/css/ ./static/css/
COPY models/ ./models/
COPY app.py .
RUN mkdir -p logs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["gunicorn","--bind","0.0.0.0:8000","--workers","2","--timeout","120","app:app"]
