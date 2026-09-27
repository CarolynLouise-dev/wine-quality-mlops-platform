FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mlflow>=3.0.0" \
    psycopg2-binary \
    boto3 \
    gunicorn

WORKDIR /mlflow
EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlops:mlopspassword@postgres:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
