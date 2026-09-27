FROM apache/airflow:2.10.2-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/mlops"
