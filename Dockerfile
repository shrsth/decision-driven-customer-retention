# Decision-Driven Customer Retention — container image.
# Builds the model on first run; serves the Streamlit dashboard.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; wheels cover numpy/scipy/scikit-learn/lifelines.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# The dashboard self-bootstraps: it runs the pipeline (downloads data, trains
# the model) on first launch if the artifacts are missing.
CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
