FROM python:3.11-slim

# system deps some of the ML libs need (prophet/pmdarima build chains)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor/ ./predictor/

# TCG_DATABASE_URL is injected at runtime as a secret env var.
# The scheduler/platform invokes this command on its schedule.
CMD ["python", "-m", "predictor.run_all"]
