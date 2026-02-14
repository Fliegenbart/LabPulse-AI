FROM python:3.11-slim

WORKDIR /app

# System deps for reportlab, kaleido, prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory for persistent storage
RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.fileWatcherType=none"]
