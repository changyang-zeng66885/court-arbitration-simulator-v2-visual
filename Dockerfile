FROM python:3.10-slim

LABEL description="基于RAG和Multi-Agent技术的商事仲裁模拟系统"

# Install minimal system dependencies
# procps is needed for the healthcheck (pgrep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (all ship pre-built wheels, no compiler needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data
COPY . .

# Ensure the output directory exists
RUN mkdir -p /app/outputs/streamlit_runs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pgrep -f "streamlit run" > /dev/null || exit 1

# Default LLM configuration — override via docker run -e or docker-compose
ENV OPENAI_API_KEY=""
ENV OPENAI_BASE_URL=""
ENV OPENAI_MODEL="qwen3.6-flash"

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]
