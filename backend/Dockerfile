# Stage 1: install dependencies + pre-download ONNX model
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."

# Bake the fastembed ONNX model into the image so there is no
# download spike on first request (which would OOM on small instances).
# Use an explicit cache path so we can reliably COPY it to the runtime stage.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Stage 2: lean runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder — no build tools in final image
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the pre-downloaded model
COPY --from=builder /app/models /app/models
ENV FASTEMBED_CACHE_PATH=/app/models

# Copy application source
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
