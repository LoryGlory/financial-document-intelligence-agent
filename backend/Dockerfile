# Stage 1: install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
# Install the package and all dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."

# Stage 2: lean runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder — no build tools in final image
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
