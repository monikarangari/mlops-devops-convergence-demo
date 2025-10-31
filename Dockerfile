FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + models (train in CI and bake the artifact)
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8080
CMD ["uvicorn", "src.infer_service:app", "--host", "0.0.0.0", "--port", "8080"]
