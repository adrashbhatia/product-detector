FROM python:3.11-slim

# System deps needed by OpenCV (used internally by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the model at build time so the first request isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
