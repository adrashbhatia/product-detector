"""
Product/Item Detector API
--------------------------
Ek FastAPI backend jo upload ki gayi image mein products/items detect karta hai
aur unki details (label, confidence, position) return karta hai.

Model: YOLOv8n (Ultralytics) — pretrained on COCO dataset.
Pehli request par model automatically download ho jayega (~6MB).
"""

import io
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Product Detector API",
    description="Upload an image and detect products/items in it using YOLOv8.",
    version="1.0.0",
)

# Allow the frontend (served from anywhere) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "yolov8n.pt"  # nano model - small & fast, good for CPU deployment
MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

model: YOLO | None = None  # loaded lazily on startup


@app.on_event("startup")
def load_model():
    """Load (and auto-download, first run only) the YOLOv8 model once at startup."""
    global model
    model = YOLO(MODEL_PATH)


# ---------------------------------------------------------------------------
# Core detection endpoint
# ---------------------------------------------------------------------------

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, or WEBP image.",
        )

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again in a moment.")

    start = time.time()
    results = model.predict(source=image, verbose=False, conf=0.25)
    elapsed_ms = round((time.time() - start) * 1000)

    detections: List[dict] = []
    result = results[0]
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "box": {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
            }
        )

    # Sort by confidence, highest first
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return JSONResponse(
        {
            "image_size": {"width": image.width, "height": image.height},
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": elapsed_ms,
        }
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------------
# Serve the frontend (so backend + frontend deploy as ONE app)
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "static"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
