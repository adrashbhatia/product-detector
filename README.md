# Product Detector — Image Detection Agent

Image upload karo, ye agent products/items detect karke label, confidence
aur position (bounding box) return karta hai. Backend + frontend dono ek
hi app mein hain, isliye deploy karna simple hai.

**Model:** YOLOv8n (Ultralytics), COCO dataset par pretrained. Ye 80 common
object/item classes detect karta hai — jaise bottle, laptop, cell phone,
backpack, handbag, book, chair, clock, cup, keyboard, remote, tv, suitcase,
umbrella, aur bahut kuch. Pehli request par model khud download ho jata hai
(~6MB), koi extra setup nahi chahiye.

## Project structure

```
product-detector/
├── backend/
│   ├── app.py              ← FastAPI server (API + frontend serve karta hai)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── static/
│       └── index.html      ← Web app (upload UI + results)
└── README.md
```

## 1. Local run karna

```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Browser mein kholo: **http://localhost:8000**

Image upload karo — turant detected items ki list aur image par bounding
boxes dikhengi.

## 2. API directly use karna

```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@/path/to/image.jpg"
```

Response:
```json
{
  "image_size": {"width": 1280, "height": 720},
  "detections": [
    {"label": "backpack", "confidence": 0.91, "box": {"x1": 120, "y1": 45, "x2": 430, "y2": 610}},
    {"label": "cell phone", "confidence": 0.78, "box": {"x1": 500, "y1": 200, "x2": 590, "y2": 340}}
  ],
  "count": 2,
  "inference_time_ms": 340
}
```

## 3. Deploy karna (free options)

### Option A — Render.com (sabse easy)

1. Is folder ko GitHub repo mein push karo.
2. [render.com](https://render.com) par **New → Web Service** banao, apna
   repo connect karo.
3. Settings:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
4. Deploy karo — 5-10 min mein live ho jayega. Free tier CPU par chalega
   (slow cold start, but works).

### Option B — Railway.app

1. GitHub repo connect karo, root directory `backend` set karo.
2. Railway `Dockerfile` ko auto-detect kar lega — bas deploy button dabao.

### Option C — Hugging Face Spaces (Docker SDK)

1. New Space banao, SDK = **Docker**.
2. `backend/` folder ka content (Dockerfile, app.py, requirements.txt,
   static/) Space ke root mein upload karo.
3. Space apne aap build ho ke live URL de dega.

Teeno options mein deploy hone ke baad ek hi URL par web app bhi milega
aur `/api/detect` API endpoint bhi — same app dono kaam karta hai.

## Notes / limitations

- YOLOv8n COCO ke 80 fixed classes tak limited hai — agar tumhe koi
  bahut specific product (jaise ek particular shoe brand) detect karna ho,
  to us par custom-trained model chahiye hoga, jo alag scope hai.
- Free hosting tiers par CPU inference thoda slow ho sakta hai
  (~300-800ms per image) — normal hai, GPU ki zaroorat nahi.
- `/api/detect` 10MB tak ki JPEG/PNG/WEBP images accept karta hai.
