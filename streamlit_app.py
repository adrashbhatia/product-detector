"""
Product/Item Detector — Streamlit version
-------------------------------------------
Ye app Streamlit Community Cloud par free deploy ho sakti hai, bina
kisi card ya phone verification ke — sirf GitHub login chahiye.
"""

import io
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="Product Detector", page_icon="🔎", layout="centered")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

st.title("🔎 Product / Item Detector")
st.caption("Image upload karo — YOLOv8 items detect karke details dega.")

uploaded_file = st.file_uploader("Image choose karo", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting..."):
        start = time.time()
        results = model.predict(source=image, verbose=False, conf=0.25)
        elapsed_ms = round((time.time() - start) * 1000)

    result = results[0]
    detections = []
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    colors = ["#E8A33D", "#4FA88F", "#D9704F", "#7C9CBF", "#C77DBF", "#8AC24A"]

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        color = colors[i % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{label} {confidence*100:.0f}%", fill=color)

        detections.append({"label": label, "confidence": round(confidence, 4)})

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    st.image(draw_image, use_container_width=True)
    st.caption(f"{len(detections)} item(s) detected · {elapsed_ms}ms")

    if detections:
        st.subheader("Detected items")
        for d in detections:
            st.write(f"**{d['label'].title()}** — {d['confidence']*100:.1f}% confidence")
    else:
        st.info("Koi item detect nahi hua. Clearer ya closer image try karo.")
else:
    st.info("Upar se ek image upload karo shuru karne ke liye.")
