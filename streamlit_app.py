"""
InsightLens — Item Detection Engine
-------------------------------------
Image upload karo, ye AI detection engine items/products detect karke
label, confidence, aur count deta hai.
"""

import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="InsightLens", page_icon="🔍", layout="centered")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a2029 0%, #12151A 55%);
    }

    .il-hero {
        padding: 18px 0 6px 0;
        border-bottom: 1px solid #2E3640;
        margin-bottom: 24px;
    }
    .il-hero h1 {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #EDEBE4;
    }
    .il-hero h1 span {
        color: #E8A33D;
    }
    .il-hero p {
        color: #8A93A0;
        font-size: 14px;
        margin: 4px 0 0 0;
        font-family: ui-monospace, monospace;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1B2027;
        border: 1px dashed #3A424D;
        border-radius: 8px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #E8A33D;
    }

    .il-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #1E2530;
        border-left: 3px solid #E8A33D;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .il-card .il-label {
        font-size: 15px;
        color: #EDEBE4;
        text-transform: capitalize;
    }
    .il-card .il-conf {
        font-family: ui-monospace, monospace;
        font-size: 13px;
        color: #4FA88F;
    }

    .il-meta {
        font-family: ui-monospace, monospace;
        font-size: 13px;
        color: #8A93A0;
        margin: 10px 0 18px 0;
    }

    .il-empty {
        color: #8A93A0;
        font-family: ui-monospace, monospace;
        font-size: 13px;
        padding: 10px 0;
    }
    </style>

    <div class="il-hero">
        <h1>Insight<span>Lens</span></h1>
        <p>AI-powered item detection engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image to scan",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Scanning image..."):
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

        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(image.width * 0.003)))
        tag = f"{label} {confidence*100:.0f}%"
        text_h = max(18, int(image.width * 0.025))
        draw.rectangle([x1, max(0, y1 - text_h), x1 + len(tag) * 9 + 10, y1], fill=color)
        draw.text((x1 + 5, max(0, y1 - text_h) + 2), tag, fill="#12151A")

        detections.append({"label": label, "confidence": round(confidence, 4)})

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    st.image(draw_image, use_container_width=True)
    st.markdown(
        f'<div class="il-meta">{len(detections)} item(s) found &middot; {elapsed_ms}ms</div>',
        unsafe_allow_html=True,
    )

    if detections:
        cards_html = "".join(
            f'<div class="il-card"><span class="il-label">{d["label"]}</span>'
            f'<span class="il-conf">{d["confidence"]*100:.1f}%</span></div>'
            for d in detections
        )
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="il-empty">No items detected. Try a clearer or closer shot.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="il-empty">Upload an image above to get started.</div>', unsafe_allow_html=True)
