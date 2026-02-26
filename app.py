import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

OBSTACLE_CLASSES = ["person", "car", "motorcycle", "bicycle", "truck", "bus"]

def analyze(image_path):
    image = cv2.imread(image_path)

    results = model(image)
    obstacle_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in OBSTACLE_CLASSES:
                obstacle_count += 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    crack_ratio = np.sum(edges>0)/edges.size

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(35,40,40),(85,255,255))
    green_ratio = np.sum(mask>0)/mask.size

    free_ratio = 1 - crack_ratio

    score = 100 - crack_ratio*50 - obstacle_count*10
    score += green_ratio*10
    score = max(0,min(100,score))

    return obstacle_count, crack_ratio, green_ratio, free_ratio, score

st.title("🚶 SafeWalk")
st.markdown("### AI-Based Sidewalk Pedestrian Safety Analyzer")

st.markdown(
"""
SafeWalk analyzes pedestrian walkways using computer vision 
to evaluate structural safety, obstruction risk, and environmental quality.

It combines:
- **YOLOv8** for obstacle detection  
- Crack detection using edge analysis  
- Green coverage estimation  
- Walkable free-space estimation  
"""
)

st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("About This Project")

st.sidebar.markdown("""
**Problem Statement**

Urban sidewalks often suffer from:
- Cracks and structural damage
- Obstructions (vehicles, objects)
- Poor walkable width
- Lack of green integration

This tool estimates pedestrian safety using AI.

---

**Model Used**
- YOLOv8 (Ultralytics)
- OpenCV Image Processing
- Rule-based Safety Scoring

---

**Output**
- Crack %
- Obstacle count
- Green coverage %
- Walkable space %
- Safety Score (0–100)
""")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "📤 Upload a sidewalk image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        image_path = tmp_file.name

    with st.spinner("Analyzing sidewalk safety..."):
        obs, crack, green, free, score = analyze(image_path)

    st.divider()

    # ----------- DISPLAY IMAGES -----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)

    with col2:
        st.subheader("Detection Output")
        st.image(image_path, use_column_width=True)

    st.divider()

    # ----------- METRICS -----------
    st.subheader("Safety Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Crack %", f"{round(crack*100,2)}%")
    m2.metric("Obstacles", obs)
    m3.metric("Green %", f"{round(green*100,2)}%")
    m4.metric("Free Space %", f"{round(free*100,2)}%")

    st.divider()

    # ----------- SAFETY SCORE -----------
    st.subheader("Overall Safety Score")

    if score >= 75:
        st.success(f"Safety Score: {score} → Low Risk")
    elif score >= 50:
        st.warning(f"Safety Score: {score} → Moderate Risk")
    else:
        st.error(f"Safety Score: {score} → High Risk")



# uploaded = st.file_uploader("Upload sidewalk image")

# if uploaded:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded.read())
#     obs, crack, green, free, score = analyze(tfile.name)

#     st.write("Obstacles:", obs)
#     st.write("Crack %:", crack*100)
#     st.write("Green %:", green*100)
#     st.write("Safety Score:", score)
