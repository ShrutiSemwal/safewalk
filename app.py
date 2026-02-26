import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeWalk AI",
    page_icon="🚶",
    layout="wide"
)

OBSTACLE_CLASSES = ["person", "car", "motorcycle", "bicycle", "truck", "bus"]

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ============================================================
# CORE COMPUTER VISION MODULES
# ============================================================

def detect_obstacles(image):
    """
    Detect dynamic and static obstacles using YOLOv8.
    Returns obstacle count and annotated image.
    """
    results = model(image)

    annotated_img = image.copy()
    obstacle_count = 0
    shown_labels = set()  # to track which labels already displayed

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in OBSTACLE_CLASSES:
                obstacle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    annotated_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),  # Red in BGR
                    2
                )

                if cls_name not in shown_labels:
                    cv2.putText(
                        annotated_img,
                        cls_name,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA
                    )
                    shown_labels.add(cls_name)

    return obstacle_count, annotated_img


def detect_cracks(image):
    """
    Crack estimation using edge density approximation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    crack_ratio = np.sum(edges > 0) / edges.size
    return crack_ratio


def detect_green_coverage(image):
    """
    Estimate environmental greenery using HSV segmentation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = np.sum(mask > 0) / mask.size
    return green_ratio


def estimate_walkable_space(crack_ratio, obstacle_count):
    """
    Estimate walkable safety factor.
    Combines structural damage and obstruction density.
    """
    obstruction_penalty = min(obstacle_count * 0.05, 0.4)
    structural_penalty = crack_ratio * 0.6

    walkable_index = 1 - (obstruction_penalty + structural_penalty)
    return max(0, walkable_index)


# ============================================================
# RESEARCH-STYLE SAFETY SCORING MODEL
# ============================================================

def compute_safety_score(crack_ratio, obstacle_count, green_ratio, walkable_index):
    """
    Weighted multi-factor pedestrian safety model.

    Weight Distribution (Research-inspired heuristic):
    - Structural Integrity: 35%
    - Obstruction Risk: 30%
    - Walkability Index: 25%
    - Environmental Quality: 10%
    """

    structural_score = (1 - crack_ratio) * 35
    obstruction_score = max(0, (1 - obstacle_count * 0.08)) * 30
    walkability_score = walkable_index * 25
    environmental_score = green_ratio * 10

    final_score = structural_score + obstruction_score + walkability_score + environmental_score

    return round(max(0, min(100, final_score)), 2)


# ============================================================
# PIPELINE
# ============================================================

def analyze_image(image_path):
    image = cv2.imread(image_path)

    obstacle_count, annotated_img = detect_obstacles(image)
    crack_ratio = detect_cracks(image)
    green_ratio = detect_green_coverage(image)
    walkable_index = estimate_walkable_space(crack_ratio, obstacle_count)

    safety_score = compute_safety_score(
        crack_ratio,
        obstacle_count,
        green_ratio,
        walkable_index
    )

    return (
        obstacle_count,
        crack_ratio,
        green_ratio,
        walkable_index,
        safety_score,
        annotated_img
    )

# ============================================================
# USER INTERFACE
# ============================================================

st.title("🚶 SafeWalk AI")
st.markdown("### AI-Based Sidewalk Pedestrian Safety Analyzer")

st.markdown("""
SafeWalk evaluates sidewalk safety using computer vision and 
multi-factor risk modeling.

### System Components
- **YOLOv8** for obstacle detection  
- Edge-density based crack estimation  
- HSV-based green coverage estimation  
- Walkability modeling  
- Weighted risk scoring algorithm  
""")

st.divider()

# Sidebar Info
st.sidebar.header("Project Overview")

st.sidebar.markdown("""
### Problem Statement
Urban pedestrian pathways suffer from:
- Cracks and Structural degradation
- Obstruction hazards (vehicles, objects)
- Reduced walkable width
- Poor green infrastructure

### Objective
Quantify pedestrian safety using AI-based visual analytics.

### **Model Used**
- YOLOv8 (Ultralytics)
- OpenCV Image Processing
- Rule-based Safety Scoring

### Output Metrics
- Obstacle Density
- Cracks and Structural Damage Ratio
- Green Coverage
- Walkability Index
- Composite Safety Score (0–100)
""")

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a Sidewalk Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        image_path = tmp_file.name

    with st.spinner("Running AI Safety Assessment..."):
        obs, crack, green, walkable, score, annotated = analyze_image(image_path)

    st.divider()

    # ============================================================
    # IMAGE DISPLAY
    # ============================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)

    with col2:
        st.subheader("AI Detection Output")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.divider()

    # ============================================================
    # Detection Summary
    # ============================================================

    st.subheader("Detection Summary")

    unique_obstacles = {}
    
    for r in model(cv2.imread(image_path)):
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in OBSTACLE_CLASSES:
                unique_obstacles[cls_name] = unique_obstacles.get(cls_name, 0) + 1
    
    if unique_obstacles:
        for k, v in unique_obstacles.items():
            st.write(f"• {k.title()} detected: {v}")
    else:
        st.write("No major obstacles detected.")

    st.divider()

    # ============================================================
    # METRICS
    # ============================================================

    st.subheader("Quantitative Safety Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Crack Ratio (Structural Damage)", f"{round(crack*100,2)}%")
        st.metric("Obstacle Count", obs)
    
    with col2:
        st.metric("Green Coverage", f"{round(green*100,2)}%")
        st.metric("Walkability Index", f"{round(walkable*100,2)}%")

    st.divider()

    # ============================================================
    # SAFETY SCORE
    # ============================================================

    st.subheader("Composite Pedestrian Safety Score")

    score_color = (
    "green" if score >= 80
    else "orange" if score >= 60
    else "red"
    )
    
    st.markdown(
        f"""
        <h1 style='text-align:center; color:{score_color};'>
        {score}
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.progress(score / 100)
    
    if score >= 80:
        st.success("Low Risk Environment")
    elif score >= 60:
        st.warning("Moderate Risk Environment")
    else:
        st.error("High Risk Environment")
