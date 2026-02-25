import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

model = YOLO("yolov8n.pt")

OBSTACLE_CLASSES = ["person", "car", "motorcycle", "bicycle", "truck", "bus"]

def analyze(image_path):
    image = cv2.imread(image_path)

    results = model(image_path)
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

st.title("SafeWalk AI")

uploaded = st.file_uploader("Upload sidewalk image")

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    obs, crack, green, free, score = analyze(tfile.name)

    st.write("Obstacles:", obs)
    st.write("Crack %:", crack*100)
    st.write("Green %:", green*100)
    st.write("Safety Score:", score)
