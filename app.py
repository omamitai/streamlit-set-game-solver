import streamlit as st
from PIL import Image
import numpy as np
import cv2
import traceback
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from itertools import combinations
from pathlib import Path

# --------------------------------------------------------------------------
# 🎨 Streamlit Page Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 📜 Inject Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# --------------------------------------------------------------------------
# 🔍 Model Loading
# --------------------------------------------------------------------------
base_dir = Path("models")
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

shape_model = load_model(str(characteristics_path / "shape_model.keras"))
fill_model = load_model(str(characteristics_path / "fill_model.keras"))

shape_detection_model = YOLO(str(shape_path / "best.pt"))
shape_detection_model.yaml = str(shape_path / "data.yaml")
shape_detection_model.conf = 0.5

card_detection_model = YOLO(str(card_path / "best.pt"))
card_detection_model.yaml = str(card_path / "data.yaml")
card_detection_model.conf = 0.5

if torch.cuda.is_available():
    card_detection_model.to('cuda')
    shape_detection_model.to('cuda')

# --------------------------------------------------------------------------
# 🛠 Utility & Processing Functions
# --------------------------------------------------------------------------
def check_and_rotate_input_image(board_image: np.ndarray, detector) -> (np.ndarray, bool):
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    if card_boxes.size == 0:
        return board_image, False
    widths = card_boxes[:, 2] - card_boxes[:, 0]
    heights = card_boxes[:, 3] - card_boxes[:, 1]
    if np.mean(heights) > np.mean(widths):
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    return board_image, False

def restore_original_orientation(image: np.ndarray, was_rotated: bool) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if was_rotated else image

def predict_color(shape_image: np.ndarray) -> str:
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
    purple_mask = cv2.inRange(hsv_image, np.array([120, 50, 50]), np.array([160, 255, 255]))
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    color_counts = {'green': cv2.countNonZero(green_mask), 'purple': cv2.countNonZero(purple_mask), 'red': cv2.countNonZero(red_mask)}
    return max(color_counts, key=color_counts.get)

def is_set(cards: list) -> bool:
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        if len({card[feature] for card in cards}) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> list:
    sets_found = []
    for combo in combinations(card_df.iterrows(), 3):
        cards = [entry[1] for entry in combo]
        if is_set(cards):
            sets_found.append({
                'set_indices': [entry[0] for entry in combo],
                'cards': [{feature: card[feature] for feature in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} for card in cards]
            })
    return sets_found

def detect_cards_from_image(board_image: np.ndarray, detector) -> list:
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [(board_image[y1:y2, x1:x2], [x1, y1, x2, y2]) for x1, y1, x2, y2 in card_boxes]

# --------------------------------------------------------------------------
# 🎲 Streamlit App UI
# --------------------------------------------------------------------------

# 🌟 Title & Description
st.markdown("<h1 class='title'>🎴 SET Game Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image of a SET game board and click <b>Find Sets</b> to detect valid sets.</p>", unsafe_allow_html=True)

# 🔹 Layout: Two Equal Columns
col1, col2 = st.columns(2, gap="medium")

# 📥 Left Column: Upload Image
with col1:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        # 🖼 Display the uploaded image with "Original Image" title
        st.markdown("### 🖼 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="🎴 Original Image", use_container_width=True, output_format="JPEG")

        # 📍 Find Sets Button
        find_sets_clicked = st.button("🔎 Find Sets", use_container_width=True)

# 🔍 Right Column: Processed Image Output
if uploaded_file and find_sets_clicked:
    with col2:
        st.markdown("### 🔍 Processed Result")
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("🔄 Processing... Please wait."):
                sets_info, final_image = classify_and_find_sets_from_array(
                    image_cv,
                    card_detection_model,
                    shape_detection_model,
                    fill_model,
                    shape_model
                )

            # Convert image back to RGB for display
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            st.image(final_image_rgb, caption="✅ Detected Sets", use_container_width=True, output_format="JPEG")

            # 📌 Expandable Results
            with st.expander("📜 View Detected Sets Details"):
                st.json(sets_info)

            # 🏆 Success Message
            st.toast("🎉 Sets detected successfully!", icon="✅")

            # 🔄 Clear Button (Resets the app)
            if st.button("🔄 Clear & Upload New Image", use_container_width=True):
                st.experimental_rerun()

        except Exception as e:
            st.error("⚠️ An error occurred during processing:")
            st.text(traceback.format_exc())
