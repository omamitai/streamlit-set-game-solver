"""
Set Game Detector Streamlit App
================================

This app detects valid sets from an uploaded image of a Set game board.
It uses computer vision and machine learning models for card detection
and feature classification, then highlights the detected sets on the image.
"""

import streamlit as st
import streamlit.components.v1 as components
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
from typing import Tuple, List, Dict

# =============================================================================
# Device Detection via Query Parameter (reloads page once)
# =============================================================================
query_params = st.query_params  # Use as a property.
if "device" not in query_params:
    device_js = """
    <script>
    const device = (window.innerWidth < 768) ? "mobile" : "desktop";
    const currentUrl = window.location.href.split('?')[0];
    window.location.href = currentUrl + "?device=" + device;
    </script>
    """
    components.html(device_js, height=0)
    st.stop()
device = query_params["device"][0]
is_mobile = (device == "mobile")

# =============================================================================
# Initialize Session State (if not already set)
# =============================================================================
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.is_mobile = is_mobile

# =============================================================================
# CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(layout="wide", page_title="SET Game Detector")

# Global CSS (provided) and device-specific CSS merged below.
st.markdown(
    """
    <style>
    /* Global styles */
    body {
      background-color: #f0f2f6;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #333;
      margin: 0;
      padding: 0;
    }
    /* Header */
    .header-container {
      text-align: center;
      margin: 10px auto;
      padding: 10px;
      background-color: #e8f0fe;
      border-radius: 8px;
    }
    .header-container h1 {
      font-size: 2.5rem;
      margin: 0;
      padding: 0;
    }
    .subtitle {
      text-align: center;
      font-size: 1.25rem;
      color: #666;
      margin-top: 5px;
    }
    /* Sidebar */
    .sidebar-header {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 10px;
      text-align: center;
    }
    /* File Uploader */
    .stFileUploader {
      border: 2px dashed #aaa !important;
      padding: 10px !important;
      border-radius: 12px !important;
      text-align: center;
      background: #fff !important;
      margin-bottom: 15px;
    }
    /* Buttons */
    .stButton>button {
      display: block;
      margin: auto;
      background-color: #007BFF;
      color: #fff;
      border: none;
      padding: 0.8rem 1.2rem;
      border-radius: 6px;
      font-size: 1rem;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
      background-color: #0069d9;
    }
    /* Loading Message */
    .loading-message {
      text-align: center;
      font-size: 1.1rem;
      color: #555;
      margin-top: 10px;
      font-style: italic;
    }
    /* Images */
    img {
      max-width: 400px;
      border-radius: 10px;
      margin-top: 10px;
      display: block;
      margin-left: auto;
      margin-right: auto;
    }

    /* Device-specific CSS */
    /* Desktop: only show the sidebar uploader */
    @media screen and (min-width: 769px) {
        .mobile-uploader { display: none; }
        .sidebar-headline { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
        .arrow { text-align: center; font-size: 2rem; margin-top: 140px; }
    }
    /* Mobile: only show the main uploader */
    @media screen and (max-width: 768px) {
        .desktop-uploader { display: none; }
        .arrow { text-align: center; font-size: 2rem; margin: 20px 0; }
    }
    .center-loader { text-align: center; font-size: 1.2rem; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown(
    """
    <div class="header-container">
        <h1>🎴 SET Game Detector</h1>
        <p class="subtitle">Upload an image of a Set game board and detect valid sets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# FILE UPLOADER (Only one uploader is rendered based on device)
# =============================================================================
if not is_mobile:
    # Desktop: use sidebar uploader with a larger headline.
    with st.sidebar:
        st.markdown('<div class="desktop-uploader">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-headline">Upload Your Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="sidebar_uploader")
        st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.processed_image = None
        st.session_state.sets_info = None
else:
    # Mobile: use main-page uploader.
    st.markdown('<div class="mobile-uploader" style="text-align:center;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"], key="main_uploader")
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.processed_image = None
        st.session_state.sets_info = None

# =============================================================================
# Processing Trigger: Desktop requires button click; mobile starts automatically.
# =============================================================================
if not is_mobile:
    with st.sidebar:
        st.info("After uploading, click **Find Sets** to start processing.")
        find_sets_clicked = st.button("🔎 Find Sets", key="find_sets")
else:
    find_sets_clicked = True

# =============================================================================
# MODEL LOADING
# =============================================================================
base_dir = Path("models")
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    shape_model = load_model(str(characteristics_path / "shape_model.keras"))
    fill_model = load_model(str(characteristics_path / "fill_model.keras"))
    return shape_model, fill_model

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    shape_detection_model = YOLO(str(shape_path / "best.pt"))
    shape_detection_model.conf = 0.5
    card_detection_model = YOLO(str(card_path / "best.pt"))
    card_detection_model.conf = 0.5
    if torch.cuda.is_available():
        card_detection_model.to("cuda")
        shape_detection_model.to("cuda")
    return card_detection_model, shape_detection_model

shape_model, fill_model = load_classification_models()
card_detection_model, shape_detection_model = load_detection_models()

# =============================================================================
# UTILITY & PROCESSING FUNCTIONS
# =============================================================================
def check_and_rotate_input_image(board_image: np.ndarray, detector: YOLO) -> Tuple[np.ndarray, bool]:
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
    if was_rotated:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def predict_color(shape_image: np.ndarray) -> str:
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
    purple_mask = cv2.inRange(hsv_image, np.array([120, 50, 50]), np.array([160, 255, 255]))
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    color_counts = {
        "green": cv2.countNonZero(green_mask),
        "purple": cv2.countNonZero(purple_mask),
        "red": cv2.countNonZero(red_mask)
    }
    return max(color_counts, key=color_counts.get)

def predict_card_features(card_image: np.ndarray, shape_detector: YOLO,
                          fill_model: tf.keras.Model, shape_model: tf.keras.Model,
                          box: List[int]) -> Dict:
    shape_results = shape_detector(card_image)
    card_h, card_w = card_image.shape[:2]
    card_area = card_w * card_h
    filtered_boxes = [
        [int(x1), int(y1), int(x2), int(y2)]
        for x1, y1, x2, y2 in shape_results[0].boxes.xyxy.cpu().numpy()
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area
    ]
    if not filtered_boxes:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': box}
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    fill_imgs, shape_imgs, color_list = [], [], []
    for fb in filtered_boxes:
        x1, y1, x2, y2 = fb
        shape_img = card_image[y1:y2, x1:x2]
        fill_img = cv2.resize(shape_img, tuple(fill_input_shape)) / 255.0
        shape_img_resized = cv2.resize(shape_img, tuple(shape_input_shape)) / 255.0
        fill_imgs.append(fill_img)
        shape_imgs.append(shape_img_resized)
        color_list.append(predict_color(shape_img))
    fill_imgs = np.array(fill_imgs)
    shape_imgs = np.array(shape_imgs)
    fill_preds = fill_model.predict(fill_imgs, batch_size=len(fill_imgs))
    shape_preds = shape_model.predict(shape_imgs, batch_size=len(shape_imgs))
    fill_labels_list = ['empty', 'full', 'striped']
    shape_labels_list = ['diamond', 'oval', 'squiggle']
    predicted_fill = [fill_labels_list[np.argmax(pred)] for pred in fill_preds]
    predicted_shape = [shape_labels_list[np.argmax(pred)] for pred in shape_preds]
    color_label = max(set(color_list), key=color_list.count)
    fill_label = max(set(predicted_fill), key=predicted_fill.count)
    shape_label = max(set(predicted_shape), key=predicted_shape.count)
    return {
        'count': len(filtered_boxes),
        'color': color_label,
        'fill': fill_label,
        'shape': shape_label,
        'box': box
    }

def is_set(cards: List[dict]) -> bool:
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        if len({card[feature] for card in cards}) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> List[dict]:
    sets_found = []
    for combo in combinations(card_df.iterrows(), 3):
        cards = [entry[1] for entry in combo]
        if is_set(cards):
            sets_found.append({
                'set_indices': [entry[0] for entry in combo],
                'cards': [
                    {feature: card[feature] for feature in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']}
                    for card in cards
                ]
            })
    return sets_found

def detect_cards_from_image(board_image: np.ndarray, detector: YOLO) -> List[tuple]:
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [
        (board_image[y1:y2, x1:x2], [x1, y1, x2, y2])
        for x1, y1, x2, y2 in card_boxes
    ]

def classify_cards_from_board_image(board_image: np.ndarray, card_detector: YOLO,
                                      shape_detector: YOLO, fill_model: tf.keras.Model,
                                      shape_model: tf.keras.Model) -> pd.DataFrame:
    cards = detect_cards_from_image(board_image, card_detector)
    card_data = []
    for card_image, box in cards:
        features = predict_card_features(card_image, shape_detector, fill_model, shape_model, box)
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": features['box']
        })
    return pd.DataFrame(card_data)

def draw_sets_on_image(board_image: np.ndarray, sets_info: List[dict]) -> np.ndarray:
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    base_thickness = 8
    base_expansion = 5
    for index, set_info in enumerate(sets_info):
        color = colors[index % len(colors)]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        for i, card in enumerate(set_info["cards"]):
            x1, y1, x2, y2 = card["Coordinates"]
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(board_image.shape[1], x2 + expansion)
            y2_exp = min(board_image.shape[0], y2 + expansion)
            cv2.rectangle(board_image, (x1_exp, y1_exp), (x2_exp, y2_exp), color, thickness)
            if i == 0:
                cv2.putText(
                    board_image, f"Set {index + 1}", (x1_exp, y1_exp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness
                )
    return board_image

def classify_and_find_sets_from_array(
    board_image: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> tuple:
    processed_image, was_rotated = check_and_rotate_input_image(board_image, card_detector)
    card_df = classify_cards_from_board_image(processed_image, card_detector, shape_detector, fill_model, shape_model)
    sets_found = find_sets(card_df)
    annotated_image = draw_sets_on_image(processed_image.copy(), sets_found)
    final_image = restore_original_orientation(annotated_image, was_rotated)
    return sets_found, final_image

# =============================================================================
# DISPLAY: Original Image & Detected Sets (Original remains visible)
# =============================================================================
if st.session_state.uploaded_file is None:
    st.info("Please upload an image using the appropriate uploader.")
else:
    if is_mobile:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, width=400, output_format="JPEG")
        detected_placeholder = st.empty()
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.original_image, width=400, output_format="JPEG")
        detected_placeholder = col2.empty()

    # =============================================================================
    # IMAGE PROCESSING (Desktop via button; Mobile automatically)
    # =============================================================================
    run_processing = is_mobile or find_sets_clicked
    if run_processing:
        loader_placeholder = st.empty()
        loader_placeholder.markdown('<div class="center-loader">Detecting sets...</div>', unsafe_allow_html=True)
        try:
            image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
            sets_info, processed_image = classify_and_find_sets_from_array(
                image_cv,
                card_detection_model,
                shape_detection_model,
                fill_model,
                shape_model,
            )
            st.session_state.processed_image = processed_image
            st.session_state.sets_info = sets_info
            cards = detect_cards_from_image(image_cv, card_detection_model)
            if not cards:
                st.error("No cards detected. Please verify that this is a valid Set game board.")
            elif not sets_info:
                st.warning("Cards detected but no valid sets found.")
            else:
                st.success("Sets detected!")
        except Exception as e:
            st.error("⚠️ An error occurred during processing:")
            st.text(traceback.format_exc())
        finally:
            loader_placeholder.empty()

    # =============================================================================
    # DISPLAY: Detected Sets with Arrow (layout adjusts per device)
    # =============================================================================
    if is_mobile:
        st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)
        st.subheader("Detected Sets")
        if st.session_state.processed_image is not None:
            detected_placeholder.image(
                cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
                width=400, output_format="JPEG"
            )
        else:
            detected_placeholder.info("Processed image will appear here after detection.")
    else:
        st.markdown('<div class="arrow" style="text-align:center;">➡️</div>', unsafe_allow_html=True)
        with detected_placeholder:
            st.subheader("Detected Sets")
            if st.session_state.processed_image is not None:
                st.image(
                    cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
                    width=400, output_format="JPEG"
                )
            else:
                st.info("Processed image will appear here after detection.")
