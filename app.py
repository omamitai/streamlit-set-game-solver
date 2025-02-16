"""
Set Game Detector Streamlit App
================================

This app detects valid sets from an uploaded image of a Set game board.
It uses computer vision and machine learning models for card detection
and feature classification, then highlights the detected sets on the image.
"""

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
from typing import Tuple, List, Dict

# =============================================================================
#                           CONFIGURATION & STYLE
# =============================================================================

st.set_page_config(layout="wide", page_title="SET Game Detector")

# Load custom CSS (styles.css can contain your global styling)
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Could not load custom CSS.")

local_css("styles.css")

# Additional inline CSS for arrow, loader, and mobile adjustments.
st.markdown(
    """
    <style>
    /* Arrow styling */
    .desktop-arrow {
        display: block;
        text-align: center;
        font-size: 2rem;
        margin-top: 140px;
    }
    .mobile-arrow {
        display: none;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 20px;
    }
    /* Loader styling for mobile */
    .center-loader {
        text-align: center;
        font-size: 1.2rem;
        margin: 20px 0;
    }
    @media screen and (max-width: 768px) {
        .desktop-arrow {
            display: none;
        }
        .mobile-arrow {
            display: block;
        }
        /* Mobile uploader is always visible on main page */
        .mobile-uploader {
            display: block;
            margin: 20px auto;
            text-align: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
#                           HEADER SECTION
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
#                FILE UPLOADER & DEVICE VERSION DETECTION
# =============================================================================

# Desktop uploader in sidebar.
st.sidebar.markdown('<div class="sidebar-header">Upload Your Image</div>', unsafe_allow_html=True)
uploaded_file_side = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"], key="sidebar_uploader")

# Mobile uploader on main page.
if "uploaded_file" not in st.session_state:
    st.markdown('<div class="mobile-uploader"><strong>Upload Your Image</strong></div>', unsafe_allow_html=True)
mobile_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="main_uploader")

# Decide which uploader is being used.
if uploaded_file_side is not None:
    st.session_state.uploaded_file = uploaded_file_side
    st.session_state.is_mobile = False
    try:
        st.session_state.original_image = Image.open(uploaded_file_side)
        st.session_state.processed_image = None
        st.session_state.sets_info = None
    except Exception as e:
        st.error("Failed to load image. Please try another file.")
elif mobile_file is not None:
    st.session_state.uploaded_file = mobile_file
    st.session_state.is_mobile = True
    try:
        st.session_state.original_image = Image.open(mobile_file)
        st.session_state.processed_image = None
        st.session_state.sets_info = None
    except Exception as e:
        st.error("Failed to load image. Please try another file.")

# For desktop, show a Find Sets button.
is_mobile = st.session_state.get("is_mobile", False)
find_sets_clicked = False
if not is_mobile:
    find_sets_clicked = st.sidebar.button("🔎 Find Sets", key="find_sets")

# =============================================================================
#                              MODEL LOADING
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
#                    UTILITY & PROCESSING FUNCTIONS
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
#                           IMAGE PROCESSING & DISPLAY
# =============================================================================

if "uploaded_file" not in st.session_state:
    st.info("Please upload an image from the sidebar or above.")
else:
    try:
        original_image = st.session_state.original_image.copy()
    except Exception as e:
        st.error("Failed to load image. Please try another file.")
        st.exception(e)
        st.stop()

    # Determine processing trigger:
    # For mobile, process automatically upon upload.
    # For desktop, process only if the "Find Sets" button is clicked.
    run_processing = False
    if st.session_state.get("is_mobile", False):
        run_processing = True
    else:
        run_processing = find_sets_clicked

    if run_processing:
        # For mobile, show a centered loader.
        if st.session_state.get("is_mobile", False):
            loader_placeholder = st.empty()
            loader_placeholder.markdown('<div class="center-loader">Detecting sets...</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown("<p class='loading-message'>Detecting sets...</p>", unsafe_allow_html=True)
        try:
            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            sets_info, processed_image = classify_and_find_sets_from_array(
                image_cv,
                card_detection_model,
                shape_detection_model,
                fill_model,
                shape_model,
            )
            st.session_state.processed_image = processed_image
            st.session_state.sets_info = sets_info
            # Feedback messages:
            cards = detect_cards_from_image(image_cv, card_detection_model)
            if not cards:
                msg = "No cards detected. Please verify that this is a valid Set game board."
                if st.session_state.get("is_mobile", False):
                    st.error(msg)
                else:
                    st.sidebar.warning(msg)
            elif not sets_info:
                msg = "Cards detected but no valid sets found."
                if st.session_state.get("is_mobile", False):
                    st.warning(msg)
                else:
                    st.sidebar.warning(msg)
            else:
                msg = "Sets detected!"
                if st.session_state.get("is_mobile", False):
                    st.success(msg)
                else:
                    st.sidebar.success(msg)
        except Exception as e:
            st.error("⚠️ An error occurred during processing:")
            st.text(traceback.format_exc())
        finally:
            if st.session_state.get("is_mobile", False):
                loader_placeholder.empty()
            else:
                st.sidebar.empty()

    # Layout: For desktop, use two columns; for mobile, they stack.
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Original Image")
        st.image(original_image, width=400, output_format="JPEG")
    # Arrow: on mobile, show above detected sets.
    if st.session_state.get("is_mobile", False):
        st.markdown('<div class="mobile-arrow">⬇️</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="desktop-arrow">➡️</div>', unsafe_allow_html=True)
    with right_col:
        st.subheader("Detected Sets")
        if "processed_image" in st.session_state and st.session_state.processed_image is not None:
            processed_image_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, width=400, output_format="JPEG")
        else:
            st.info("Processed image will appear here after detection.")
