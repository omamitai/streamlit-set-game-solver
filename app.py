"""
SET Game Detector - iOS Optimized
================================

A Streamlit application that identifies valid SETs from uploaded images of SET card games.
Designed with Apple's Human Interface Guidelines for an optimal iPhone experience.
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
import random
import time

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SET Detector",
    page_icon="🃏",
    layout="wide"
)

# =============================================================================
# SET THEME COLORS - SET Card Game Palette
# =============================================================================
SET_THEME = {
    "primary": "#7C3AED",     # SET Purple
    "secondary": "#10B981",   # SET Green
    "accent": "#EC4899",      # SET Pink
    "red": "#EF4444",         # SET Red
    "green": "#10B981",       # SET Green
    "purple": "#8B5CF6",      # SET Light Purple
    "background": "#F4F1FA",  # Light Purple Background
    "card": "#FFFFFF",        # White
    "text": "#1F2937",        # Dark Text
    "text_muted": "#6B7280",  # Gray Text
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "processed" not in st.session_state:
    st.session_state.processed = False
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "sets_info" not in st.session_state:
    st.session_state.sets_info = None
if "is_mobile" not in st.session_state:
    st.session_state.is_mobile = True  # Default to mobile
if "should_reset" not in st.session_state:
    st.session_state.should_reset = False
if "no_cards_detected" not in st.session_state:
    st.session_state.no_cards_detected = False
if "no_sets_found" not in st.session_state:
    st.session_state.no_sets_found = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "initial"
if "image_height" not in st.session_state:
    st.session_state.image_height = 400
if "show_original" not in st.session_state:
    st.session_state.show_original = False
if "app_view" not in st.session_state:
    st.session_state.app_view = "upload"  # Possible values: "upload", "processing", "results"

# =============================================================================
# CUSTOM CSS - iOS-inspired styling
# =============================================================================
def load_custom_css():
    """
    Loads custom CSS for an iOS-inspired UI following Apple's Human Interface Guidelines.
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Text:wght@400;500;600&display=swap');

    :root {
        --set-purple: #7C3AED;
        --set-green: #10B981;
        --set-red: #EF4444;
        --set-pink: #EC4899;
        --set-light-purple: #8B5CF6;
        --set-background: #F4F1FA;
        --set-card: #FFFFFF;
        --set-text: #1F2937;
        --set-text-muted: #6B7280;
        --set-border: rgba(124, 58, 237, 0.2);
    }

    body {
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        background-color: var(--set-background);
        color: var(--set-text);
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background-image: linear-gradient(to bottom right, #F4F1FA, #F0F9FF);
    }

    /* Custom Streamlit override - zero padding for mobile */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 100%;
    }

    /* SET Game Header */
    .ios-header {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.5rem 0.75rem;
        margin: 0 auto 0.6rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.15);
        transition: all 0.2s ease;
        max-width: 180px;
    }
    
    .ios-header h1 {
        font-family: -apple-system, 'SF Pro Display', BlinkMacSystemFont, sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-light-purple) 50%, var(--set-pink) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    /* iOS Navigation Bar (Fixed at top) */
    .ios-nav-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        width: 100%;
        padding: 0.75rem 1rem;
        background: rgba(247, 247, 252, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Mobile container - full width */
    .ios-container {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Image area with iOS card styling */
    .ios-card {
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 0.75rem;
    }

    /* SET Game Primary Button */
    .ios-button-primary > button {
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-light-purple) 100%);
        color: white;
        border: none;
        padding: 0.65rem;
        border-radius: 10px;
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.1s ease-out;
        width: 100%;
        margin: 0.25rem 0 !important;
        min-height: 44px; /* iOS minimum touch target */
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.25);
    }
    
    .ios-button-primary > button:hover {
        box-shadow: 0 3px 8px rgba(124, 58, 237, 0.35);
    }
    
    .ios-button-primary > button:active {
        transform: scale(0.98);
        box-shadow: 0 1px 4px rgba(124, 58, 237, 0.2);
    }

    /* SET Game Secondary Button */
    .ios-button-secondary > button {
        background: rgba(255, 255, 255, 0.9);
        color: var(--set-purple);
        border: 1px solid rgba(124, 58, 237, 0.25);
        padding: 0.65rem;
        border-radius: 10px;
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.1s ease-out;
        width: 100%;
        margin: 0.25rem 0 !important;
        min-height: 44px; /* iOS minimum touch target */
        box-shadow: 0 1px 3px rgba(124, 58, 237, 0.1);
    }
    
    .ios-button-secondary > button:hover {
        background: rgba(255, 255, 255, 1);
        border-color: rgba(124, 58, 237, 0.4);
    }
    
    .ios-button-secondary > button:active {
        transform: scale(0.98);
        background: rgba(239, 246, 255, 1);
    }

    /* Image Container - Reduced height */
    .ios-image-container {
        margin: 0;
        position: relative;
        border-radius: 10px;
        overflow: hidden;
        height: 180px;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.1);
    }
    
    .ios-image-container img {
        width: 100%;
        height: 100%;
        object-fit: contain; /* Preserve aspect ratio */
    }
    
    /* SET Game Static Loading State */
    .ios-loader-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 160px;
        margin: 1rem 0;
    }
    
    .ios-loader-text {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--set-text-muted);
        background: rgba(255, 255, 255, 0.8);
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.1);
    }

    /* SET Game alert messages */
    .ios-alert {
        padding: 0.65rem 0.75rem;
        border-radius: 10px;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 500;
        min-height: 40px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .ios-alert-error {
        background-color: rgba(239, 68, 68, 0.08);
        color: var(--set-red);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .ios-alert-error::before {
        content: "⚠️";
        margin-right: 0.5rem;
    }
    
    .ios-alert-warning {
        background-color: rgba(245, 158, 11, 0.08);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .ios-alert-warning::before {
        content: "ℹ️";
        margin-right: 0.5rem;
    }
    
    .ios-alert-success {
        background-color: rgba(16, 185, 129, 0.08);
        color: var(--set-green);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .ios-alert-success::before {
        content: "✅";
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* iOS-style label */
    .ios-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--ios-text-muted);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Grid layout for side-by-side views */
    .ios-grid {
        display: flex;
        gap: 0.75rem;
    }
    
    .ios-column {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* Hide Streamlit elements we don't need */
    footer {
        display: none !important;
    }
    
    /* Hide the default Streamlit header/footer */
    header {
        display: none !important;
    }
    
    /* Override Streamlit image styles */
    [data-testid="stImage"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* File uploader iOS style */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem !important;
        border: 1px dashed rgba(0, 122, 255, 0.3);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stFileUploader"] > div > button {
        background-color: var(--ios-blue) !important;
        color: white !important;
        border-radius: 8px !important;
        min-height: 44px;
    }
    
    /* Remove all extra margins from button container */
    .stButton {
        margin: 0 !important; 
    }
    
    /* SET Game Results badge */
    .ios-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-light-purple) 100%);
        color: white;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.25rem;
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.2);
        letter-spacing: 0.01em;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# MODEL PATHS & LOADING
# =============================================================================
base_dir = Path("models")
char_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024" 
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Loads the Keras models for 'shape' and 'fill' classification from disk.
    Returns (shape_model, fill_model).
    """
    try:
        model_shape = load_model(str(char_path / "shape_model.keras"))
        model_fill = load_model(str(char_path / "fill_model.keras"))
        return model_shape, model_fill
    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    """
    Loads the YOLO detection models for cards and shapes from disk.
    Returns (card_detector, shape_detector).
    """
    try:
        detector_shape = YOLO(str(shape_path / "best.pt"))
        detector_shape.conf = 0.5
        detector_card = YOLO(str(card_path / "best.pt"))
        detector_card.conf = 0.5
        if torch.cuda.is_available():
            detector_card.to("cuda")
            detector_shape.to("cuda")
        return detector_card, detector_shape
    except Exception as e:
        st.error(f"Error loading detection models: {e}")
        return None, None

# Attempt to load all models
try:
    model_shape, model_fill = load_classification_models()
    detector_card, detector_shape = load_detection_models()
    models_loaded = all([model_shape, model_fill, detector_card, detector_shape])
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# =============================================================================
# UTILITY & DETECTION FUNCTIONS
# =============================================================================
def verify_and_rotate_image(board_image: np.ndarray, card_detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Checks if the detected cards are oriented primarily vertically or horizontally.
    If they're vertical, rotates the board_image 90 degrees clockwise for consistent processing.
    Returns (possibly_rotated_image, was_rotated_flag).
    """
    detection = card_detector(board_image)
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0:
        return board_image, False

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Rotate if average height > average width
    if np.mean(heights) > np.mean(widths):
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False

def restore_orientation(img: np.ndarray, was_rotated: bool) -> np.ndarray:
    """
    Restores original orientation if the image was previously rotated.
    """
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr: np.ndarray) -> str:
    """
    Rough color classification using HSV thresholds to differentiate 'red', 'green', 'purple'.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))

    # Red can wrap around hue=0, so we combine both ends
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    counts = {
        "green": cv2.countNonZero(mask_green),
        "purple": cv2.countNonZero(mask_purple),
        "red": cv2.countNonZero(mask_red),
    }
    return max(counts, key=counts.get)

def detect_cards(board_img: np.ndarray, card_detector: YOLO) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Runs YOLO on the board_img to detect card bounding boxes.
    Returns a list of (card_image, [x1, y1, x2, y2]) for each detected card.
    """
    result = card_detector(board_img)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    detected_cards = []

    for x1, y1, x2, y2 in boxes:
        detected_cards.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2]))
    return detected_cards

def predict_card_features(
    card_img: np.ndarray,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model,
    card_box: List[int]
) -> Dict:
    """
    Predicts the 'count', 'color', 'fill', 'shape' features for a single card.
    It uses a shape_detector YOLO model to locate shapes, then passes them to fill_model and shape_model.
    """
    # Detect shapes on the card
    shape_detections = shape_detector(card_img)
    c_h, c_w = card_img.shape[:2]
    card_area = c_w * c_h

    # Filter out spurious shape detections
    shape_boxes = []
    for coords in shape_detections[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = coords.astype(int)
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area:
            shape_boxes.append([x1, y1, x2, y2])

    if not shape_boxes:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': card_box
        }

    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs = []
    shape_imgs = []
    color_candidates = []

    # Prepare each detected shape region for classification
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        shape_crop = card_img[sy1:sy2, sx1:sx2]

        fill_crop = cv2.resize(shape_crop, fill_input_size) / 255.0
        shape_crop_resized = cv2.resize(shape_crop, shape_input_size) / 255.0

        fill_imgs.append(fill_crop)
        shape_imgs.append(shape_crop_resized)
        color_candidates.append(predict_color(shape_crop))

    fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=len(fill_imgs))
    shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=len(shape_imgs))

    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']

    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]

    # Take the most common color/fill/shape across all shape detections for the card
    final_color = max(set(color_candidates), key=color_candidates.count)
    final_fill = max(set(fill_result), key=fill_result.count)
    final_shape = max(set(shape_result), key=shape_result.count)

    return {
        'count': len(shape_boxes),
        'color': final_color,
        'fill': final_fill,
        'shape': final_shape,
        'box': card_box
    }

def classify_cards_on_board(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> pd.DataFrame:
    """
    Detects cards on the board, then classifies each card's features.
    Returns a DataFrame with columns: 'Count', 'Color', 'Fill', 'Shape', 'Coordinates'.
    """
    detected_cards = detect_cards(board_img, card_detector)
    card_rows = []

    for (card_img, box) in detected_cards:
        card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
        card_rows.append({
            "Count": card_feats['count'],
            "Color": card_feats['color'],
            "Fill": card_feats['fill'],
            "Shape": card_feats['shape'],
            "Coordinates": card_feats['box']
        })

    return pd.DataFrame(card_rows)

def valid_set(cards: List[dict]) -> bool:
    """
    Checks if the given 3 cards collectively form a valid SET.
    """
    for feature in ["Count", "Color", "Fill", "Shape"]:
        if len({card[feature] for card in cards}) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df: pd.DataFrame) -> List[dict]:
    """
    Finds all possible SETs from the card DataFrame.
    Each SET is a dictionary with 'set_indices' and 'cards' fields.
    """
    found_sets = []
    for combo in combinations(cards_df.iterrows(), 3):
        cards = [c[1] for c in combo]  # c is (index, row)
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [
                    {f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']}
                    for card in cards
                ]
            })
    return found_sets

def draw_detected_sets(board_img: np.ndarray, sets_detected: List[dict]) -> np.ndarray:
    """
    Annotates the board image with bounding boxes for each detected SET.
    Each SET is drawn in a different color and offset for better visibility.
    """
    # iOS Brand colors (BGR format)
    ios_colors = [
        (255, 59, 48),    # Red
        (52, 199, 89),    # Green
        (0, 122, 255),    # Blue
        (255, 149, 0),    # Orange
        (175, 82, 222),   # Purple
        (255, 45, 85)     # Pink
    ]
    
    base_thickness = 4
    base_expansion = 5

    for idx, single_set in enumerate(sets_detected):
        color = ios_colors[idx % len(ios_colors)]
        thickness = base_thickness + (idx % 3)
        expansion = base_expansion + 10 * (idx % 3)

        for i, card_info in enumerate(single_set["cards"]):
            x1, y1, x2, y2 = card_info["Coordinates"]
            # Expand the bounding box slightly
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(board_img.shape[1], x2 + expansion)
            y2e = min(board_img.shape[0], y2 + expansion)

            cv2.rectangle(board_img, (x1e, y1e), (x2e, y2e), color, thickness)

            # Label the first card with Set number
            if i == 0:
                label_bg_size = cv2.getTextSize(f"Set {idx+1}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(board_img, 
                             (x1e, y1e - 25), 
                             (x1e + label_bg_size[0] + 10, y1e),
                             color, -1)
                cv2.putText(
                    board_img,
                    f"Set {idx + 1}",
                    (x1e + 5, y1e - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
    return board_img

def identify_sets_from_image(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> Tuple[List[dict], np.ndarray]:
    """
    End-to-end pipeline to classify cards on the board and detect valid sets.
    Returns a list of sets and an annotated image.
    """
    # 1. Check and fix orientation if needed
    processed, was_rotated = verify_and_rotate_image(board_img, card_detector)

    # 2. Verify that cards are present
    cards = detect_cards(processed, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], processed

    # 3. Classify each card's features, then find sets
    df_cards = classify_cards_on_board(processed, card_detector, shape_detector, fill_model, shape_model)
    found_sets = locate_all_sets(df_cards)

    if not found_sets:
        st.session_state.no_sets_found = True
        return [], processed

    # 4. Draw sets on a copy of the image
    annotated = draw_detected_sets(processed.copy(), found_sets)

    # 5. Restore orientation if we rotated earlier
    final_output = restore_orientation(annotated, was_rotated)
    return found_sets, final_output

def optimize_image_size(img_pil: Image.Image, max_dim=800) -> Image.Image:
    """
    Resizes a PIL image to optimize for mobile viewing while preserving aspect ratio.
    """
    width, height = img_pil.size
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))

        return img_pil.resize((new_width, new_height), Image.LANCZOS)
    return img_pil

# =============================================================================
# UI RENDERING HELPERS
# =============================================================================
def render_header():
    """
    Renders a SET-themed header for the app.
    """
    header_html = """
    <div class="ios-header">
        <h1>SET Detector</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_loading():
    """
    Shows static loading message without animation.
    """
    loader_html = """
    <div class="ios-loader-container">
        <div class="ios-loader-text">Analyzing cards...</div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

def render_error(message: str):
    """
    Renders an iOS-style error message.
    """
    html = f"""
    <div class="ios-alert ios-alert-error">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_warning(message: str):
    """
    Renders an iOS-style warning message.
    """
    html = f"""
    <div class="ios-alert ios-alert-warning">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_success_message(num_sets: int):
    """
    Renders an iOS-style success message.
    """
    if num_sets == 0:
        return
        
    html = f"""
    <div class="ios-alert ios-alert-success">
        Found {num_sets} SET{'' if num_sets == 1 else 's'}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def detect_mobile_device():
    """
    Sets proper viewport for mobile devices with iOS-specific meta tags.
    """
    js_snippet = """
    <script>
        // Set proper viewport for iOS
        if (!document.querySelector('meta[name="viewport"]')) {
            var meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
            document.getElementsByTagName('head')[0].appendChild(meta);
        }
        
        // Add iOS status bar meta tags
        var statusBarMeta = document.createElement('meta');
        statusBarMeta.name = 'apple-mobile-web-app-status-bar-style';
        statusBarMeta.content = 'black-translucent';
        document.getElementsByTagName('head')[0].appendChild(statusBarMeta);
        
        // Add iOS web app capable meta tag
        var webAppMeta = document.createElement('meta');
        webAppMeta.name = 'apple-mobile-web-app-capable';
        webAppMeta.content = 'yes';
        document.getElementsByTagName('head')[0].appendChild(webAppMeta);
    </script>
    """
    st.markdown(js_snippet, unsafe_allow_html=True)
    return True

def reset_app_state():
    """
    Clears and reinitializes session state to reset the app.
    """
    # Preserve device type detection
    is_mobile = st.session_state.get("is_mobile", True)
    
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != "is_mobile":
            del st.session_state[key]

    # Now reinitialize with defaults
    st.session_state.processed = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    st.session_state.image_height = 400
    st.session_state.uploader_key = str(random.randint(1000, 9999))
    st.session_state.should_reset = True
    st.session_state.show_original = False
    st.session_state.is_mobile = is_mobile
    st.session_state.app_view = "upload"
    
# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """
    Main application entry point with iOS-style layout optimized for iPhone.
    """
    # 1. Load custom iOS-style CSS
    load_custom_css()
    
    # 2. Set proper viewport for iOS
    is_mobile = detect_mobile_device()
    st.session_state.is_mobile = is_mobile
    
    # 3. Handle app reset if needed
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # 4. Display iOS-style header
    render_header()
    
    # 5. APP FLOW - Single-screen approach for iPhone
    
    # UPLOAD SCREEN
    if not st.session_state.get("uploaded_file"):
        # Center align uploader with custom styling
        with st.container():
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 2rem;">
                <div style="width: 100px; height: 100px; border-radius: 20px; margin: 0 auto 1.25rem; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2); 
                 background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%); display: flex; justify-content: center; align-items: center;">
                <div style="font-size: 2.5rem; color: white; font-weight: bold;">SET</div>
            </div>
                <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 1.5rem; text-align: center;">
                    Upload a photo of your SET game
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # iOS-style file uploader
            st.file_uploader(
                "Upload a SET board image",
                type=["png", "jpg", "jpeg"],
                key=f"uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed"
            )
            
            if st.session_state.get("uploader_" + st.session_state.uploader_key):
                uploaded_file = st.session_state.get("uploader_" + st.session_state.uploader_key)
                
                # Reset session state for new image
                for key in ['processed', 'processed_image', 'sets_info', 'original_image',
                            'no_cards_detected', 'no_sets_found', 'show_original']:
                    if key in st.session_state:
                        if key in ('processed', 'no_cards_detected', 'no_sets_found', 'show_original'):
                            st.session_state[key] = False
                        else:
                            st.session_state[key] = None

                st.session_state.uploaded_file = uploaded_file
                try:
                    img_pil = Image.open(uploaded_file)
                    img_pil = optimize_image_size(img_pil, max_dim=800)
                    st.session_state.original_image = img_pil
                    st.session_state.image_height = img_pil.height
                    st.session_state.app_view = "preview"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load the image")
                    st.error(str(e))
    
    # PREVIEW SCREEN - Show original with Find Sets button
    elif st.session_state.app_view == "preview":
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="ios-button-secondary">', unsafe_allow_html=True)
            if st.button("Cancel", key="cancel_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Find Sets", key="find_sets_btn", use_container_width=True):
                st.session_state.app_view = "processing"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # PROCESSING SCREEN
    elif st.session_state.app_view == "processing":
        render_loading()
        
        # Process the image
        try:
            img_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
            sets_info, processed_img = identify_sets_from_image(
                img_cv, detector_card, detector_shape, model_fill, model_shape
            )
            st.session_state.sets_info = sets_info
            st.session_state.processed_image = processed_img
            st.session_state.processed = True
            st.session_state.app_view = "results"
            st.rerun()
        except Exception as e:
            render_error("Error processing image")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            
            # Add retry button
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Again", key="retry_btn", use_container_width=True):
                st.session_state.app_view = "preview"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # RESULTS SCREEN
    elif st.session_state.app_view == "results":
        # Handle error cases
        if st.session_state.no_cards_detected:
            render_error("No cards detected in the image")
            
            # Show original image
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Try again button
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Image", key="try_again_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif st.session_state.no_sets_found:
            render_warning("No valid SETs found in this game")
            
            # Show original image
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Try again button
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Image", key="no_sets_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Success case - show results
            num_sets = len(st.session_state.sets_info)
            
            # Results header with badge
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 0.4rem;">
                <div class="ios-badge">{num_sets} SET{'' if num_sets == 1 else 's'} Found</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display processed image with sets highlighted
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Single action button
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Analyze Another Card", key="new_img_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
