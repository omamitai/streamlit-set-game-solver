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
# CUSTOM CSS - Enhanced iOS-inspired styling
# =============================================================================
def load_custom_css():
    """
    Loads enhanced custom CSS for an iOS-inspired UI following Apple's Human Interface Guidelines.
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Text:wght@400;500;600&display=swap');

    :root {
        --set-purple: #7C3AED;
        --set-green: #10B981;
        --set-red: #EF4444;
        --set-pink: #EC4899;
        --set-light-purple: #8B5CF6;
        --set-background: #F8F5FF;
        --set-card: #FFFFFF;
        --set-text: #1F2937;
        --set-text-muted: #6B7280;
        --set-border: rgba(124, 58, 237, 0.25);
    }

    body {
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        background-color: var(--set-background);
        color: var(--set-text);
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background-image: linear-gradient(135deg, #F4F1FA 0%, #F0F9FF 50%, #F9F5FF 100%);
    }

    /* Custom Streamlit override - minimal padding for mobile */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 100%;
    }

    /* Enhanced header style with glassmorphism effect */
    .ios-header {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.7rem 1.2rem;
        margin: 0 auto 0.75rem;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        box-shadow: 0 3px 12px rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.2);
        transition: all 0.2s ease;
        max-width: 85%;
    }
    
    .ios-header h1 {
        font-family: -apple-system, 'SF Pro Display', BlinkMacSystemFont, sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-pink) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        text-shadow: 0 1px 2px rgba(124, 58, 237, 0.1);
    }
    
    /* iOS Navigation Bar (Fixed at top) */
    .ios-nav-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        width: 100%;
        padding: 0.75rem 1rem;
        background: rgba(248, 245, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(124, 58, 237, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Mobile container - full width */
    .ios-container {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Image area with enhanced iOS card styling */
    .ios-card {
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(124, 58, 237, 0.1);
        margin-bottom: 0.75rem;
    }

    /* Enhanced SET-themed buttons */
    .ios-button-primary > button {
        background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%);
        color: white;
        border: none;
        padding: 0.7rem;
        border-radius: 14px;
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.15s ease-out;
        width: 100%;
        margin: 0.25rem 0 !important;
        min-height: 50px; /* Increased touch target */
        box-shadow: 0 3px 12px rgba(124, 58, 237, 0.35);
        letter-spacing: 0.01em;
        text-transform: uppercase;
    }
    
    .ios-button-primary > button:hover {
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.45);
        transform: translateY(-1px);
    }
    
    .ios-button-primary > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(124, 58, 237, 0.3);
    }

    /* Secondary button with SET theme */
    .ios-button-secondary > button {
        background: linear-gradient(135deg, #EC4899 0%, #F472B6 100%);
        color: white;
        border: none;
        padding: 0.7rem;
        border-radius: 14px;
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.15s ease-out;
        width: 100%;
        margin: 0.25rem 0 !important;
        min-height: 50px; /* Increased touch target */
        box-shadow: 0 3px 12px rgba(236, 72, 153, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.01em;
    }
    
    .ios-button-secondary > button:hover {
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
        transform: translateY(-1px);
    }
    
    .ios-button-secondary > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(236, 72, 153, 0.25);
    }
    
    /* Find Sets button variant with enhanced styling */
    .find-sets-button > button {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.3);
    }
    
    .find-sets-button > button:hover {
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .find-sets-button > button:active {
        box-shadow: 0 2px 5px rgba(16, 185, 129, 0.25);
    }

    /* Enhanced image container with adaptive sizing */
    .ios-image-container {
        margin: 0.75rem auto;
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        max-height: 300px; /* Higher but still fits on iPhone */
        height: calc(100vh - 250px); /* Adaptive height */
        min-height: 180px;
        width: 92%;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
        box-shadow: 0 3px 12px rgba(124, 58, 237, 0.2);
        border: 1px solid rgba(124, 58, 237, 0.25);
    }
    
    .ios-image-container img {
        width: 100%;
        height: 100%;
        object-fit: contain; /* Preserve aspect ratio */
    }
    
    /* Improved SET Game animated loader with glassmorphism */
    .ios-loader-container {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(244, 241, 250, 0.7); /* Lighter background using SET theme */
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        z-index: 10;
        border-radius: 16px; /* Match container's radius */
    }
    
    .ios-loader {
        width: 32px;
        height: 32px;
        border: 3px solid rgba(124, 58, 237, 0.2);
        border-top: 3px solid var(--set-purple);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 0.8rem;
        box-shadow: 0 0 10px rgba(124, 58, 237, 0.15);
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .ios-loader-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--set-purple);
        background: rgba(255, 255, 255, 0.85);
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.2);
    }

    /* Enhanced SET Game alert messages */
    .ios-alert {
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        font-size: 0.9rem;
        font-weight: 600;
        min-height: 44px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    }
    
    .ios-alert-error {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--set-red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    
    .ios-alert-error::before {
        content: "⚠️";
        margin-right: 0.6rem;
        font-size: 1.1rem;
    }
    
    .ios-alert-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    
    .ios-alert-warning::before {
        content: "ℹ️";
        margin-right: 0.6rem;
        font-size: 1.1rem;
    }
    
    .ios-alert-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--set-green);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    .ios-alert-success::before {
        content: "✅";
        margin-right: 0.6rem;
        font-size: 1.1rem;
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
    
    /* Enhanced file uploader custom styling */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 16px;
        padding: 1rem !important;
        border: 1px dashed rgba(124, 58, 237, 0.35);
        box-shadow: 0 3px 8px rgba(124, 58, 237, 0.1);
    }
    
    [data-testid="stFileUploader"] > div > button {
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-light-purple) 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        min-height: 48px;
        box-shadow: 0 3px 8px rgba(124, 58, 237, 0.25);
    }
    
    /* Remove all extra margins from button container */
    .stButton {
        margin: 0 !important; 
    }
    
    /* Enhanced SET Game Results badge */
    .ios-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-light-purple) 100%);
        color: white;
        border-radius: 14px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.25rem;
        box-shadow: 0 3px 6px rgba(124, 58, 237, 0.25);
        letter-spacing: 0.01em;
    }

    /* Card info tooltip with glassmorphism effect */
    .ios-tooltip {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.15);
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--set-text);
        max-width: 200px;
        position: absolute;
        z-index: 20;
    }

    /* Animated pulsing indicator for detected sets */
    .set-indicator {
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--set-green);
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
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
    # Enhanced SET-themed colors (BGR format)
    set_colors = [
        (68, 68, 239),    # Red
        (89, 199, 16),    # Green
        (246, 92, 139),   # Pink
        (0, 149, 255),    # Orange
        (222, 82, 175),   # Purple
        (85, 45, 255)     # Hot Pink
    ]
    
    base_thickness = 4
    base_expansion = 5

    for idx, single_set in enumerate(sets_detected):
        color = set_colors[idx % len(set_colors)]
        thickness = base_thickness + (idx % 3)
        expansion = base_expansion + 10 * (idx % 3)

        for i, card_info in enumerate(single_set["cards"]):
            x1, y1, x2, y2 = card_info["Coordinates"]
            # Expand the bounding box slightly
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(board_img.shape[1], x2 + expansion)
            y2e = min(board_img.shape[0], y2 + expansion)

            # Draw more attractive rounded rectangle with gradient effect
            cv2.rectangle(board_img, (x1e, y1e), (x2e, y2e), color, thickness)
            
            # Add corners with thicker lines for iOS-style appearance
            corner_length = min(30, (x2e - x1e) // 4)
            corner_thickness = thickness + 1
            
            # Top-left corner
            cv2.line(board_img, (x1e, y1e + corner_length), (x1e, y1e), color, corner_thickness)
            cv2.line(board_img, (x1e, y1e), (x1e + corner_length, y1e), color, corner_thickness)
            
            # Top-right corner
            cv2.line(board_img, (x2e - corner_length, y1e), (x2e, y1e), color, corner_thickness)
            cv2.line(board_img, (x2e, y1e), (x2e, y1e + corner_length), color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(board_img, (x1e, y2e - corner_length), (x1e, y2e), color, corner_thickness)
            cv2.line(board_img, (x1e, y2e), (x1e + corner_length, y2e), color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(board_img, (x2e - corner_length, y2e), (x2e, y2e), color, corner_thickness)
            cv2.line(board_img, (x2e, y2e), (x2e, y2e - corner_length), color, corner_thickness)

            # Label the first card with Set number with enhanced styling
            if i == 0:
                # Create pill-shaped badge background
                text = f"Set {idx+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                badge_width = text_size[0] + 16
                badge_height = 26
                badge_x = x1e
                badge_y = y1e - badge_height - 5
                
                # Draw pill-shaped badge
                cv2.rectangle(board_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              color, -1, cv2.LINE_AA)
                
                # Draw text in center of badge
                cv2.putText(
                    board_img,
                    text,
                    (badge_x + 8, badge_y + badge_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
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

def optimize_image_size(img_pil: Image.Image, max_dim=700) -> Image.Image:
    """
    Resizes a PIL image to optimize for mobile viewing while preserving aspect ratio.
    Reduced max_dim to 700 to ensure better fit on iPhone screens.
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
    Renders a stylish SET-themed header with enhanced glassmorphism effect.
    """
    header_html = """
    <div class="ios-header">
        <h1>SET Card Game Detector</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_loading():
    """
    Shows enhanced animated loading spinner centered on the image.
    """
    loader_html = """
    <div class="ios-loader-container">
        <div class="ios-loader"></div>
        <div class="ios-loader-text">Analyzing cards...</div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

def render_error(message: str):
    """
    Renders an enhanced iOS-style error message.
    """
    html = f"""
    <div class="ios-alert ios-alert-error">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_warning(message: str):
    """
    Renders an enhanced iOS-style warning message.
    """
    html = f"""
    <div class="ios-alert ios-alert-warning">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_success_message(num_sets: int):
    """
    Renders an enhanced iOS-style success message.
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
    Sets proper viewport for mobile devices with enhanced iOS-specific meta tags.
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
        
        // Add apple touch icon
        var touchIconMeta = document.createElement('link');
        touchIconMeta.rel = 'apple-touch-icon';
        touchIconMeta.href = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAk3SURBVHgB7d1dbFxXFQfw/7p2J3bsOJuUQpoNhU/TJA05a4JFqGpLk44Ui0CjUlWLOvMSTvuEgPZtGlpa2IsoL2yceVClPkBbwQMgFdQKFgK1SRALkk09QDDgVpuSUJE6/gg0XrOvE3vsuZ9zr+978zvSKJk7vrZnZ/9sbp179hmCEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCiM7DEEKUwQ84hOZMNnuQHyDyXOZ5kXXgzTsZzfqxYeP0LUN75n1sGgvw0vFe/gZ0MOlRIWqYzxTiJtj1FiKWhyGCaFPw3EwE2GgGEMXKXXyDBZODtJ5e2zW3QQgIHoU5N2tDqGR0FqXjyVnXrElcJLe1eSC/5GvDC+m0QCfhSP6uuYc9Jzh1mTr8fB8mhNrVJg4Gs8HhbMsQJcw4GRo4eZnsMh85Y+UHHEIMtqsbY2dY2WNEzh2GC/zLXGpgp+N9WRsHb/CCQU5iD5gP+N5qBpfPLwfDmTgYR5tEG9NjJwe4JhFHbTLOzCXXODl2MsP/nMdKFJeS+AUvOkOzLHUeNY7POu+Krc75AZfhBtNQnB9s4xfcgtYbxW3HKf7lzGJDxnwwncvt3HGJ/52uTf6fmU/gCL//GmYyYx4cJyNhPGALJHbMoMOGbLKLRF2vvYW75OKmtTW7l/+9lJ5HNczGfJ9/+fVgwlgrg8llABpynf7eZF8qhSSo+wfmvfb2zd/iP35PL2fJXuJjgkq9Sw+eY4P+7a4buNnPj5YuVDyR+XWH83l39sPU2MN5kx2GorqPQxxiC7SIL2MZTLGKxzH10wBHY3t758ql8FXXH8nt2ZHJv+lwfTjUyuHK7txMKkfZ0O2t7IJGNG7kcM14hf/a32lUDhUjrxkPcyMfjTlK8CNv2vGf/9HFk7V+73wuZZZXU3F3kRn/DhJKrdAjmePcjx5A65yLu7zP3cjq4j8Pptc//aSLCw0Xf7n9NXcxbefw+OjEPWhTakdoFfb+hCKfPLffzeB9NtYvDm9ZyLt4b7PfN5PJPP+ZrCr5gfxARqegbSdSKKT+hXbAGd7lL06k13Hqa7S65/uZX/EiO8L/zXYf+OOVYw8+NjF99yfn7n7v7mX3DxcnuONkGZpJ3EJz58e7p0eDl9H+RnEFsD3rdTq3RxYbQjRLXGiOyJ9FQsRNOaJrp6ARK+VwhZuBpjS6l+NIE+9PRTiSTnEsJEzchXJH0n7v3N4Zno+GY4L/OXnqwMIGX3PZdjdHOGdH/sTXOvJ8Iop38d1FnEfyJW4N3SgavIR5fHRy8ey+BZMv/nHx7vvXLkRnclRPKNeOGR4zHIcP5lYgOCZ08nq6fCO5z0UYg85OlE052pwxzPHZfGzvHC/YA7WFrr1aKKy9z79P9CIQkfQXOlokHznVBRGa44IH3x3LWw4d6y88c17F1n7fGmrcQvPgwJmRW66P3NLVjvvfT3VNYnE8NrD03JnTB5Y/+dQdXl5P7YJEUxV6uh7CkeBrSw9/8Nn3vP29qVvf1zVvXXrt9KF9y9mR20yu05vNfnFvKOL4jYu7/qdXV1dDlbfzd0WC2DLvXUHbKU07Oo4XK/VVSDQ1hS5ujFCtMvTIQn3lwbTLM+X8NepZK/VE1Mv0DByBxFNR6PjjctFZzXOnD4fXOoQtOr1O9hANFhJCRaGjo3PU9QJ1X97e9+z5OxaPbjnPc/kEpYX2I18c2Ty7mfuQQGplQqNSO7hGfgMC6bSGLgnNPa7V7IjsQxM3SSeQkjV0tZqNTi0iISOzTjrV0GVpaVjNBcKK/qGaHhpLR6FLnqNtVevkEw2S0kInjYpCu7J0Lo2OzOZGXYWub9PppZRTT+hS0T6UmkJvcf5+u+FcmU4MfgKnmdCJWgbXINWFrq3Xzn2wPSGJu3RNQiehyUbNQz/v+eyvITvb2zedHZmbhF9vd9HKFHXAUjdDJ/FGm9RXOey3npnfvvXi7t2/7d965ZXtWy9exRZOHEgSP+mgaZJFaP45Cq2Ln/o0+4vZcLvHdRBW+0lrE6Gp+2ZHqoGcuAVlh2WJCMqlHEp6KU38UZrUF7qLxpZTT+gvp36Z3LxUFZo6JDKfJcWtoxuiqtA0sFP0xwNQRVWhSxf27fgZwiapmkNrIvPeGHK3OQrJbCiRqpSjPWMaY0pTEV1KVaEDgf5ykTlIdaE7veeqXlOWcrhaNmCdRVmhuziyqHpNbSdcwlJOZaE7+YMDdFKZcrixSNRN9Cc6laUc7mXR5fWNj37n4uqBvYd/91b/1H/g7VZV6E7ejqfqI1t+9e9jd/3+N3fL2+yAVAJooOwjW8xkcgwSQV2ha6T9I1ssX/MKRSw1he7kkTnqtR8hgbQUutHIdPCJThdnxj8HiaTj1rbwzGyNHeWBUdYGC6rl5g4XF6d+d3Dq2YnDH31E8/X7R2fy3W5DQC+fCgvMdeJYAqlIOaJRDG8f12pWNFCRljn+H04fymE6m+n6wnbyvLlWSgodfzSQ1bE/G6q0FPq5x59Ql3b0qx1Fa+kZbNBTZ++KP1LFbA6bPe/bkDBqC/3qc492zPZ5/u3nP9T0nzMHFiRyYTtMbaGjUTqgQnNj34SEUVvo6C/KvWuORCn2S0kZmQOqC00xk7Q9bW0w4WSaYWgqTDReOuOFAafp2FUoQPFy2hZ+AInVEYUuK3b86ePFdcpU2KGJb0FidUyhaxU9Q9Pn/Mw2QWK1kh6h1ZS9dLFBjkBiddQautGx5+IpnZkJ2G+AIpBoBtqEzg+XqyGGlLpcapR1Ua/7PXwLFKHFizJZC+i4VKS00PWEP2RP12K60HScrJpx3hQ6Toc56RiKoSbnaBeJ28K3QSeoSjlazRQ+Dy0KwxQWGsUphZt7I6bwbmgGXVMOK/rDazkLGu+fqrPQXCn/T6ENY2MWHQUMTTnU9Lb4PddCE+pHaAD1I3S3U7XLRIuTtbM+qR+huxU3+BZoRtMI7crrIdkI3b7ub7wQ2tE0Qvs7Z+/13ONlDYzEw4OzaYQ+/viT/IGbQovF66OhebQX2vr9/wEK9Fvn5iR/HQAAAABJRU5ErkJggg==';
        document.getElementsByTagName('head')[0].appendChild(touchIconMeta);
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
    Main application entry point with enhanced SET-themed layout optimized for iPhone.
    """
    # 1. Load custom SET-themed CSS
    load_custom_css()
    
    # 2. Set proper viewport for iOS
    is_mobile = detect_mobile_device()
    st.session_state.is_mobile = is_mobile
    
    # 3. Handle app reset if needed
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # 4. Display SET-themed header
    render_header()
    
    # 5. APP FLOW - Single-screen approach for iPhone
    
    # UPLOAD SCREEN
    if not st.session_state.get("uploaded_file"):
        # Center align uploader with custom styling
        with st.container():
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 1rem;">
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; text-align: center; 
                            background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); 
                            -webkit-background-clip: text; background-clip: text; 
                            -webkit-text-fill-color: transparent;">
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
                    img_pil = optimize_image_size(img_pil, max_dim=700)  # Reduced max size for better mobile fit
                    st.session_state.original_image = img_pil
                    st.session_state.image_height = img_pil.height
                    st.session_state.app_view = "preview"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load the image")
                    st.error(str(e))
    
    # PREVIEW SCREEN - Show original with Find Sets button
    elif st.session_state.app_view == "preview":
        # Container for the image and loading overlay
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Only show Find Sets button (enhanced style)
        st.markdown('<div class="ios-button-primary find-sets-button">', unsafe_allow_html=True)
        if st.button("Find Sets", key="find_sets_btn", use_container_width=True):
            st.session_state.app_view = "processing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PROCESSING SCREEN
    elif st.session_state.app_view == "processing":
        # Container for the image and loading overlay
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        # Overlay the enhanced loader on the image
        render_loading()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
            
            # Add retry button with enhanced style
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
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Try again button with enhanced style
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Image", key="try_again_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif st.session_state.no_sets_found:
            render_warning("No valid SETs found in this game")
            
            # Show processed image
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Try again button with enhanced style
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Image", key="no_sets_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Success case - show results with enhanced styling
            num_sets = len(st.session_state.sets_info)
            
            # Results header with enhanced badge
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 0.4rem;">
                <div class="ios-badge">{num_sets} SET{'' if num_sets == 1 else 's'} Found</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display processed image with sets highlighted
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Single action button with enhanced style
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Analyze Another Game", key="new_img_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
