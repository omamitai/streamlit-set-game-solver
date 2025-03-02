"""
Set Game Detector App
=====================

A Streamlit application that identifies valid SETs from an uploaded image of a SET game board.
It uses computer vision (YOLO) for card detection, TensorFlow models for feature classification,
and highlights detected sets on the board image.
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
import base64
import random
import time

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SET Game Detector",
    layout="wide"
)

# =============================================================================
# SET THEME COLORS
# =============================================================================
SET_THEME = {
    "primary": "#7C3AED",     # Purple
    "secondary": "#10B981",   # Green
    "accent": "#EC4899",      # Pink
    "red": "#EF4444",         # Red
    "green": "#10B981",       # Green
    "purple": "#8B5CF6",      # Light purple
    "background": "#F9F9FC",  # Light background
    "card": "#FFFFFF",        # Card background
    "text": "#222222",        # Text color
    "text_muted": "#666666",  # Muted text
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# We place this early to ensure these states are available before subsequent operations
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
    st.session_state.is_mobile = False
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

# =============================================================================
# CUSTOM CSS
# =============================================================================
def load_custom_css():
    """
    Loads custom CSS for a cohesive SET-themed UI with iOS-like aesthetics.
    Includes responsive design adjustments for mobile devices.
    """
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Text:wght@400;500;600&display=swap');

    :root {{
        --set-primary: {SET_THEME["primary"]};
        --set-secondary: {SET_THEME["secondary"]};
        --set-accent: {SET_THEME["accent"]};
        --set-red: {SET_THEME["red"]};
        --set-green: {SET_THEME["green"]};
        --set-purple: {SET_THEME["purple"]};
        --set-background: {SET_THEME["background"]};
        --set-card: {SET_THEME["card"]};
        --set-text: {SET_THEME["text"]};
        --set-text-muted: {SET_THEME["text_muted"]};
    }}

    body {{
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--set-background);
        color: var(--set-text);
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
    }}

    /* Custom Streamlit override */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}

    /* Minimal Header for Mobile */
    .set-header-minimal {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .set-header-minimal h1 {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-primary) 50%, var(--set-accent) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    /* Mobile Optimized View */
    .mobile-view-container {
        padding: 0.5rem;
    }
    
    .mobile-compact-btn {
        padding: 0.75rem !important;
        margin-top: 0.5rem !important;
        border-radius: 10px !important;
    }
    
    .toggle-original-btn {
        background: transparent !important;
        color: var(--set-primary) !important;
        box-shadow: none !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        margin-top: 0.5rem !important;
    }
    
    .toggle-original-btn:hover {
        background: rgba(124, 58, 237, 0.05) !important;
    }
    
    .mobile-results-container {
        margin-top: 0.5rem;
    }

    /* Cards */
    .set-card {{
        background-color: var(--set-card);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 100%;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.8);
    }}
    
    .set-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.15);
    }}
    
    .set-card h3 {{
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        margin-top: 0;
        font-weight: 600;
        font-size: 1.4rem;
        color: var(--set-primary);
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(124, 58, 237, 0.2);
        text-align: center;
        letter-spacing: -0.01em;
    }}

    /* Upload Area */
    .upload-area {{
        border: 2px dashed rgba(124, 58, 237, 0.3);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        background-color: rgba(124, 58, 237, 0.03);
        cursor: pointer;
        margin-bottom: 1.5rem;
    }}
    
    .upload-area:hover {{
        border-color: var(--set-primary);
        background-color: rgba(124, 58, 237, 0.06);
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, var(--set-primary) 0%, var(--set-accent) 100%);
        color: white;
        border: none;
        padding: 0.9rem 1.5rem;
        border-radius: 12px;
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        margin-top: 1rem;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.25);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4);
    }}
    
    .stButton>button:active {{
        transform: translateY(1px);
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3);
    }}

    /* Loader */
    .loader-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
        margin: 1rem 0;
    }}
    
    .loader {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .loader-dot {{
        width: 12px;
        height: 12px;
        margin: 0 6px;
        border-radius: 50%;
        display: inline-block;
        animation: loader 1.8s infinite cubic-bezier(0.45, 0.05, 0.55, 0.95) both;
    }}
    
    .loader-dot-1 {{
        background-color: var(--set-red);
        animation-delay: -0.32s;
    }}
    
    .loader-dot-2 {{
        background-color: var(--set-green);
        animation-delay: -0.16s;
    }}
    
    .loader-dot-3 {{
        background-color: var(--set-purple);
        animation-delay: 0s;
    }}
    
    @keyframes loader {{
        0%, 80%, 100% {{ transform: scale(0); opacity: 0.7; }}
        40% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Image Container */
    .image-container {{
        margin: 1.5rem 0;
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.6);
    }}
    
    .image-container:hover {{
        transform: scale(1.01);
        box-shadow: 0 15px 40px rgba(124, 58, 237, 0.18);
    }}
    
    .image-container img {{
        display: block;
        width: 100%;
        height: auto;
    }}
    
    /* Caption styling for image containers */
    .image-container .caption {{
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-top: 1px solid rgba(255, 255, 255, 0.5);
        font-weight: 500;
    }}

    /* System Message */
    .system-message {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }}
    
    .system-message p {{
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--set-text-muted);
        margin: 0;
    }}

    /* Error Message */
    .error-message {{
        background-color: rgba(239, 68, 68, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }}
    
    .error-message::before {{
        content: "⚠️";
        font-size: 1.4rem;
        margin-right: 0.8rem;
    }}
    
    .error-message p {{
        margin: 0;
        font-weight: 500;
        color: var(--set-red);
    }}

    /* Warning Message */
    .warning-message {{
        background-color: rgba(245, 158, 11, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }}
    
    .warning-message::before {{
        content: "ℹ️";
        font-size: 1.4rem;
        margin-right: 0.8rem;
    }}
    
    .warning-message p {{
        margin: 0;
        font-weight: 500;
        color: #F59E0B;
    }}
    
    /* Success Message */
    .success-message {{
        background-color: rgba(16, 185, 129, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }}
    
    .success-message::before {{
        content: "✅";
        font-size: 1.4rem;
        margin-right: 0.8rem;
    }}
    
    .success-message p {{
        margin: 0;
        font-weight: 500;
        color: var(--set-green);
    }}
    
    /* SET Cards Container */
    .set-results-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    /* Image Wrapper */
    .img-wrapper {{
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        transition: all 0.3s ease;
    }}

    /* Separator */
    .set-separator {{
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.5), transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }}

    /* Update responsive adjustments */
    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        /* Minimize padding in Streamlit containers */
        .block-container {{
            padding-top: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }}
        
        .stButton>button {{
            padding: 0.7rem !important;
            font-size: 0.9rem !important;
        }}
        
        /* Reduce margins */
        .image-container {{
            margin: 0.5rem 0 !important;
        }}
        
        /* Make messages more compact */
        .system-message,
        .error-message,
        .warning-message,
        .success-message {{
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }}
        
        /* Reduce image margins */
        [data-testid="stImage"] {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}
        
        /* Reduce caption size */
        .caption {{
            font-size: 0.8rem !important;
            padding: 0.5rem !important;
        }}
    }}
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
    Each SET is drawn in a different color and offset (thickness & expansion) 
    so that overlapping sets are visible.
    """
    # Some distinct BGR colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    base_thickness = 8
    base_expansion = 5

    for idx, single_set in enumerate(sets_detected):
        color = colors[idx % len(colors)]
        thickness = base_thickness + 2 * idx
        expansion = base_expansion + 15 * idx

        for i, card_info in enumerate(single_set["cards"]):
            x1, y1, x2, y2 = card_info["Coordinates"]
            # Expand the bounding box slightly
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(board_img.shape[1], x2 + expansion)
            y2e = min(board_img.shape[0], y2 + expansion)

            cv2.rectangle(board_img, (x1e, y1e), (x2e, y2e), color, thickness)

            # Label only the first card's box with "Set <number>"
            if i == 0:
                cv2.putText(
                    board_img,
                    f"Set {idx + 1}",
                    (x1e, y1e - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    thickness
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

def optimize_image_size(img_pil: Image.Image, max_dim=900) -> Image.Image:
    """
    Resizes a PIL image if its largest dimension exceeds max_dim, to reduce processing time.
    Optimized for mobile viewing.
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
def render_loading():
    """
    Shows a simple 3-dot animated loader styled with SET colors.
    """
    loader_html = """
    <div class="loader-container">
        <div class="loader">
            <div class="loader-dot loader-dot-1"></div>
            <div class="loader-dot loader-dot-2"></div>
            <div class="loader-dot loader-dot-3"></div>
        </div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

# Arrow function removed as requested

def render_error(message: str):
    """
    Renders a styled error message in a box with a left border.
    """
    html = f"""
    <div class="error-message">
        <p>{message}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_warning(message: str):
    """
    Renders a styled warning message in a box with a left border.
    """
    html = f"""
    <div class="warning-message">
        <p>{message}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_process_prompt():
    """
    Renders a styled "system message" to prompt the user to click 'Find Sets'.
    """
    html = """
    <div class="system-message">
        <p>Tap "Find Sets" to analyze</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
def render_success_message(num_sets):
    """
    Renders a styled success message indicating the number of sets found.
    """
    if num_sets == 0:
        return
        
    html = f"""
    <div class="success-message">
        <p>Found {num_sets} SET{'' if num_sets == 1 else 's'}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
def render_separator():
    """
    Renders a visual separator between sections.
    """
    html = """
    <div class="set-separator"></div>
    """
    st.markdown(html, unsafe_allow_html=True)

def reset_app_state():
    """
    Clears and reinitializes session state, forcing the UI to reset.
    """
    preserve_keys = ["is_mobile"]
    to_remove = [k for k in st.session_state.keys() if k not in preserve_keys]

    for k in to_remove:
        del st.session_state[k]

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
    st.session_state.reset_timestamp = time.time()

    # Force clearing of caches if desired
    st.cache_data.clear()

    # If you want to clear the query params (if any)
    st.query_params.clear()

def detect_mobile_device():
    """
    Simple placeholder for mobile detection.
    We use a JavaScript snippet that sets a flag in sessionStorage,
    but here we simply return the session state's known value if set.
    This is improved for better iPhone detection.
    """
    js_snippet = """
    <script>
        function isMobile() {
            return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || window.innerWidth <= 768;
        }
        
        if (isMobile()) {
            sessionStorage.setItem('isMobile', 'true');
            document.body.classList.add('mobile-view');
            // Add a meta tag to ensure proper scaling on mobile
            if (!document.querySelector('meta[name="viewport"]')) {
                var meta = document.createElement('meta');
                meta.name = 'viewport';
                meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
                document.getElementsByTagName('head')[0].appendChild(meta);
            }
        } else {
            sessionStorage.setItem('isMobile', 'false');
        }
    </script>
    """
    st.markdown(js_snippet, unsafe_allow_html=True)
    
    # Force mobile view on for smaller screens
    if "is_mobile" not in st.session_state:
        st.session_state.is_mobile = True  # Default to mobile for safer experience
    
    return st.session_state.is_mobile

def render_header():
    """
    Renders a minimalistic header for the app.
    """
    header_html = """
    <div class="set-header-minimal">
        <h1>SET Detector</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # 1. Load our custom CSS
    load_custom_css()

    # 2. Detect mobile or desktop
    is_mobile = detect_mobile_device()
    st.session_state.is_mobile = is_mobile

    # 3. If a reset was triggered, rerun to clear stale states
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()

    # 4. Render header
    render_header()

    # 5. Optimized layout for iPhone/mobile screens
    # Auto-detect but default to mobile view for safer experience
    is_mobile = detect_mobile_device()
    st.session_state.is_mobile = is_mobile
    
    # 3. If a reset was triggered, rerun to clear stale states
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # Mobile-optimized layout
    # Clear header taking minimal space
    render_header()
    
    # UPLOAD STEP - Only show if no file is uploaded
    if not st.session_state.get("uploaded_file"):
        upload_container = st.container()
        with upload_container:
            st.markdown('<div class="mobile-view-container">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload a SET board image",
                type=["png", "jpg", "jpeg"],
                key=f"uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed",
                help="Take a photo or choose an image of your SET game"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file and uploaded_file != st.session_state.get("uploaded_file", None):
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
                    img_pil = optimize_image_size(img_pil, max_dim=900) # Smaller for mobile
                    st.session_state.original_image = img_pil
                    st.session_state.image_height = img_pil.height
                except Exception as e:
                    st.error(f"Failed to load the image: {str(e)}")
    
    # If a file is uploaded, handle processing flow
    if st.session_state.get("uploaded_file"):
        st.markdown('<div class="mobile-view-container">', unsafe_allow_html=True)
        
        # ONLY show original image initially or if toggled
        if not st.session_state.get("processed") or st.session_state.get("show_original", False):
            with st.container():
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(
                    st.session_state.original_image,
                    caption="Original Image",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # PROCESSING SECTION
        # Process button (if not already processing or processed)
        if not st.session_state.get("processed") and not st.session_state.get("start_processing"):
            if st.button("Find Sets", key="find_sets_btn", use_container_width=True, 
                      help="Process the uploaded image to find SET cards"):
                st.session_state.processed = False
                st.session_state.processed_image = None
                st.session_state.sets_info = None
                st.session_state.no_cards_detected = False
                st.session_state.no_sets_found = False
                st.session_state.show_original = False
                st.session_state.start_processing = True
                st.rerun()
        
        # Processing state - show loader
        if st.session_state.get("start_processing"):
            render_loading()
            try:
                img_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
                sets_info, processed_img = identify_sets_from_image(
                    img_cv, detector_card, detector_shape, model_fill, model_shape
                )
                st.session_state.sets_info = sets_info
                st.session_state.processed_image = processed_img
                st.session_state.processed = True
                st.session_state.start_processing = False
                st.rerun()
            except Exception as e:
                st.error("An error occurred during processing:")
                st.code(traceback.format_exc())
                st.session_state.start_processing = False
        
        # RESULTS SECTION
        elif st.session_state.get("processed"):
            with st.container():
                st.markdown('<div class="mobile-results-container">', unsafe_allow_html=True)
                
                # Case 1: No cards found
                if st.session_state.no_cards_detected:
                    render_error("No cards detected")
                
                # Case 2: Cards found but no sets
                elif st.session_state.no_sets_found:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    pm = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                    st.image(pm, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    render_warning("No valid SETs found")
                
                # Case 3: Sets found!
                else:
                    render_success_message(len(st.session_state.sets_info))
                    
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    pm = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                    st.image(
                        pm,
                        caption=f"{len(st.session_state.sets_info)} SET{'' if len(st.session_state.sets_info) == 1 else 's'} found",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # CONTROLS - Two buttons side by side for toggle and reset
                col1, col2 = st.columns(2)
                with col1:
                    toggle_label = "Hide Original" if st.session_state.get("show_original", False) else "Show Original"
                    if st.button(toggle_label, key="toggle_btn", use_container_width=True):
                        st.session_state.show_original = not st.session_state.get("show_original", False)
                        st.rerun()
                
                with col2:
                    if st.button("New Image", key="reset_btn", use_container_width=True):
                        reset_app_state()
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def run_app():
    """
    Wrapper to run the main function if name == '_main_'.
    """
    main()

if __name__ == "__main__":
    run_app()
