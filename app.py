"""
SET Game Detector - Production Ready Streamlit App
=================================================

An advanced computer vision application that detects valid sets in the SET card game.
Built with Streamlit, OpenCV, TensorFlow, PyTorch, and YOLOv8.

Features:
- Responsive design works on both desktop and mobile devices
- Automated card detection and set identification
- High-quality visual interface with SET-themed styling
- Desktop: Side-by-side original and processed images with manual processing
- Mobile: Auto-processing with optimized layout for touch interactions
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
import time
import random
import base64
import re
from typing import Tuple, List, Dict, Union, Optional

# =============================================================================
# CONFIGURATION & PAGE SETUP
# =============================================================================
st.set_page_config(
    layout="wide", 
    page_title="SET Game Detector",
    page_icon="🎴",
    initial_sidebar_state="auto"
)

# Define SET theme colors
SET_COLORS = {
    "primary": "#7C3AED",    # Purple
    "secondary": "#10B981",  # Green
    "accent": "#EC4899",     # Pink
    "red": "#EF4444",        # Red for cards
    "green": "#10B981",      # Green for cards
    "purple": "#8B5CF6",     # Purple for cards
    "background": "#F9FAFB", # Light background
    "card": "#FFFFFF",       # Card background
    "text": "#1F2937",       # Dark text
    "lightText": "#6B7280"   # Secondary text
}

# =============================================================================
# INITIALIZATION OF SESSION STATE
# =============================================================================
def init_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        "processed": False,              # Flag for whether image processing is done
        "start_processing": False,       # Flag to trigger processing
        "uploaded_file": None,           # The file object from the uploader
        "original_image": None,          # Original PIL image
        "processed_image": None,         # Processed image with sets highlighted
        "sets_info": None,               # Information about detected sets
        "is_mobile": False,              # Whether we're in mobile layout mode
        "should_reset": False,           # Flag to trigger a complete reset
        "no_cards_detected": False,      # Flag for when no cards are found
        "no_sets_found": False,          # Flag for when no sets are found
        "uploader_key": "initial",       # Key to force file uploader resets
        "models_loaded": False,          # Flag for model loading status
        "error_message": None,           # Store any error messages
        "detection_time": None,          # Time taken for processing
    }
    
    # Only set values that don't already exist
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# =============================================================================
# CSS STYLING
# =============================================================================
def load_css():
    """Load custom CSS with responsive design for both mobile and desktop"""
    css = f"""
    <style>
    /* Global styles with SET theme */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    :root {{
        --primary: {SET_COLORS["primary"]};
        --secondary: {SET_COLORS["secondary"]};
        --accent: {SET_COLORS["accent"]};
        --red: {SET_COLORS["red"]};
        --green: {SET_COLORS["green"]};
        --purple: {SET_COLORS["purple"]};
        --background: {SET_COLORS["background"]};
        --card: {SET_COLORS["card"]};
        --text: {SET_COLORS["text"]};
        --light-text: {SET_COLORS["lightText"]};
    }}
    
    body {{
        font-family: 'Poppins', sans-serif;
        background-color: var(--background);
        color: var(--text);
    }}
    
    /* Main container styles - fix Streamlit's default margins */
    .main .block-container {{
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    /* SET Header */
    .set-header {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        position: relative;
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    
    .set-header h1 {{
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, var(--purple) 0%, var(--primary) 50%, var(--accent) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    .set-header p {{
        font-size: 1.1rem;
        opacity: 0.9;
        color: var(--light-text);
        max-width: 600px;
        text-align: center;
        margin: 0;
    }}
    
    /* Card styles */
    .set-card {{
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }}
    
    .set-card h3 {{
        margin-top: 0;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
        text-align: center;
        color: var(--primary);
        font-weight: 600;
    }}
    
    /* Upload area */
    .upload-area {{
        border: 2px dashed rgba(124, 58, 237, 0.5);
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        background-color: rgba(124, 58, 237, 0.05);
        cursor: pointer;
        margin-bottom: 1.2rem;
    }}
    
    .upload-area:hover {{
        border-color: var(--primary);
        background-color: rgba(124, 58, 237, 0.1);
        transform: translateY(-2px);
    }}
    
    /* Button styling */
    .stButton>button {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px rgba(124, 58, 237, 0.25);
    }}
    
    .stButton>button:hover {{
        opacity: 0.95;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(124, 58, 237, 0.3);
    }}
    
    .stButton>button:active {{
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.2);
    }}
    
    /* Loading animation */
    .loader-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 250px;
        flex-direction: column;
    }}
    
    .loader {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }}
    
    .loader-dot {{
        width: 12px;
        height: 12px;
        margin: 0 8px;
        border-radius: 50%;
        display: inline-block;
        animation: loader 1.5s infinite ease-in-out both;
    }}
    
    .loader-text {{
        font-size: 1.1rem;
        color: var(--light-text);
        margin-top: 1rem;
        text-align: center;
    }}
    
    .loader-dot-1 {{
        background-color: var(--red);
        animation-delay: -0.3s;
    }}
    
    .loader-dot-2 {{
        background-color: var(--green);
        animation-delay: -0.15s;
    }}
    
    .loader-dot-3 {{
        background-color: var(--purple);
        animation-delay: 0s;
    }}
    
    @keyframes loader {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    
    /* Image container */
    .image-container {{
        margin: 1rem 0;
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        background: white;
        border: 1px solid rgba(0, 0, 0, 0.05);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }}
    
    /* Processing placeholder */
    .system-message {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    
    .system-message p {{
        font-size: 1.2rem;
        font-weight: 500;
        color: var(--primary);
        margin: 0;
    }}
    
    /* Messages styling */
    .message {{
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    
    .message-icon {{
        margin-right: 0.8rem;
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .message-content {{
        flex: 1;
    }}
    
    .message-content p {{
        margin: 0;
        font-weight: 500;
    }}
    
    .error-message {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--red);
    }}
    
    .error-message .message-icon,
    .error-message p {{
        color: var(--red);
    }}
    
    .warning-message {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
    }}
    
    .warning-message .message-icon,
    .warning-message p {{
        color: #F59E0B;
    }}
    
    .success-message {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--green);
    }}
    
    .success-message .message-icon,
    .success-message p {{
        color: var(--green);
    }}
    
    /* Mobile uploader styling */
    .mobile-uploader {{
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0 1.8rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }}
    
    .mobile-uploader h3 {{
        color: var(--primary);
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
    }}
    
    /* Remove file uploader label if we want to hide it */
    .hide-label p {{
        display: none !important;
    }}
    
    /* Sidebar styling */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
        border-right: 1px solid rgba(124, 58, 237, 0.1);
    }}
    
    [data-testid="stSidebarNav"] {{
        background-image: linear-gradient(180deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    }}
    
    /* Stats box */
    .stats-box {{
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }}
    
    .stats-item {{
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }}
    
    .stats-item:last-child {{
        border-bottom: none;
    }}
    
    .stats-label {{
        font-weight: 500;
        color: var(--light-text);
    }}
    
    .stats-value {{
        font-weight: 600;
        color: var(--primary);
    }}
    
    /* Instruction Text */
    .instruction-text {{
        margin: 1rem 0;
        padding: 1rem;
        background-color: rgba(124, 58, 237, 0.05);
        border-radius: 8px;
        color: var(--light-text);
    }}
    
    /* Mobile-specific styles */
    @media (max-width: 991px) {{
        /* Completely hide sidebar on mobile */
        section[data-testid="stSidebar"] {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: hidden !important;
            z-index: -1 !important;
        }}
        
        /* Override Streamlit's default container width */
        .main .block-container {{
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }}
        
        /* Header adjustments */
        .set-header h1 {{ 
            font-size: 2rem; 
        }}
        
        .set-header p {{ 
            font-size: 0.9rem; 
        }}
        
        .set-header {{
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        
        /* Button adjustments for touch */
        .stButton>button {{
            padding: 1rem;
            font-size: 1rem;
            height: auto;
            min-height: 3.5rem;
        }}
        
        /* Ensure images are fully responsive */
        .image-container img {{
            width: 100% !important;
            height: auto !important;
            max-width: 100% !important;
        }}
        
        /* Tighter message spacing */
        .system-message, .message {{
            margin: 1rem 0;
            padding: 1rem;
        }}
        
        /* Better loader sizing */
        .loader-container {{
            height: 180px;
        }}
        
        .loader-dot {{
            width: 10px;
            height: 10px;
        }}
        
        /* Adjust font sizes */
        p, .stats-label, .stats-value, .instruction-text {{
            font-size: 0.95rem;
        }}
        
        /* Adjust spacing */
        .image-container {{
            margin: 0.8rem 0;
        }}
        
        /* Make file uploader more touch-friendly */
        .st-emotion-cache-1erivf3, .st-emotion-cache-1gulkj5 {{
            min-height: 100px !important;
        }}
    }}
    
    /* Default Streamlit element adjustments */
    .stRadio [role="radiogroup"] {{
        padding: 0 !important;
    }}
    
    /* Fix file uploader */
    .uploadFile p {{
        display: none !important;
    }}
    
    /* Fix streamlit footer and branding */
    footer {{
        visibility: hidden;
    }}
    header {{
        visibility: hidden;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================
base_dir = Path("models")
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Load the TensorFlow models for shape and fill classification with error handling"""
    try:
        shape_model = load_model(str(characteristics_path / "shape_model.keras"))
        fill_model = load_model(str(characteristics_path / "fill_model.keras"))
        return shape_model, fill_model
    except Exception as e:
        st.error(f"Error loading classification models: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    """Load the YOLOv8 models for card and shape detection with error handling"""
    try:
        shape_detection_model = YOLO(str(shape_path / "best.pt"))
        shape_detection_model.conf = 0.5
        card_detection_model = YOLO(str(card_path / "best.pt"))
        card_detection_model.conf = 0.5
        if torch.cuda.is_available():
            card_detection_model.to("cuda")
            shape_detection_model.to("cuda")
            # Log info about CUDA if available
            st.session_state['gpu_info'] = f"Using GPU: {torch.cuda.get_device_name(0)}"
        return card_detection_model, shape_detection_model
    except Exception as e:
        st.error(f"Error loading detection models: {str(e)}")
        return None, None

# =============================================================================
# UTILITY & PROCESSING FUNCTIONS
# =============================================================================
def check_and_rotate_input_image(board_image: np.ndarray, detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Check if image needs rotation based on card orientation and rotate if needed.
    Returns the adjusted image and a flag indicating whether rotation was applied.
    """
    try:
        card_results = detector(board_image)
        card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
        if card_boxes.size == 0:
            return board_image, False
        
        # Compare width vs height of detected cards
        widths = card_boxes[:, 2] - card_boxes[:, 0]
        heights = card_boxes[:, 3] - card_boxes[:, 1]
        
        # If mean height > mean width, rotate the image
        if np.mean(heights) > np.mean(widths):
            return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
        return board_image, False
    except Exception as e:
        # If any error occurs, return the original image
        st.warning(f"Orientation check failed: {str(e)}")
        return board_image, False

def restore_original_orientation(image: np.ndarray, was_rotated: bool) -> np.ndarray:
    """Restore the original orientation of the image if it was rotated"""
    if was_rotated:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def predict_color(shape_image: np.ndarray) -> str:
    """
    Predict the color of a shape using HSV color filtering.
    Returns 'red', 'green', or 'purple'.
    """
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    
    # Define color range masks in HSV
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
    purple_mask = cv2.inRange(hsv_image, np.array([120, 50, 50]), np.array([160, 255, 255]))
    
    # Red wraps around in HSV, so check in two ranges
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Count non-zero pixels for each color
    color_counts = {
        "green": cv2.countNonZero(green_mask),
        "purple": cv2.countNonZero(purple_mask),
        "red": cv2.countNonZero(red_mask)
    }
    
    # Return the color with most matching pixels
    return max(color_counts, key=color_counts.get)

def predict_card_features(
    card_image: np.ndarray, 
    shape_detector: YOLO,
    fill_model: tf.keras.Model, 
    shape_model: tf.keras.Model,
    box: List[int]
) -> Dict:
    """
    Extract all features from a single card image:
    - Count (number of shapes)
    - Color (red, green, purple)
    - Fill (empty, full, striped)
    - Shape (diamond, oval, squiggle)
    """
    # Detect shapes within the card
    shape_results = shape_detector(card_image)
    card_h, card_w = card_image.shape[:2]
    card_area = card_w * card_h
    
    # Filter out small detections that might be noise
    filtered_boxes = [
        [int(x1), int(y1), int(x2), int(y2)]
        for x1, y1, x2, y2 in shape_results[0].boxes.xyxy.cpu().numpy()
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area  # Ensure shape is at least 3% of card area
    ]
    
    # If no shapes detected, return empty features
    if not filtered_boxes:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': box}
    
    # Prepare inputs for the classification models
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    fill_imgs, shape_imgs, color_list = [], [], []
    
    # Process each detected shape
    for fb in filtered_boxes:
        x1, y1, x2, y2 = fb
        shape_img = card_image[y1:y2, x1:x2]
        
        # Skip invalid shapes (e.g., out of bounds)
        if shape_img.size == 0 or min(shape_img.shape) < 5:
            continue
            
        # Resize for the fill model
        fill_img = cv2.resize(shape_img, tuple(fill_input_shape)) / 255.0
        
        # Resize for the shape model
        shape_img_resized = cv2.resize(shape_img, tuple(shape_input_shape)) / 255.0
        
        # Collect preprocessed images and colors
        fill_imgs.append(fill_img)
        shape_imgs.append(shape_img_resized)
        color_list.append(predict_color(shape_img))
    
    # If no valid shapes after filtering, return empty features
    if not fill_imgs:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': box}
    
    # Convert to numpy arrays for batch prediction
    fill_imgs = np.array(fill_imgs)
    shape_imgs = np.array(shape_imgs)
    
    # Run model predictions
    fill_preds = fill_model.predict(fill_imgs, batch_size=len(fill_imgs), verbose=0)
    shape_preds = shape_model.predict(shape_imgs, batch_size=len(shape_imgs), verbose=0)
    
    # Get class labels
    fill_labels_list = ['empty', 'full', 'striped']
    shape_labels_list = ['diamond', 'oval', 'squiggle']
    
    # Get predicted class for each shape
    predicted_fill = [fill_labels_list[np.argmax(pred)] for pred in fill_preds]
    predicted_shape = [shape_labels_list[np.argmax(pred)] for pred in shape_preds]
    
    # Use majority vote for each feature
    color_label = max(set(color_list), key=color_list.count)
    fill_label = max(set(predicted_fill), key=predicted_fill.count)
    shape_label = max(set(predicted_shape), key=predicted_shape.count)
    
    # Return complete feature set
    return {
        'count': len(filtered_boxes),
        'color': color_label,
        'fill': fill_label,
        'shape': shape_label,
        'box': box
    }

def is_set(cards: List[dict]) -> bool:
    """
    Check if 3 cards form a valid set according to SET game rules.
    A valid set requires that for each attribute, the cards either 
    all have the same value or all have different values.
    """
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        values = {card[feature] for card in cards}
        # Either all the same (1 unique value) or all different (3 unique values)
        if len(values) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> List[dict]:
    """Find all valid sets among the detected cards"""
    sets_found = []
    
    # Check all combinations of 3 cards
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
    """
    Detect cards in the board image using the YOLO model.
    Returns a list of tuples (card_image, bounding_box).
    """
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    return [
        (board_image[y1:y2, x1:x2], [x1, y1, x2, y2])
        for x1, y1, x2, y2 in card_boxes
    ]

def classify_cards_from_board_image(
    board_image: np.ndarray, 
    card_detector: YOLO,
    shape_detector: YOLO, 
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> pd.DataFrame:
    """
    Process a board image to detect cards and classify their features.
    Returns a DataFrame with all card information.
    """
    # Detect cards
    cards = detect_cards_from_image(board_image, card_detector)
    
    # Extract features for each card
    card_data = []
    for card_image, box in cards:
        # Skip empty or invalid card images
        if card_image.size == 0 or min(card_image.shape) < 10:
            continue
            
        features = predict_card_features(card_image, shape_detector, fill_model, shape_model, box)
        
        # Skip cards where no shapes were detected
        if features['count'] == 0:
            continue
            
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": features['box']
        })
    
    # Convert to DataFrame for easier analysis
    return pd.DataFrame(card_data)

def draw_sets_on_image(board_image: np.ndarray, sets_info: List[dict]) -> np.ndarray:
    """
    Draw rectangles and labels around sets on the image.
    Each set gets a different color and visual treatment.
    """
    # Define colors for different sets (BGR format for OpenCV)
    colors = [
        (255, 50, 50),    # Bright red
        (50, 255, 50),    # Bright green
        (50, 50, 255),    # Bright blue
        (255, 255, 50),   # Cyan
        (255, 50, 255),   # Magenta
        (50, 255, 255),   # Yellow
        (180, 105, 255),  # Pink
        (50, 170, 255),   # Orange
    ]
    
    # Base settings for visual elements
    base_thickness = 8
    base_expansion = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    
    # Create a copy of the image to draw on
    result_image = board_image.copy()
    
    # Draw each set with a unique visual style
    for index, set_info in enumerate(sets_info):
        color = colors[index % len(colors)]
        thickness = base_thickness + 2 * (index % 3)  # Vary thickness slightly
        expansion = base_expansion + 15 * (index % 3)  # Vary box size slightly
        
        # Draw each card in the set
        for i, card in enumerate(set_info["cards"]):
            x1, y1, x2, y2 = card["Coordinates"]
            
            # Expand the box for visibility
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(result_image.shape[1], x2 + expansion)
            y2_exp = min(result_image.shape[0], y2 + expansion)
            
            # Draw the rectangle
            cv2.rectangle(result_image, (x1_exp, y1_exp), (x2_exp, y2_exp), color, thickness)
            
            # Add set label to the first card in each set
            if i == 0:
                # Draw the "Set X" label with better positioning and visibility
                label = f"Set {index + 1}"
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                # Position text above the box with padding
                text_x = x1_exp
                text_y = max(text_size[1] + 10, y1_exp - 10)
                
                # Draw text with a background for better visibility
                cv2.putText(
                    result_image, label, (text_x, text_y),
                    font, font_scale, color, thickness
                )
    
    return result_image

def classify_and_find_sets_from_array(
    board_image: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> tuple:
    """
    Process the board image to find sets. Main processing pipeline.
    Returns (sets_found, processed_image).
    """
    start_time = time.time()
    
    # Reset flags at the beginning
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    
    # Check if rotation is needed
    processed_image, was_rotated = check_and_rotate_input_image(board_image, card_detector)
    
    # Check if cards are detected
    try:
        cards = detect_cards_from_image(processed_image, card_detector)
        
        if not cards:
            st.session_state.no_cards_detected = True
            end_time = time.time()
            st.session_state.detection_time = end_time - start_time
            return [], processed_image
        
        # Classify cards and find sets
        card_df = classify_cards_from_board_image(
            processed_image, card_detector, shape_detector, fill_model, shape_model
        )
        
        # Handle empty dataframe - no valid cards detected
        if card_df.empty:
            st.session_state.no_cards_detected = True
            end_time = time.time()
            st.session_state.detection_time = end_time - start_time
            return [], processed_image
        
        # Find all valid sets
        sets_found = find_sets(card_df)
        
        # Check if sets are found
        if not sets_found:
            st.session_state.no_sets_found = True
            end_time = time.time()
            st.session_state.detection_time = end_time - start_time
            return [], processed_image
        
        # Draw sets on the image
        annotated_image = draw_sets_on_image(processed_image.copy(), sets_found)
        
        # Restore original orientation if needed
        final_image = restore_original_orientation(annotated_image, was_rotated)
        
        end_time = time.time()
        st.session_state.detection_time = end_time - start_time
        
        return sets_found, final_image
        
    except Exception as e:
        st.error(f"Error in card detection: {str(e)}")
        traceback.print_exc()
        end_time = time.time()
        st.session_state.detection_time = end_time - start_time
        return [], board_image

def optimize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize large images to improve performance while maintaining aspect ratio.
    Returns the optimized PIL Image.
    """
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

# =============================================================================
# UI COMPONENT FUNCTIONS
# =============================================================================
def render_loader(message: str = ""):
    """Render a SET-themed loader with custom message"""
    loader_html = f"""
    <div class="loader-container">
        <div class="loader">
            <div class="loader-dot loader-dot-1"></div>
            <div class="loader-dot loader-dot-2"></div>
            <div class="loader-dot loader-dot-3"></div>
        </div>
        {f'<div class="loader-text">{message}</div>' if message else ''}
    </div>
    """
    return st.markdown(loader_html, unsafe_allow_html=True)

def render_message(message: str, type: str = "info"):
    """
    Render a styled message of various types: error, warning, success, info
    Each with appropriate styling and icon
    """
    icon_map = {
        "error": "❌",
        "warning": "⚠️",
        "success": "✅",
        "info": "ℹ️"
    }
    
    icon = icon_map.get(type, "ℹ️")
    
    message_html = f"""
    <div class="{type}-message message">
        <div class="message-icon">{icon}</div>
        <div class="message-content">
            <p>{message}</p>
        </div>
    </div>
    """
    return st.markdown(message_html, unsafe_allow_html=True)

def render_process_message():
    """Render a styled system message to prompt user to process the image"""
    message_html = """
    <div class="system-message">
        <p>👆 Click "Find Sets" to process the image</p>
    </div>
    """
    return st.markdown(message_html, unsafe_allow_html=True)

def render_header():
    """Render the SET-themed header"""
    header_html = """
    <div class="set-header">
        <h1>🎴 SET Game Detector</h1>
        <p>Upload an image of a SET game board and our AI will automatically detect all valid sets</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def reset_session_state():
    """Reset all session state variables for a clean slate"""
    # Create a list of keys to preserve
    preserved_keys = ['is_mobile', 'models_loaded', 'gpu_info']
    
    # Save the values we want to preserve
    preserved_values = {}
    for key in preserved_keys:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]
    
    # Identify all keys that should be removed
    keys_to_remove = [key for key in st.session_state.keys() if key not in preserved_keys]
    
    # Delete all non-preserved keys
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reinitialize with defaults
    st.session_state.processed = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    st.session_state.error_message = None
    st.session_state.detection_time = None
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value
    
    # Force file uploader to reset by generating a new random key
    st.session_state.uploader_key = str(random.randint(1000, 9999))
    
    # Signal that a reset has been performed
    st.session_state.should_reset = True
    
    # Clear URL params except mobile flag
    params = st.query_params
    mobile_param = params.get('mobile', None)
    
    # Clear all URL params
    st.query_params.clear()
    
    # Restore mobile param if present
    if mobile_param == 'true':
        st.query_params['mobile'] = 'true'

def detect_mobile() -> bool:
    """
    Comprehensive mobile detection using multiple methods.
    Returns True if the device is likely mobile.
    """
    # Check URL parameter first (most reliable)
    params = st.query_params
    if "mobile" in params and params["mobile"] == "true":
        return True
    
    # Use session state if already detected
    if st.session_state.get("is_mobile", False):
        return True
    
    # Check for common mobile user agent patterns
    try:
        user_agent = st.get_user_info().get("user_agent", "").lower()
        mobile_patterns = [
            "mobile", "android", "iphone", "ipad", "ipod", 
            "windows phone", "blackberry", "iemobile", "opera mini"
        ]
        for pattern in mobile_patterns:
            if pattern in user_agent:
                return True
    except:
        pass
    
    # Default to desktop experience
    return False

def render_stats_box(sets_info: List[dict], detection_time: float):
    """Render a stats box with information about the detected sets"""
    stats_html = f"""
    <div class="stats-box">
        <div class="stats-item">
            <span class="stats-label">Sets Found:</span>
            <span class="stats-value">{len(sets_info)}</span>
        </div>
        <div class="stats-item">
            <span class="stats-label">Cards Analyzed:</span>
            <span class="stats-value">{sum(len(s['cards']) for s in sets_info) // 3}</span>
        </div>
        <div class="stats-item">
            <span class="stats-label">Processing Time:</span>
            <span class="stats-value">{detection_time:.2f} seconds</span>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

# =============================================================================
# MOBILE LAYOUT
# =============================================================================
def render_mobile_layout():
    """Render the mobile-optimized layout with auto-processing"""
    # Aggressively ensure sidebar is completely hidden for mobile
    st.markdown(
        """
        <style>
        /* Hide sidebar using multiple approaches for certainty */
        section[data-testid="stSidebar"] {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: hidden !important;
            z-index: -1 !important;
            opacity: 0 !important;
        }
        div[data-testid="stSidebarNav"] {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        /* Expand main content area to full width */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Mobile uploader section with clear styling
    st.markdown(
        '''
        <div class="mobile-uploader">
            <h3>📱 Upload SET Game Image</h3>
            <p>Image will be processed automatically after upload</p>
        </div>
        ''', 
        unsafe_allow_html=True
    )
    
    # File uploader with clear label
    mobile_uploaded_file = st.file_uploader(
        label="Upload your SET game photo",
        type=["png", "jpg", "jpeg"],
        key=f"mobile_uploader_{st.session_state.get('uploader_key', 'default')}",
        help="Take a photo or upload an image of your SET game board"
    )
    
    # Auto-process when a new file is uploaded
    if mobile_uploaded_file is not None and mobile_uploaded_file != st.session_state.get("uploaded_file", None):
        # Reset state for new file
        for key in ['processed', 'processed_image', 'sets_info',
                   'no_cards_detected', 'no_sets_found', 'detection_time']:
            if key in st.session_state:
                if key in ['processed', 'no_cards_detected', 'no_sets_found']:
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None
        
        # Set the new file
        st.session_state.uploaded_file = mobile_uploaded_file
        
        # Start fresh with the new image
        try:
            image = Image.open(mobile_uploaded_file)
            image = optimize_image(image)
            st.session_state.original_image = image
            
            # Auto-start processing (no button needed)
            st.session_state.start_processing = True
            st.rerun()  # Force rerun to show loading state
        except Exception as e:
            st.session_state.error_message = f"Failed to load image: {str(e)}"
            render_message(st.session_state.error_message, "error")
    
    # Show original image if available
    if st.session_state.get("original_image", None) is not None:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, 
               caption="Original Image", 
               use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing or results display
        if st.session_state.get("start_processing", False):
            # Show loading animation
            render_loader("Analyzing game board...")
            
            try:
                # Process image with models (ensuring they exist)
                if not all([card_detection_model, shape_detection_model, fill_model, shape_model]):
                    raise Exception("Models not properly loaded")
                    
                image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
                
                # Extra validation
                if image_cv.size == 0 or min(image_cv.shape) < 10:
                    raise Exception("Invalid image dimensions")
                
                sets_info, processed_image = classify_and_find_sets_from_array(
                    image_cv, card_detection_model, shape_detection_model, fill_model, shape_model
                )
                
                # Update session state
                st.session_state.processed_image = processed_image
                st.session_state.sets_info = sets_info
                st.session_state.processed = True
                st.session_state.start_processing = False
                
                # Force refresh
                st.rerun()
            except Exception as e:
                error_msg = f"An error occurred during processing: {str(e)}"
                st.session_state.error_message = error_msg
                render_message(error_msg, "error")
                st.code(traceback.format_exc(), language="python")
                st.session_state.start_processing = False
        
        # Show processed results if available
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                render_message("No cards were detected. Please try a clearer image of a SET game board.", "error")
            elif st.session_state.get("no_sets_found", False):
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, caption="No sets found", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                render_message("Cards were found but no valid SETs in this board. Try adding more cards!", "warning")
            elif st.session_state.get("processed_image", None) is not None:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, 
                       caption=f"Detected {len(st.session_state.sets_info)} Sets", 
                       use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show stats
                if st.session_state.detection_time and st.session_state.sets_info:
                    render_stats_box(st.session_state.sets_info, st.session_state.detection_time)
            
            # Reset button for mobile - below the image results
            if st.button("⟳ Try Another Image", key="mobile_reset_button", use_container_width=True):
                reset_session_state()
                st.rerun()

# =============================================================================
# DESKTOP LAYOUT
# =============================================================================
def render_desktop_layout():
    """Render the desktop layout with sidebar and two columns"""
    # Setup the sidebar for desktop upload
    with st.sidebar:
        st.markdown('<h3 style="text-align: center; margin-bottom: 15px;">Upload Your Image</h3>', unsafe_allow_html=True)
        
        # Handle file uploading - detect changes
        uploaded_file = st.file_uploader(
            label="Upload SET image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="Upload a photo of your SET game board",
            key=f"desktop_uploader_{st.session_state.uploader_key}"
        )
        
        # If new file is uploaded, reset processing state
        if uploaded_file is not None and uploaded_file != st.session_state.get("uploaded_file", None):
            # Reset state when new file is uploaded
            for key in ['processed', 'processed_image', 'sets_info', 'original_image', 
                       'no_cards_detected', 'no_sets_found', 'detection_time']:
                if key in st.session_state:
                    if key in ['processed', 'no_cards_detected', 'no_sets_found']:
                        st.session_state[key] = False
                    else:
                        st.session_state[key] = None
            
            # Set the new file
            st.session_state.uploaded_file = uploaded_file
            
            # Start fresh with the new image
            try:
                image = Image.open(uploaded_file)
                image = optimize_image(image)
                st.session_state.original_image = image
            except Exception as e:
                st.session_state.error_message = f"Failed to load image: {str(e)}"
                st.error(st.session_state.error_message)
        
        # Process button in sidebar
        if uploaded_file is not None:
            st.markdown('<div style="margin: 20px 0 15px 0;">', unsafe_allow_html=True)
            if st.button("🔎 Find Sets", use_container_width=True):
                # Clear previous results and error flags
                st.session_state.processed = False
                st.session_state.processed_image = None
                st.session_state.sets_info = None
                st.session_state.no_cards_detected = False
                st.session_state.no_sets_found = False
                st.session_state.error_message = None
                st.session_state.start_processing = True
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show GPU info if available
            if 'gpu_info' in st.session_state:
                st.markdown(
                    f'''
                    <div style="margin-top: 15px; font-size: 0.8rem; color: #666;">
                        {st.session_state['gpu_info']}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
    
    # Create two equal columns for the main content
    col1, col2 = st.columns([1, 1])
    
    # Original image column
    with col1:
        st.markdown('<h3>Original Image</h3>', unsafe_allow_html=True)
        
        # Show original image if available
        if st.session_state.get("original_image", None) is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, 
                   use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show placeholder when no image uploaded
            st.markdown(
                '''
                <div style="text-align: center; padding: 50px 20px; background: rgba(124, 58, 237, 0.05); 
                            border-radius: 12px; margin-top: 20px; border: 1px dashed rgba(124, 58, 237, 0.3);">
                    <h4 style="color: #666;">Upload an image from the sidebar</h4>
                    <p style="color: #888; font-size: 0.9rem;">The uploaded SET game image will appear here</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
    
    # Results column
    with col2:
        st.markdown('<h3>Detection Results</h3>', unsafe_allow_html=True)
        
        if st.session_state.get("start_processing", False):
            # Show loading animation
            render_loader("Analyzing game board...")
            
            try:
                # Convert image for processing
                image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
                
                # Process image with models
                sets_info, processed_image = classify_and_find_sets_from_array(
                    image_cv,
                    card_detection_model,
                    shape_detection_model,
                    fill_model,
                    shape_model,
                )
                
                # Update session state
                st.session_state.processed_image = processed_image
                st.session_state.sets_info = sets_info
                st.session_state.processed = True
                st.session_state.start_processing = False
                
                # Force refresh
                st.rerun()
            except Exception as e:
                error_msg = f"An error occurred during processing: {str(e)}"
                st.session_state.error_message = error_msg
                render_message(error_msg, "error")
                st.code(traceback.format_exc(), language="python")
                st.session_state.start_processing = False
                
        # Show processed image if available
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                # Show error message
                render_message("Hmm... I couldn't detect any cards. Make sure your image shows a SET game board clearly.", "error")
                
            elif st.session_state.get("no_sets_found", False):
                # Show the processed image without sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, 
                       use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show warning message
                render_message("I found cards but no valid SETs in this board. The dealer might need to add more cards!", "warning")
                
            elif st.session_state.get("processed_image", None) is not None:
                # Show the processed image with detected sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show success message and stats
                if st.session_state.sets_info:
                    num_sets = len(st.session_state.sets_info)
                    set_text = "set" if num_sets == 1 else "sets"
                    render_message(f"Success! Found {num_sets} {set_text} in your SET game board", "success")
                    
                    # Display stats if available
                    if st.session_state.detection_time:
                        render_stats_box(st.session_state.sets_info, st.session_state.detection_time)
            
            # Reset button - completely clears app state
            st.button("⟳ Analyze New Image", use_container_width=True, on_click=reset_session_state)
                
        # Show a styled message to prompt the user to process
        elif st.session_state.get("original_image", None) is not None and not st.session_state.get("processed", False):
            render_process_message()
        
        else:
            # Show placeholder when no processing has happened
            st.markdown(
                '''
                <div style="text-align: center; padding: 50px 20px; background: rgba(124, 58, 237, 0.05); 
                            border-radius: 12px; margin-top: 20px; border: 1px dashed rgba(124, 58, 237, 0.3);">
                    <h4 style="color: #666;">Detection results will appear here</h4>
                    <p style="color: #888; font-size: 0.9rem;">Upload an image and click "Find Sets" to begin</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application function"""
    # Load CSS
    load_css()
    
    # If reset is pending, clear everything
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # Detect mobile device
    is_mobile = detect_mobile()
    
    # Store in session state
    st.session_state.is_mobile = is_mobile
    
    # Render header
    render_header()
    
    # Load models with error handling if not already loaded
    global card_detection_model, shape_detection_model, fill_model, shape_model
    
    if not st.session_state.get("models_loaded", False):
        try:
            with st.spinner("Loading detection models..."):
                shape_model, fill_model = load_classification_models()
                card_detection_model, shape_detection_model = load_detection_models()
                
                # Verify models loaded successfully
                if all([shape_model, fill_model, card_detection_model, shape_detection_model]):
                    st.session_state.models_loaded = True
                else:
                    raise Exception("One or more models failed to load")
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            st.session_state.error_message = error_msg
            render_message(error_msg, "error")
            return
    else:
        # Models already loaded - retrieve from cache
        shape_model, fill_model = load_classification_models()
        card_detection_model, shape_detection_model = load_detection_models()
    
    # Choose layout based on detected device
    if is_mobile:
        render_mobile_layout()
    else:
        render_desktop_layout()
    
    # Show any persistent error message
    if st.session_state.get("error_message"):
        render_message(st.session_state.error_message, "error")

if __name__ == "__main__":
    main()
