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
import base64
import random
import time

# =============================================================================
# CONFIGURATION 
# =============================================================================
st.set_page_config(layout="wide", page_title="SET Game Detector")

# Define SET theme colors
SET_COLORS = {
    "primary": "#7C3AED",
    "secondary": "#10B981",
    "accent": "#EC4899",
    "red": "#EF4444",
    "green": "#10B981",
    "purple": "#8B5CF6",
}

# =============================================================================
# INITIALIZE SESSION STATE - Moved to top to ensure initialization before access
# =============================================================================
# Initialize all session state variables with default values
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

# =============================================================================
# CSS STYLING
# =============================================================================
def load_css():
    """Load custom CSS with responsive design"""
    css = f"""
    <style>
    /* Global styles with SET theme */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    body {{
        font-family: 'Poppins', sans-serif;
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
        padding: 1rem;
        border-radius: 12px;
    }}
    
    .set-header h1 {{
        font-size: 2.5rem;
        margin-bottom: 0;
        background: linear-gradient(90deg, {SET_COLORS["purple"]} 0%, {SET_COLORS["primary"]} 50%, {SET_COLORS["accent"]} 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .set-header p {{
        font-size: 1.1rem;
        opacity: 0.8;
    }}
    
    /* Card styles */
    .set-card {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 1rem;
    }}
    
    .set-card h3 {{
        margin-top: 0;
        border-bottom: 2px solid {SET_COLORS["primary"]};
        padding-bottom: 0.5rem;
        text-align: center;
    }}
    
    /* Upload area */
    .upload-area {{
        border: 2px dashed rgba(124, 58, 237, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        background-color: rgba(124, 58, 237, 0.05);
        cursor: pointer;
        margin-bottom: 1rem;
    }}
    
    .upload-area:hover {{
        border-color: {SET_COLORS["primary"]};
        background-color: rgba(124, 58, 237, 0.1);
    }}
    
    /* Button styling */
    .stButton>button {{
        background: linear-gradient(90deg, {SET_COLORS["primary"]} 0%, {SET_COLORS["accent"]} 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }}
    
    .stButton>button:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }}
    
    /* Loading animation */
    .loader-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }}
    
    .loader {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .loader-dot {{
        width: 10px;
        height: 10px;
        margin: 0 6px;
        border-radius: 50%;
        display: inline-block;
        animation: loader 1.5s infinite ease-in-out both;
    }}
    
    .loader-dot-1 {{
        background-color: {SET_COLORS["red"]};
        animation-delay: -0.3s;
    }}
    
    .loader-dot-2 {{
        background-color: {SET_COLORS["green"]};
        animation-delay: -0.15s;
    }}
    
    .loader-dot-3 {{
        background-color: {SET_COLORS["purple"]};
        animation-delay: 0s;
    }}
    
    @keyframes loader {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    
    /* Image container */
    .image-container {{
        margin-top: 0.5rem;
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }}
    
    /* Processing placeholder (system message style) */
    .system-message {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    
    .system-message p {{
        font-size: 1.1rem;
        font-weight: 400;
        color: #666666;
        margin: 0;
    }}
    
    /* Error messages */
    .error-message {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid {SET_COLORS["red"]};
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }}
    
    .error-message p {{
        margin: 0;
        font-weight: 500;
        color: {SET_COLORS["red"]};
    }}
    
    .warning-message {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }}
    
    .warning-message p {{
        margin: 0;
        font-weight: 500;
        color: #F59E0B;
    }}
    
    /* Mobile uploader styling */
    .mobile-uploader {{
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0 1.5rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    
    .mobile-uploader h3 {{
        color: {SET_COLORS["primary"]};
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
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
            font-size: 1.8rem; 
        }}
        
        .set-header p {{ 
            font-size: 0.9rem; 
        }}
        
        .set-header {{
            padding: 0.8rem;
            margin-bottom: 1rem;
        }}
        
        /* Button adjustments for touch */
        .stButton>button {{
            padding: 0.8rem 1rem;
            font-size: 1rem;
            height: auto;
            min-height: 3rem;
        }}
        
        /* Ensure images are fully responsive */
        .image-container img {{
            width: 100% !important;
            height: auto !important;
            max-width: 100% !important;
        }}
        
        /* Tighter message spacing */
        .system-message, .error-message, .warning-message {{
            margin: 1rem 0;
            padding: 1rem;
        }}
        
        /* Better loader sizing */
        .loader-container {{
            height: 150px;
        }}
        
        .loader-dot {{
            width: 12px;
            height: 12px;
        }}
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
    try:
        shape_model = load_model(str(characteristics_path / "shape_model.keras"))
        fill_model = load_model(str(characteristics_path / "fill_model.keras"))
        return shape_model, fill_model
    except Exception as e:
        st.error(f"Error loading classification models: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    try:
        shape_detection_model = YOLO(str(shape_path / "best.pt"))
        shape_detection_model.conf = 0.5
        card_detection_model = YOLO(str(card_path / "best.pt"))
        card_detection_model.conf = 0.5
        if torch.cuda.is_available():
            card_detection_model.to("cuda")
            shape_detection_model.to("cuda")
        return card_detection_model, shape_detection_model
    except Exception as e:
        st.error(f"Error loading detection models: {str(e)}")
        return None, None

# Load models with error handling
try:
    shape_model, fill_model = load_classification_models()
    card_detection_model, shape_detection_model = load_detection_models()
    models_loaded = all([shape_model, fill_model, card_detection_model, shape_detection_model])
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

# =============================================================================
# UTILITY & PROCESSING FUNCTIONS
# =============================================================================
def check_and_rotate_input_image(board_image: np.ndarray, detector: YOLO) -> Tuple[np.ndarray, bool]:
    """Check if image needs rotation and rotate if needed"""
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
    """Restore the original orientation of the image"""
    if was_rotated:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def predict_color(shape_image: np.ndarray) -> str:
    """Predict the color of a shape using HSV color filtering"""
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
    """Extract features from a single card"""
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
    """Check if 3 cards form a valid set"""
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        if len({card[feature] for card in cards}) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> List[dict]:
    """Find all valid sets in the cards"""
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
    """Detect cards in the board image"""
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [
        (board_image[y1:y2, x1:x2], [x1, y1, x2, y2])
        for x1, y1, x2, y2 in card_boxes
    ]

def classify_cards_from_board_image(board_image: np.ndarray, card_detector: YOLO,
                                      shape_detector: YOLO, fill_model: tf.keras.Model,
                                      shape_model: tf.keras.Model) -> pd.DataFrame:
    """Classify all cards detected in the board image"""
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
    """Draw rectangles and labels around sets on the image"""
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
    """Process the board image to find sets"""
    # Check if rotation is needed
    processed_image, was_rotated = check_and_rotate_input_image(board_image, card_detector)
    
    # Check if cards are detected
    cards = detect_cards_from_image(processed_image, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], processed_image
    
    # Classify cards and find sets
    card_df = classify_cards_from_board_image(processed_image, card_detector, shape_detector, fill_model, shape_model)
    sets_found = find_sets(card_df)
    
    # Check if sets are found
    if not sets_found:
        st.session_state.no_sets_found = True
        return [], processed_image
    
    # Draw sets on the image
    annotated_image = draw_sets_on_image(processed_image.copy(), sets_found)
    final_image = restore_original_orientation(annotated_image, was_rotated)
    
    return sets_found, final_image

def optimize_image(image, max_size=1200):
    """Resize large images to improve performance"""
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
def render_loader():
    """Render a SET-themed loader"""
    loader_html = """
    <div class="loader-container">
        <div class="loader">
            <div class="loader-dot loader-dot-1"></div>
            <div class="loader-dot loader-dot-2"></div>
            <div class="loader-dot loader-dot-3"></div>
        </div>
    </div>
    """
    return st.markdown(loader_html, unsafe_allow_html=True)

def render_error_message(message):
    """Render a styled error message"""
    error_html = f"""
    <div class="error-message">
        <p>{message}</p>
    </div>
    """
    return st.markdown(error_html, unsafe_allow_html=True)

def render_warning_message(message):
    """Render a styled warning message"""
    warning_html = f"""
    <div class="warning-message">
        <p>{message}</p>
    </div>
    """
    return st.markdown(warning_html, unsafe_allow_html=True)

def render_process_message():
    """Render a styled system message to prompt user to process the image"""
    message_html = """
    <div class="system-message">
        <p>Click "Find Sets" to process the image</p>
    </div>
    """
    return st.markdown(message_html, unsafe_allow_html=True)

def render_header():
    """Render the SET-themed header"""
    header_html = """
    <div class="set-header">
        <h1>🎴 SET Game Detector</h1>
        <p>Upload an image of a SET game board and detect all valid sets</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def reset_session_state():
    """Reset all session state variables to their initial values"""
    # Create a list of all session state keys that need to be preserved
    preserved_keys = ['is_mobile']
    
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
    
    # Now reinitialize with clean defaults
    st.session_state.processed = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value
    
    # Force file uploader to reset by generating a new random key
    st.session_state.uploader_key = str(random.randint(1000, 9999))
    
    # Signal to the app that a full reset is needed
    st.session_state.should_reset = True
    
    # Add a timestamp to force a complete refresh
    st.session_state.reset_timestamp = time.time()
    
    # Don't clear URL params if they include mobile=true
    params = st.query_params
    mobile_param = params.get('mobile', None)
    
    # Clear URL params
    st.query_params.clear()
    
    # Restore mobile param if it was present
    if mobile_param == 'true':
        st.query_params['mobile'] = 'true'

def detect_mobile():
    """Simple mobile detection based on user agent or session state"""
    # Add JavaScript to detect screen size
    st.markdown("""
    <script>
        // Detect mobile based on screen width
        const isMobile = window.innerWidth <= 991;
        if (isMobile) {
            document.body.classList.add('mobile-view');
        }
    </script>
    """, unsafe_allow_html=True)
    
    # Try to detect mobile based on user agent in request header
    try:
        # Attempt to access user agent from request headers
        if hasattr(st, "request"):
            request_headers = getattr(st, "request", None)
            if request_headers and hasattr(request_headers, "headers"):
                user_agent = request_headers.headers.get("User-Agent", "").lower()
                if "mobile" in user_agent or "android" in user_agent or "iphone" in user_agent:
                    return True
    except:
        pass
    
    # Simple check for common mobile/touch devices via screen width
    # For simplicity, we use a fixed "mobile" parameter that can be toggled via URL
    params = st.query_params
    if params.get("mobile", [False])[0]:
        return True
        
    # Fallback to session state
    return st.session_state.get("is_mobile", False)

# =============================================================================
# MOBILE LAYOUT
# =============================================================================
def render_mobile_layout():
    """Render the mobile-optimized layout with auto-processing"""
    # Ensure sidebar is hidden for mobile
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {display: none !important;}
        div[data-testid="stSidebarNav"] {display: none !important;}
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Mobile uploader section with clear styling
    st.markdown(
        '''
        <div style="text-align: center; margin-bottom: 20px; padding: 15px; 
                    background: linear-gradient(90deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%); 
                    border-radius: 12px;">
            <h3 style="margin-bottom: 10px; color: #7C3AED;">📱 Upload SET Game Image</h3>
            <p style="font-size: 0.9rem; color: #666;">Image will be processed automatically after upload</p>
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
                   'no_cards_detected', 'no_sets_found']:
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
            st.error(f"Failed to load image: {str(e)}")
    
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
            render_loader()
            
            try:
                # Process image with models
                image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
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
                st.error("⚠️ An error occurred during processing:")
                st.code(traceback.format_exc())
                st.session_state.start_processing = False
        
        # Show processed results if available
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                render_error_message("No cards were detected. Please try a clearer image of a SET game board.")
            elif st.session_state.get("no_sets_found", False):
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, caption="Processed Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                render_warning_message("Cards were found but no valid SETs in this board. Try adding more cards!")
            elif st.session_state.get("processed_image", None) is not None:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, 
                       caption=f"Detected {len(st.session_state.sets_info)} Sets", 
                       use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
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
        st.markdown('<h3 style="text-align: center;">Upload Your Image</h3>', unsafe_allow_html=True)
        
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
                       'no_cards_detected', 'no_sets_found']:
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
                st.error(f"Failed to load image: {str(e)}")
        
        # Process button in sidebar
        if uploaded_file is not None:
            if st.button("🔎 Find Sets", use_container_width=True):
                # Clear previous results and error flags
                st.session_state.processed = False
                st.session_state.processed_image = None
                st.session_state.sets_info = None
                st.session_state.no_cards_detected = False
                st.session_state.no_sets_found = False
                st.session_state.start_processing = True
    
    # Create two equal columns for the main content
    col1, col2 = st.columns([1, 1])
    
    # Original image column
    with col1:
        # Show original image if available
        if st.session_state.get("original_image", None) is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, 
                   caption="Original Image", 
                   use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Results column
    with col2:
        if st.session_state.get("start_processing", False):
            # Show loading animation
            render_loader()
            
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
                st.error("⚠️ An error occurred during processing:")
                st.code(traceback.format_exc())
                st.session_state.start_processing = False
                
        # Show processed image if available
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                # Show error message
                render_error_message("Hmm... I couldn't detect any cards. Make sure your image shows a SET game board clearly.")
                
            elif st.session_state.get("no_sets_found", False):
                # Show the processed image without sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, 
                       caption="Processed Image", 
                       use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show warning message
                render_warning_message("I found cards but no valid SETs in this board. The dealer might need to add more cards!")
                
            elif st.session_state.get("processed_image", None) is not None:
                # Show the processed image with detected sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_img, 
                       caption=f"Detected {len(st.session_state.sets_info)} Sets", 
                       use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button - completely clears app state
            st.button("⟳ Analyze New Image", use_container_width=True, on_click=reset_session_state)
                
        # Show a styled message to prompt the user to process
        elif st.session_state.get("original_image", None) is not None and not st.session_state.get("processed", False):
            render_process_message()

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
    
    # Check for mobile parameter in URL
    params = st.query_params
    if "mobile" in params:
        is_mobile = params["mobile"] == "true"
    else:
        # Otherwise use basic detection (this could be enhanced)
        import re
        user_agent = st.get_user_info().get("user_agent", "")
        is_mobile = bool(re.search(r'Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini', user_agent))
    
    # Store in session state
    st.session_state.is_mobile = is_mobile
    
    # Render header
    render_header()
    
    # Choose layout based on detected device
    if is_mobile:
        # Hide sidebar for mobile
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {display: none !important;}
            div[data-testid="stSidebarNav"] {display: none !important;}
            </style>
            """,
            unsafe_allow_html=True
        )
        render_mobile_layout()
    else:
        render_desktop_layout()

if __name__ == "__main__":
    main()
