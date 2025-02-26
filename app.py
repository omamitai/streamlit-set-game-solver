import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import time
import random
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple, Any

# Set page configuration
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define SET game colors
SET_COLORS = {
    "primary": "#6A0DAD",  # Purple
    "accent": "#4CAF50",   # Green
    "highlight": "#F44336", # Red
    "purple": "#9C27B0",
    "red": "#E91E63",
    "green": "#4CAF50",
    "background": "#F5F5F5",
    "text": "#212121"
}

# Define CSS for the app
def load_css():
    css = """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #6A0DAD, #4CAF50);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        padding: 10px 0;
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 20px;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(90deg, #6A0DAD, #4CAF50);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Image Container */
    .img-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background: white;
        transition: all 0.3s ease;
    }
    
    .img-container:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: translateY(-3px);
    }
    
    /* Loader */
    .loader-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    .loader {
        display: flex;
        justify-content: space-between;
        width: 80px;
    }
    
    .loader-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loader-dot-1 {
        background-color: #6A0DAD;
        animation-delay: -0.32s;
    }
    
    .loader-dot-2 {
        background-color: #4CAF50;
        animation-delay: -0.16s;
    }
    
    .loader-dot-3 {
        background-color: #F44336;
    }
    
    @keyframes bounce {
        0%, 80%, 100% { 
            transform: scale(0);
        } 40% { 
            transform: scale(1.0);
        }
    }
    
    /* Direction Arrow */
    .direction-arrow {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        height: 40px;
        margin: 20px 0;
    }
    
    .direction-arrow svg {
        width: 40px;
        height: 40px;
    }
    
    .mobile-arrow {
        transform: rotate(90deg);
        margin: 30px 0;
    }
    
    /* System Messages */
    .system-message, .error-message, .warning-message {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
    }
    
    .system-message {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    
    .error-message {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #EF9A9A;
    }
    
    .warning-message {
        background-color: #FFF8E1;
        color: #FF8F00;
        border: 1px solid #FFE082;
    }
    
    /* Mobile Specific Styles */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Mobile Detection Function
def is_mobile():
    """Check if user is on a mobile device"""
    # Using streamlit's experimental feature to get browser info (if available)
    user_agent = ""
    try:
        # This is theoretical and might not work in all Streamlit versions
        user_agent = st.experimental_get_query_params().get("user_agent", [""])[0]
    except:
        pass
    
    # Fallback to screen width detection
    # Set a default mobile state (will be updated via JavaScript)
    if "is_mobile" not in st.session_state:
        st.session_state.is_mobile = False
    
    # Use JavaScript to detect mobile and update session state
    js_mobile_detection = """
    <script>
    const isMobile = window.innerWidth < 768;
    if (isMobile !== %s) {
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: isMobile
        }, "*");
    }
    </script>
    """ % str(st.session_state.is_mobile).lower()
    
    st.markdown(js_mobile_detection, unsafe_allow_html=True)
    
    # Check for keywords in user agent if available
    if user_agent and any(x in user_agent.lower() for x in ["android", "iphone", "ipad", "mobile"]):
        return True
    
    # Return the state stored by JavaScript
    return st.session_state.is_mobile

# Initialize Session State
def init_session_state():
    """Initialize session state variables"""
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
    if "no_cards_detected" not in st.session_state:
        st.session_state.no_cards_detected = False
    if "no_sets_found" not in st.session_state:
        st.session_state.no_sets_found = False
    if "image_height" not in st.session_state:
        st.session_state.image_height = 400  # Default image height
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(random.randint(1000, 9999))

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
    
    # Check if cards are detected
    cards = detect_cards_from_image(processed_image, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], processed_image
    
    card_df = classify_cards_from_board_image(processed_image, card_detector, shape_detector, fill_model, shape_model)
    sets_found = find_sets(card_df)
    
    # Check if sets are found
    if not sets_found:
        st.session_state.no_sets_found = True
        # Just return the original processed image since there are no sets to draw
        return [], processed_image
    
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

def render_arrow(direction="horizontal", image_height=None):
    """Render a SET-themed direction arrow with dynamic positioning"""
    class_name = "mobile-arrow" if direction == "vertical" else ""
    
    # Calculate the margin-top based on image height if provided
    margin_style = ""
    if image_height is not None:
        # Center the arrow vertically to match the image height
        margin_top = max(50, image_height / 2 - 20)  # 20px is half the arrow height
        margin_style = f"style='margin-top: {margin_top}px;'"
    
    arrow_html = f"""
    <div class="direction-arrow {class_name}" {margin_style}>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="url(#gradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{SET_COLORS["purple"]}" />
                    <stop offset="50%" stop-color="{SET_COLORS["primary"]}" />
                    <stop offset="100%" stop-color="{SET_COLORS["accent"]}" />
                </linearGradient>
            </defs>
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
    </div>
    """
    return st.markdown(arrow_html, unsafe_allow_html=True)

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

def reset_session_state():
    """Reset all session state variables to their initial values"""
    # Create a list of all session state keys that need to be preserved
    preserved_keys = ['is_mobile']
    
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
    st.session_state.image_height = 400  # Default image height
    
    # Force file uploader to reset by generating a new random key
    st.session_state.uploader_key = str(random.randint(1000, 9999))

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Initialize the app
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Detect if user is on mobile
    mobile_view = is_mobile()
    st.session_state.is_mobile = mobile_view
    
    # Page Header - Same for both mobile and desktop
    st.markdown('<h1 class="main-header">SET Game Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image to find valid SET combinations</p>', unsafe_allow_html=True)
    
    # Load models
    try:
        with st.spinner("Loading models..."):
            shape_model, fill_model = load_classification_models()
            card_detection_model, shape_detection_model = load_detection_models()
            models_loaded = all([shape_model, fill_model, card_detection_model, shape_detection_model])
            if not models_loaded:
                st.error("Failed to load all required models. Please refresh and try again.")
                return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Mobile Layout
    if mobile_view:
        # Hide sidebar
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {display: none;}
            section[data-testid="stSidebarUserContent"] {display: none;}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Main content area for mobile
        # File uploader in main area
        uploaded_file = st.file_uploader("Upload SET Game Image", 
                                         type=["jpg", "jpeg", "png"], 
                                         key=f"mobile_uploader_{st.session_state.uploader_key}")
        
        # Auto-process on mobile
        if uploaded_file and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.start_processing = True
            st.session_state.processed = False  # Reset processing flag
            st.session_state.no_cards_detected = False
            st.session_state.no_sets_found = False
            
            # Read and optimize image
            image_bytes = uploaded_file.getvalue()
            pil_image = Image.open(io.BytesIO(image_bytes))
            optimized_image = optimize_image(pil_image)
            
            # Store in session state
            st.session_state.original_image = optimized_image
            
            # Convert for processing
            open_cv_image = np.array(optimized_image)
            # Convert RGB to BGR for OpenCV processing
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            
            with st.spinner("Processing image..."):
                # Process image
                sets_info, processed_cv_image = classify_and_find_sets_from_array(
                    open_cv_image, 
                    card_detection_model, 
                    shape_detection_model, 
                    fill_model, 
                    shape_model
                )
                
                # Convert back to RGB for display
                processed_rgb_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGR2RGB)
                st.session_state.processed_image = Image.fromarray(processed_rgb_image)
                st.session_state.sets_info = sets_info
                st.session_state.processed = True
        
        # Display original image
        if st.session_state.original_image:
            st.subheader("Original Image")
            st.image(st.session_state.original_image, use_column_width=True)
            
            # Calculate image height for arrow positioning
            img_height = st.session_state.original_image.height
            st.session_state.image_height = img_height
            
            # Vertical arrow for mobile
            render_arrow(direction="vertical")
            
            # Show processing results
            if st.session_state.processed:
                st.subheader("Processed Image")
                
                # Handle different scenarios
                if st.session_state.no_cards_detected:
                    render_error_message("No cards detected. Is this a SET game image?")
                    st.image(st.session_state.original_image, use_column_width=True)
                elif st.session_state.no_sets_found:
                    render_warning_message("Cards detected, but no valid sets found.")
                    st.image(st.session_state.processed_image, use_column_width=True)
                else:
                    # Show successful results
                    st.image(st.session_state.processed_image, use_column_width=True)
                    
                # Reset button
                if st.button("Analyze Another Image", key="mobile_reset"):
                    reset_session_state()
                    st.rerun()
    
    # Desktop Layout
    else:
        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-header">Upload Image</div>', unsafe_allow_html=True)
            
            # File uploader in sidebar for desktop
            uploaded_file = st.file_uploader("Choose a SET game image", 
                                            type=["jpg", "jpeg", "png"], 
                                            key=f"desktop_uploader_{st.session_state.uploader_key}")
            
            # Store file in session state
            if uploaded_file and uploaded_file != st.session_state.uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.processed = False  # Reset processing flag
                st.session_state.no_cards_detected = False
                st.session_state.no_sets_found = False
                
                # Read and optimize image
                image_bytes = uploaded_file.getvalue()
                pil_image = Image.open(io.BytesIO(image_bytes))
                optimized_image = optimize_image(pil_image)
                
                # Store in session state
                st.session_state.original_image = optimized_image
            
            # Process button (only for desktop)
            if st.session_state.original_image and not st.session_state.processed:
                if st.button("Find Sets"):
                    st.session_state.start_processing = True
            
            # Reset button
            if st.session_state.processed:
                if st.button("Analyze Another Image"):
                    reset_session_state()
                    st.rerun()
        
        # Main content area - desktop uses two-column layout
        if st.session_state.original_image:
            # Create two columns for original and processed images
            col1, col_arrow, col2 = st.columns([5, 1, 5])
            
            with col1:
                st.subheader("Original Image")
                st.image(st.session_state.original_image, use_column_width=True)
                
                # Calculate and save image height for arrow positioning
                img_height = st.session_state.original_image.height
                st.session_state.image_height = img_height
            
            # Arrow column
            with col_arrow:
                # Display a processing message or arrow depending on the state
                if not st.session_state.processed and not st.session_state.start_processing:
                    render_process_message()
                else:
                    # Render arrow with dynamic vertical positioning
                    render_arrow(direction="horizontal", image_height=st.session_state.image_height)
            
            with col2:
                st.subheader("Processed Image")
                
                # If processing is requested but not yet completed
                if st.session_state.start_processing and not st.session_state.processed:
                    with st.spinner("Processing image..."):
                        # Convert for processing
                        open_cv_image = np.array(st.session_state.original_image)
                        # Convert RGB to BGR for OpenCV processing
                        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                        
                        # Process image
                        sets_info, processed_cv_image = classify_and_find_sets_from_array(
                            open_cv_image, 
                            card_detection_model, 
                            shape_detection_model, 
                            fill_model, 
                            shape_model
                        )
                        
                        # Convert back to RGB for display
                        processed_rgb_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGR2RGB)
                        st.session_state.processed_image = Image.fromarray(processed_rgb_image)
                        st.session_state.sets_info = sets_info
                        st.session_state.processed = True
                        st.session_state.start_processing = False
                        
                        # Force a rerender
                        st.rerun()
                
                # Show processing results
                if st.session_state.processed:
                    if st.session_state.no_cards_detected:
                        render_error_message("No cards detected. Is this a SET game image?")
                        st.image(st.session_state.original_image, use_column_width=True)
                    elif st.session_state.no_sets_found:
                        render_warning_message("Cards detected, but no valid sets found.")
                        st.image(st.session_state.processed_image, use_column_width=True)
                    else:
                        # Show successful results
                        st.image(st.session_state.processed_image, use_column_width=True)
                else:
                    # Placeholder for before processing starts
                    st.info("Upload an image and click 'Find Sets' to see the results here.")

if __name__ == "__main__":
    main()
