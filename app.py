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
    
    /* Empty state message */
    .empty-message {{
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }}
    
    .empty-message p {{
        font-size: 1.1rem;
        margin: 0;
        font-weight: 500;
        color: #6d28d9;
    }}
    
    /* Direction arrow */
    .direction-arrow {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        padding: 1rem 0;
    }}
    
    .direction-arrow svg {{
        width: 40px;
        height: 40px;
        filter: drop-shadow(0 0 8px rgba(124, 58, 237, 0.5));
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
    
    /* Processing placeholder */
    .processing-placeholder {{
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        height: 70%;
        margin-top: 2rem;
    }}
    
    /* For mobile devices */
    @media (max-width: 991px) {{
        /* Hide sidebar on mobile */
        section[data-testid="stSidebar"] {{
            display: none;
        }}
        
        .set-header h1 {{ 
            font-size: 1.8rem; 
        }}
        
        .set-header p {{ 
            font-size: 0.9rem; 
        }}
        
        /* More compact for mobile */
        .set-header {{
            padding: 0.8rem;
            margin-bottom: 0.8rem;
        }}
        
        /* Adjust arrow for mobile */
        .mobile-arrow {{
            transform: rotate(90deg);
            margin: 0.5rem 0;
        }}
        
        /* Mobile layout adjustments */
        .mobile-container {{
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .stButton>button {{
            padding: 0.6rem 1rem;
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

def render_arrow(direction="horizontal"):
    """Render a SET-themed direction arrow"""
    class_name = "mobile-arrow" if direction == "vertical" else ""
    arrow_html = f"""
    <div class="direction-arrow {class_name}">
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

def render_empty_message():
    """Render a styled empty state message"""
    message_html = """
    <div class="empty-message">
        <p>Please upload an image to detect sets</p>
    </div>
    """
    return st.markdown(message_html, unsafe_allow_html=True)

def reset_session_state():
    """Reset all session state variables to their initial values"""
    st.session_state.processed = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None

# =============================================================================
# HEADER
# =============================================================================
def render_header():
    """Render the SET-themed header"""
    header_html = """
    <div class="set-header">
        <h1>🎴 SET Game Detector</h1>
        <p>Upload an image of a SET game board and detect all valid sets</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load CSS
    load_css()
    
    # Render header
    render_header()
    
    # Create columns for desktop layout first (even if we're on mobile)
    # This ensures consistent layout structure
    col1, col_arrow, col2 = st.columns([5, 1, 5])
    
    # Setup the sidebar for desktop upload
    with st.sidebar:
        st.markdown('<h3 style="text-align: center;">Upload Your Image</h3>', unsafe_allow_html=True)
        
        # File uploader in sidebar for desktop
        uploaded_file = st.file_uploader(
            label="Upload SET image",  # Added label to fix warning
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",  # Hide the label visually
            help="Upload a photo of your SET game board"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            # Process button
            if st.button("🔎 Find Sets"):
                # Clear previous results when starting new processing
                st.session_state.processed = False
                st.session_state.processed_image = None
                st.session_state.sets_info = None
                st.session_state.start_processing = True
    
    # For mobile view - additional file uploader
    is_mobile = st.session_state.is_mobile
    
    if is_mobile:
        # Only show mobile uploader if no file is uploaded yet
        if not st.session_state.uploaded_file:
            mobile_uploaded_file = st.file_uploader(
                label="Upload SET image (mobile)",  # Added label to fix warning
                type=["png", "jpg", "jpeg"],
                key="mobile_uploader",
                label_visibility="collapsed",  # Hide the label visually
                help="Upload a photo of your SET game board"
            )
            
            if mobile_uploaded_file is not None:
                st.session_state.uploaded_file = mobile_uploaded_file
                # Process button for mobile
                if st.button("🔎 Find Sets", key="mobile_button"):
                    # Clear previous results when starting new processing
                    st.session_state.processed = False
                    st.session_state.processed_image = None
                    st.session_state.sets_info = None
                    st.session_state.start_processing = True
    
    # Load image if uploaded but not yet loaded
    if st.session_state.uploaded_file is not None and st.session_state.original_image is None:
        try:
            image = Image.open(st.session_state.uploaded_file)
            image = optimize_image(image)
            st.session_state.original_image = image
        except Exception as e:
            st.error(f"Failed to load image: {str(e)}")
    
    # Original image column
    with col1:
        # Show original image if available
        if st.session_state.original_image is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, 
                   caption="Original Image", 
                   use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show placeholder message when no image is uploaded
            render_empty_message()
    
    # Arrow column - only show if an image is uploaded
    with col_arrow:
        if st.session_state.original_image is not None:
            render_arrow()
    
    # Results column
    with col2:
        if st.session_state.start_processing:
            # Show smaller loading animation
            render_loader()
            
            try:
                # Convert image for processing without spinner text
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
        elif st.session_state.processed and st.session_state.processed_image is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_img, 
                   caption=f"Detected {len(st.session_state.sets_info)} Sets", 
                   use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("⟳ Analyze Another Image"):
                # Reset ALL state variables using the dedicated function
                reset_session_state()
                # Force refresh
                st.rerun()
                
        # Show a placeholder before processing if image is uploaded
        elif st.session_state.original_image is not None and not st.session_state.processed:
            # Simple placeholder that doesn't take too much space
            st.markdown('<div class="processing-placeholder">', unsafe_allow_html=True)
            st.markdown('<p>Click "Find Sets" to process the image</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
