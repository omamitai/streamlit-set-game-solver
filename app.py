"""
SET Game Detector Streamlit App
===============================

A modern, responsive app that detects valid sets from SET game images
using computer vision and machine learning, with optimized interfaces
for both mobile and desktop devices.
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
from typing import Tuple, List, Dict, Optional, Union
import time
import base64

# =============================================================================
# CONFIGURATION & DEVICE DETECTION
# =============================================================================
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect device type via query parameter (reloads page once)
query_params = st.query_params
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
# SESSION STATE INITIALIZATION
# =============================================================================
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.is_mobile = is_mobile
    st.session_state.processing = False
    st.session_state.processing_complete = False
    st.session_state.processing_time = 0
    st.session_state.card_count = 0
    st.session_state.set_count = 0
    st.session_state.error = None

# =============================================================================
# STYLING & THEME
# =============================================================================

# SET game color palette
SET_COLORS = {
    "purple": "#712F79",  # SET purple
    "green": "#148F77",   # SET green
    "red": "#CB4335",     # SET red
    "bg_light": "#F8F9FA",
    "bg_card": "#FFFFFF",
    "text_primary": "#212529",
    "text_secondary": "#6C757D",
    "accent": "#32373D",
    "success": "#28A745",
    "warning": "#FFC107",
    "error": "#DC3545"
}

# Custom CSS with SET theme and responsive design
css = f"""
<style>
    /* SET Theme Colors */
    :root {{
        --set-purple: {SET_COLORS["purple"]};
        --set-green: {SET_COLORS["green"]};
        --set-red: {SET_COLORS["red"]};
        --bg-light: {SET_COLORS["bg_light"]};
        --bg-card: {SET_COLORS["bg_card"]};
        --text-primary: {SET_COLORS["text_primary"]};
        --text-secondary: {SET_COLORS["text_secondary"]};
        --accent: {SET_COLORS["accent"]};
        --success: {SET_COLORS["success"]};
        --warning: {SET_COLORS["warning"]};
        --error: {SET_COLORS["error"]};
    }}
    
    /* Global Styles */
    .stApp {{
        background-color: var(--bg-light);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary);
    }}
    
    p, span, div {{
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }}
    
    /* SET Header */
    .set-header {{
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    .set-header h1 {{
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, var(--set-purple), var(--set-green), var(--set-red));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }}
    
    .set-header p {{
        color: var(--text-secondary);
        font-size: 1.2rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }}
    
    /* Card Styles */
    .set-card {{
        background-color: var(--bg-card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-top: 5px solid transparent;
        transition: all 0.3s ease;
    }}
    
    .set-card-purple {{
        border-top-color: var(--set-purple);
    }}
    
    .set-card-green {{
        border-top-color: var(--set-green);
    }}
    
    .set-card-red {{
        border-top-color: var(--set-red);
    }}
    
    .set-card h3 {{
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: var(--accent);
    }}
    
    /* Uploader Styles */
    .uploader-container {{
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background-color: var(--bg-card);
        transition: all 0.3s ease;
    }}
    
    .uploader-container:hover {{
        border-color: var(--set-green);
    }}
    
    .uploader-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--set-green);
    }}
    
    /* Results Display */
    .results-container {{
        text-align: center;
    }}
    
    .results-heading {{
        font-size: 1.8rem;
        margin-bottom: 1rem;
        color: var(--accent);
    }}
    
    .set-stats {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
    }}
    
    .stat-item {{
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--set-purple);
    }}
    
    .stat-label {{
        font-size: 1rem;
        color: var(--text-secondary);
    }}
    
    /* Image Container */
    .image-container {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 0 auto;
        max-width: 100%;
        text-align: center;
    }}
    
    /* Button Styles */
    .set-button {{
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    
    .primary-button {{
        background-color: var(--set-purple);
        color: white;
    }}
    
    .primary-button:hover {{
        background-color: #5a2e60;
    }}
    
    .secondary-button {{
        background-color: var(--set-green);
        color: white;
    }}
    
    .secondary-button:hover {{
        background-color: #0e6655;
    }}
    
    /* Loading Animation */
    .loading-container {{
        text-align: center;
        padding: 2rem;
    }}
    
    .loader {{
        display: inline-block;
        width: 80px;
        height: 80px;
        margin-bottom: 1rem;
    }}
    
    .loader:after {{
        content: " ";
        display: block;
        width: 64px;
        height: 64px;
        margin: 8px;
        border-radius: 50%;
        border: 6px solid;
        border-color: var(--set-purple) transparent var(--set-green) transparent;
        animation: loader 1.2s linear infinite;
    }}
    
    @keyframes loader {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .loading-text {{
        font-size: 1.2rem;
        color: var(--text-secondary);
    }}
    
    /* Footer */
    .set-footer {{
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }}
    
    /* Responsive Adjustments */
    /* Mobile Specific */
    @media (max-width: 768px) {{
        .set-header h1 {{
            font-size: 2.2rem;
        }}
        
        .set-header p {{
            font-size: 1rem;
            padding: 0 1rem;
        }}
        
        .set-card {{
            padding: 1rem;
        }}
        
        .uploader-container {{
            padding: 1.5rem 1rem;
        }}
        
        .stat-value {{
            font-size: 2rem;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
        }}
        
        .set-stats {{
            gap: 1rem;
        }}
        
        .mobile-stack {{
            flex-direction: column;
        }}
        
        .mobile-hidden {{
            display: none;
        }}
    }}
    
    /* Desktop Specific */
    @media (min-width: 769px) {{
        .desktop-flex {{
            display: flex;
            gap: 2rem;
        }}
        
        .desktop-hidden {{
            display: none;
        }}
        
        .sidebar-container {{
            max-width: 400px;
            padding-right: 2rem;
        }}
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.5s ease-in-out;
    }}
    
    /* Customizing Streamlit Elements */
    .stButton > button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        background-color: var(--set-purple);
        color: white;
    }}
    
    .stButton > button:hover {{
        background-color: #5a2e60;
    }}
    
    /* Hide Streamlit elements we don't want */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    
    div.stFileUploader > div:first-child {{
        width: 100%;
    }}
</style>

<!-- Import Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap" rel="stylesheet">
"""

st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_svg_card(color="purple", shape="diamond", fill="empty", count=1):
    """Generate SVG for SET card symbols based on attributes."""
    card_width, card_height = 140, 70
    
    # Define colors
    colors = {
        "purple": SET_COLORS["purple"],
        "green": SET_COLORS["green"],
        "red": SET_COLORS["red"]
    }
    
    # Shape properties
    shape_width, shape_height = 30, 15
    
    # Start SVG
    svg = f'<svg width="{card_width}" height="{card_height}" viewBox="0 0 {card_width} {card_height}" xmlns="http://www.w3.org/2000/svg">'
    
    # Background
    svg += f'<rect width="{card_width}" height="{card_height}" rx="10" fill="white" stroke="#E0E0E0" stroke-width="1"/>'
    
    # Calculate positions based on count
    positions = []
    if count == 1:
        positions = [(card_width/2, card_height/2)]
    elif count == 2:
        gap = 15
        positions = [
            (card_width/2, card_height/2 - gap),
            (card_width/2, card_height/2 + gap)
        ]
    elif count == 3:
        gap = 18
        positions = [
            (card_width/2, card_height/2 - gap),
            (card_width/2, card_height/2),
            (card_width/2, card_height/2 + gap)
        ]
    
    # Draw shapes
    for cx, cy in positions:
        if shape == "diamond":
            points = f"{cx-shape_width/2},{cy} {cx},{cy-shape_height/2} {cx+shape_width/2},{cy} {cx},{cy+shape_height/2}"
            if fill == "empty":
                svg += f'<polygon points="{points}" fill="none" stroke="{colors[color]}" stroke-width="2"/>'
            elif fill == "full":
                svg += f'<polygon points="{points}" fill="{colors[color]}" stroke="{colors[color]}" stroke-width="1"/>'
            else:  # striped
                svg += f'<polygon points="{points}" fill="url(#stripes-{color})" stroke="{colors[color]}" stroke-width="2"/>'
        
        elif shape == "oval":
            if fill == "empty":
                svg += f'<ellipse cx="{cx}" cy="{cy}" rx="{shape_width/2}" ry="{shape_height/2}" fill="none" stroke="{colors[color]}" stroke-width="2"/>'
            elif fill == "full":
                svg += f'<ellipse cx="{cx}" cy="{cy}" rx="{shape_width/2}" ry="{shape_height/2}" fill="{colors[color]}" stroke="{colors[color]}" stroke-width="1"/>'
            else:  # striped
                svg += f'<ellipse cx="{cx}" cy="{cy}" rx="{shape_width/2}" ry="{shape_height/2}" fill="url(#stripes-{color})" stroke="{colors[color]}" stroke-width="2"/>'
        
        elif shape == "squiggle":
            # Simplified squiggle as rounded rectangle
            if fill == "empty":
                svg += f'<rect x="{cx-shape_width/2}" y="{cy-shape_height/2}" width="{shape_width}" height="{shape_height}" rx="7" fill="none" stroke="{colors[color]}" stroke-width="2"/>'
            elif fill == "full":
                svg += f'<rect x="{cx-shape_width/2}" y="{cy-shape_height/2}" width="{shape_width}" height="{shape_height}" rx="7" fill="{colors[color]}" stroke="{colors[color]}" stroke-width="1"/>'
            else:  # striped
                svg += f'<rect x="{cx-shape_width/2}" y="{cy-shape_height/2}" width="{shape_width}" height="{shape_height}" rx="7" fill="url(#stripes-{color})" stroke="{colors[color]}" stroke-width="2"/>'
    
    # Add stripe patterns if needed
    if "striped" in fill:
        for c in colors:
            svg += f'''
            <defs>
                <pattern id="stripes-{c}" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">
                    <line x1="0" y="0" x2="0" y2="4" stroke="{colors[c]}" stroke-width="2" />
                </pattern>
            </defs>
            '''
    
    # Close SVG
    svg += '</svg>'
    return svg

def render_custom_header():
    """Render the custom SET-themed header."""
    st.markdown(
        """
        <div class="set-header">
            <h1>SET Game Detector</h1>
            <p>Upload an image of your SET game board and let AI find all valid sets!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def get_card_icon_html():
    """Generate HTML for the card upload icon."""
    return """
    <div class="uploader-icon">
        <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="6" width="18" height="15" rx="2" stroke="currentColor" stroke-width="2"/>
            <path d="M16 2V6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M8 2V6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M3 11H21" stroke="currentColor" stroke-width="2"/>
            <path d="M9 16L11 14M11 14L13 16M11 14V19" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    """

def custom_file_uploader(key):
    """Create a custom styled file uploader with SET theme."""
    uploaded_file = None
    
    # Custom uploader UI
    with st.container():
        st.markdown(
            f"""
            <div class="uploader-container">
                {get_card_icon_html()}
                <h3>Upload SET Game Image</h3>
                <p>Drag and drop or click to select an image of your SET game board</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # The actual streamlit uploader (will be styled via CSS)
        uploaded_file = st.file_uploader(
            "Upload an image of your SET game",
            type=["jpg", "jpeg", "png"],
            key=key,
            label_visibility="collapsed"
        )
    
    return uploaded_file

def render_loading_animation(message="Processing your image..."):
    """Render a custom loading animation."""
    st.markdown(
        f"""
        <div class="loading-container">
            <div class="loader"></div>
            <p class="loading-text">{message}</p>
            <p>Detecting cards, analyzing shapes, colors, and fills...</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_stats_display(set_count, card_count, processing_time):
    """Render statistics about the detection results."""
    st.markdown(
        f"""
        <div class="set-stats">
            <div class="stat-item">
                <div class="stat-value">{set_count}</div>
                <div class="stat-label">Sets Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{card_count}</div>
                <div class="stat-label">Cards Detected</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{processing_time:.1f}s</div>
                <div class="stat-label">Processing Time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_set_cards(sets_info):
    """Render visual representations of the detected sets using the SET card SVGs."""
    if not sets_info or len(sets_info) == 0:
        return
    
    for i, set_info in enumerate(sets_info[:3]):  # Show at most 3 sets
        with st.container():
            st.markdown(f"### Set {i+1}")
            cols = st.columns(3)
            
            for j, card in enumerate(set_info["cards"]):
                with cols[j]:
                    # Create SVG representation of the card
                    card_svg = get_svg_card(
                        color=card["Color"],
                        shape=card["Shape"].lower(),
                        fill=card["Fill"],
                        count=card["Count"]
                    )
                    
                    # Display the SVG
                    st.markdown(card_svg, unsafe_allow_html=True)
                    
                    # Display card properties
                    st.markdown(f"""
                    **Count**: {card["Count"]}  
                    **Color**: {card["Color"].capitalize()}  
                    **Shape**: {card["Shape"].capitalize()}  
                    **Fill**: {card["Fill"].capitalize()}
                    """)
    
    if len(sets_info) > 3:
        st.markdown(f"*...and {len(sets_info) - 3} more sets*")

def get_image_download_link(img, filename="set_detected.png", text="Download Annotated Image"):
    """Generate a download link for the processed image."""
    buffered = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_str = base64.b64encode(cv2.imencode('.png', cv2.cvtColor(np.array(buffered), cv2.COLOR_RGB2BGR))[1]).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="set-button secondary-button">{text}</a>'
    return href

# =============================================================================
# MODEL LOADING AND IMAGE PROCESSING FUNCTIONS
# =============================================================================
base_dir = Path("models")
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Load classification models with caching for better performance."""
    shape_model = load_model(str(characteristics_path / "shape_model.keras"))
    fill_model = load_model(str(characteristics_path / "fill_model.keras"))
    return shape_model, fill_model

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    """Load detection models with caching for better performance."""
    shape_detection_model = YOLO(str(shape_path / "best.pt"))
    shape_detection_model.conf = 0.5
    card_detection_model = YOLO(str(card_path / "best.pt"))
    card_detection_model.conf = 0.5
    if torch.cuda.is_available():
        card_detection_model.to("cuda")
        shape_detection_model.to("cuda")
    return card_detection_model, shape_detection_model

def check_and_rotate_input_image(board_image: np.ndarray, detector: YOLO) -> Tuple[np.ndarray, bool]:
    """Check if image needs rotation and rotate if necessary."""
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
    """Restore the original orientation of an image."""
    if was_rotated:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def predict_color(shape_image: np.ndarray) -> str:
    """Predict the color of a shape in the image."""
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
    """Predict features of a card (count, color, fill, shape)."""
    shape_results = shape_detector(card_image)
    card_h, card_w = card_image.shape[:2]
    card_area = card_w * card_h
    
    # Filter out small detections
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
    
    # Batch predictions for better performance
    fill_preds = fill_model.predict(fill_imgs, batch_size=len(fill_imgs))
    shape_preds = shape_model.predict(shape_imgs, batch_size=len(shape_imgs))
    
    fill_labels_list = ['empty', 'full', 'striped']
    shape_labels_list = ['diamond', 'oval', 'squiggle']
    
    predicted_fill = [fill_labels_list[np.argmax(pred)] for pred in fill_preds]
    predicted_shape = [shape_labels_list[np.argmax(pred)] for pred in shape_preds]
    
    # Get the most common predictions for each feature
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
    """Check if the given cards form a valid SET."""
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        # For each feature, all cards must have the same value or all different values
        feature_values = {card[feature] for card in cards}
        if len(feature_values) != 1 and len(feature_values) != 3:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> List[dict]:
    """Find all valid SETs in the detected cards."""
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

def
