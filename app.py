"""
SET Game Detector Streamlit App
================================

This app detects valid sets from an uploaded image of a Set game board.
It uses computer vision and machine learning models for card detection
and feature classification, then highlights the detected sets on the image.

The app is designed for both desktop and mobile use, with different interfaces
optimized for each platform.
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
import os
import platform

# =============================================================================
# CONFIGURATION 
# =============================================================================
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="auto"
)

# Define SET theme colors
SET_COLORS = {
    "primary": "#7C3AED",    # Vivid purple
    "secondary": "#10B981",  # Emerald green
    "accent": "#EC4899",     # Pink
    "red": "#EF4444",        # Bright red for SET cards
    "green": "#10B981",      # Emerald green for SET cards
    "purple": "#8B5CF6",     # Royal purple for SET cards
    "background": "#F9FAFB", # Light gray background
    "card": "#FFFFFF",       # White card background
    "text": "#1F2937",       # Dark gray text
}

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
def init_session_state():
    """Initialize all session state variables with default values"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.processed = False
        st.session_state.processing_complete = False
        st.session_state.start_processing = False
        st.session_state.uploaded_file = None
        st.session_state.original_image = None
        st.session_state.processed_image = None
        st.session_state.sets_info = None
        st.session_state.is_mobile = False
        st.session_state.should_reset = False
        st.session_state.no_cards_detected = False
        st.session_state.no_sets_found = False
        st.session_state.uploader_key = "initial"
        st.session_state.image_height = 400
        st.session_state.processing_error = None
        st.session_state.models_loaded = False
        st.session_state.last_reset_time = time.time()
        
init_session_state()

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
        background-color: {SET_COLORS["background"]};
        color: {SET_COLORS["text"]};
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
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, {SET_COLORS["purple"]} 0%, {SET_COLORS["primary"]} 50%, {SET_COLORS["accent"]} 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    .set-header p {{
        font-size: 1.1rem;
        opacity: 0.8;
        text-align: center;
    }}
    
    /* Card styles */
    .set-card {{
        background-color: {SET_COLORS["card"]};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 1.5rem;
    }}
    
    .set-card h3 {{
        margin-top: 0;
        border-bottom: 2px solid {SET_COLORS["primary"]};
        padding-bottom: 0.8rem;
        text-align: center;
        font-weight: 600;
    }}
    
    /* Upload area */
    .upload-area {{
        border: 2px dashed rgba(124, 58, 237, 0.5);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background-color: rgba(124, 58, 237, 0.05);
        cursor: pointer;
        margin-bottom: 1.5rem;
    }}
    
    .upload-area:hover {{
        border-color: {SET_COLORS["primary"]};
        background-color: rgba(124, 58, 237, 0.1);
        transform: translateY(-2px);
    }}
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {{
        background-color: rgba(249, 250, 251, 0.95);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
        padding: 1rem;
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
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    
    .stButton>button:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }}
    
    /* Loading animation */
    .loader-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        gap: 1rem;
    }}
    
    .loader {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .loader-dot {{
        width: 12px;
        height: 12px;
        margin: 0 8px;
        border-radius: 50%;
        display: inline-block;
        animation: loader 1.5s infinite ease-in-out both;
    }}
    
    .loader-dot-1 {{
        background-color: {SET_COLORS["red"]};
        animation-delay: -0.32s;
    }}
    
    .loader-dot-2 {{
        background-color: {SET_COLORS["green"]};
        animation-delay: -0.16s;
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
        margin-top: 0.3rem;
        margin-bottom: 0.5rem;
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        max-height: 40vh;
    }}
    
    .image-container:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.18);
    }}
    
    /* Image caption */
    .image-caption {{
        text-align: center;
        font-size: 1rem;
        color: {SET_COLORS["text"]};
        margin-top: 0.8rem;
        font-weight: 500;
    }}
    
    /* System message */
    .system-message {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }}
    
    /* Image placeholder - improved styles */
    .image-placeholder {{
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.03) 0%, rgba(236, 72, 153, 0.03) 100%);
        border: 2px dashed rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .image-placeholder:hover {{
        border-color: rgba(124, 58, 237, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    }}
    
    .image-placeholder::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.1) 50%,
            rgba(255, 255, 255, 0) 100%
        );
        transform: rotate(30deg);
        animation: shimmer 3s infinite linear;
        z-index: 1;
    }}
    
    @keyframes shimmer {{
        from {{ transform: translateX(-100%) rotate(30deg); }}
        to {{ transform: translateX(100%) rotate(30deg); }}
    }}
    
    .image-placeholder-icon {{
        font-size: 2rem;
        margin-bottom: 0.75rem;
        color: rgba(124, 58, 237, 0.4);
        background: rgba(124, 58, 237, 0.1);
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        z-index: 2;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.15);
    }}
    
    .image-placeholder-text {{
        font-size: 0.9rem;
        color: #6B7280;
        max-width: 80%;
        text-align: center;
        position: relative;
        z-index: 2;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.7);
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }}
    
    .system-message p {{
        font-size: 1.1rem;
        font-weight: 500;
        color: #555;
        margin: 0;
    }}
    
    /* Error messages */
    .error-message {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid {SET_COLORS["red"]};
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }}
    
    .error-message p {{
        margin: 0;
        font-weight: 500;
        color: {SET_COLORS["red"]};
    }}
    
    /* Warning messages */
    .warning-message {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }}
    
    .warning-message p {{
        margin: 0;
        font-weight: 500;
        color: #F59E0B;
    }}
    
    /* Success message */
    .success-message {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid {SET_COLORS["secondary"]};
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }}
    
    .success-message p {{
        margin: 0;
        font-weight: 500;
        color: {SET_COLORS["secondary"]};
    }}
    
    /* Set explanation */
    .set-explanation {{
        background-color: rgba(139, 92, 246, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }}
    
    .set-explanation h4 {{
        color: {SET_COLORS["primary"]};
        margin-top: 0;
        margin-bottom: 0.5rem;
    }}
    
    .set-explanation ul {{
        margin: 0;
        padding-left: 1.2rem;
    }}
    
    /* For mobile devices */
    @media (max-width: 991px) {{
        /* Hide sidebar on mobile */
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        .set-header h1 {{ 
            font-size: 1.8rem; 
        }}
        
        .set-header p {{ 
            font-size: 0.9rem; 
        }}
        
        /* More compact header for mobile */
        .set-header {{
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        
        /* Mobile layout adjustments */
        .mobile-container {{
            padding: 0.8rem;
            margin-bottom: 1rem;
        }}
        
        .stButton>button {{
            padding: 0.8rem 1rem;
        }}
        
        /* Mobile images */
        .image-container img {{
            max-width: 100% !important;
        }}
        
        /* More compact loader for mobile */
        .loader-container {{
            height: 120px;
        }}
        
        /* Adjust system message for mobile */
        .system-message {{
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .system-message p {{
            font-size: 1rem;
        }}
        
        /* Mobile-friendly file uploader */
        .mobile-upload-container {{
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
            padding: 1.2rem;
            border-radius: 12px;
            margin: 0.8rem 0 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            text-align: center;
        }}
        
        /* Center placeholder icon on mobile */
        .image-placeholder-icon {{
            margin: 0 auto 0.75rem auto;
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
    """Load classification models for shape and fill detection"""
    try:
        shape_model = load_model(str(characteristics_path / "shape_model.keras"))
        fill_model = load_model(str(characteristics_path / "fill_model.keras"))
        return shape_model, fill_model
    except Exception as e:
        st.error(f"Error loading classification models: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    """Load YOLOv8 models for card and shape detection"""
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

# Load models once with error handling
def ensure_models_loaded():
    """Ensure all models are loaded successfully"""
    if not st.session_state.get("models_loaded", False):
        try:
            st.session_state.shape_model, st.session_state.fill_model = load_classification_models()
            st.session_state.card_detection_model, st.session_state.shape_detection_model = load_detection_models()
            models_loaded = all([
                st.session_state.shape_model, 
                st.session_state.fill_model, 
                st.session_state.card_detection_model, 
                st.session_state.shape_detection_model
            ])
            st.session_state.models_loaded = models_loaded
            if not models_loaded:
                st.error("Failed to load all required models. Please check your model files.")
                return False
            return True
        except Exception as e:
            st.error(f"Error during model loading: {str(e)}")
            st.session_state.models_loaded = False
            return False
    return st.session_state.models_loaded

# =============================================================================
# MOBILE DETECTION
# =============================================================================
def detect_mobile():
    """
    Enhanced mobile detection with aggressive sidebar hiding
    
    Returns:
        bool: True if device is detected as mobile, False otherwise
    """
    # Improved mobile detection script
    mobile_detector_js = """
    <script>
        (function() {
            // Aggressive sidebar hiding for mobile
            function hideSidebar() {
                const style = document.createElement('style');
                style.innerHTML = `
                    /* Hide sidebar completely */
                    [data-testid="stSidebar"] { 
                        display: none !important;
                        width: 0 !important;
                        height: 0 !important;
                        position: absolute !important;
                        left: -9999px !important;
                        z-index: -1000 !important;
                        opacity: 0 !important;
                        pointer-events: none !important;
                        visibility: hidden !important;
                    }
                    
                    /* Hide sidebar elements */
                    [data-testid="stSidebarNavItems"] {
                        display: none !important;
                    }
                    
                    /* Force main content width */
                    [data-testid="stAppViewContainer"] > section:first-child {
                        min-width: 100% !important;
                        max-width: 100% !important;
                    }
                    
                    .main .block-container {
                        max-width: 100% !important;
                        padding-left: 1rem !important;
                        padding-right: 1rem !important; 
                        padding-top: 1rem !important;
                    }
                `;
                document.head.appendChild(style);
                
                // Also try to find and remove sidebar DOM elements
                setTimeout(() => {
                    const sidebar = document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.style.display = 'none';
                        sidebar.style.width = '0';
                        sidebar.style.height = '0';
                        sidebar.style.position = 'absolute';
                        sidebar.style.left = '-9999px';
                    }
                }, 100);
            }
            
            // Improved mobile detection
            function detectMobile() {
                const isMobile = (
                    // Width-based detection (most reliable)
                    window.innerWidth <= 991 ||
                    // User agent detection
                    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                    // Touch capability detection
                    ('ontouchstart' in window) || (navigator.maxTouchPoints > 0)
                );
                
                // Save to localStorage for persistence
                localStorage.setItem('isMobile', isMobile ? 'true' : 'false');
                
                // Add body class for CSS targeting
                if (isMobile) {
                    document.body.classList.add('mobile-view');
                    hideSidebar();
                }
                
                return isMobile;
            }
            
            // Run immediately
            const isMobile = detectMobile();
            
            // Run on load
            window.addEventListener('load', () => {
                detectMobile();
                if (localStorage.getItem('isMobile') === 'true') {
                    hideSidebar();
                }
            });
            
            // Run on resize with debounce
            let resizeTimer;
            window.addEventListener('resize', () => {
                clearTimeout(resizeTimer);
                resizeTimer = setTimeout(detectMobile, 200);
            });
            
            // Run repeatedly for Streamlit's dynamic loading
            setInterval(() => {
                if (localStorage.getItem('isMobile') === 'true') {
                    hideSidebar();
                }
            }, 500);
        })();
    </script>
    """
    st.markdown(mobile_detector_js, unsafe_allow_html=True)
    
    # Additional CSS specifically for mobile
    mobile_css = """
    <style>
    @media (max-width: 991px) {
        /* Hide hamburger menu button */
        button[kind="header"] {
            display: none !important;
        }
        
        /* Adjust header padding */
        header[data-testid="stHeader"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Ensure content uses full width */
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Improve file uploader appearance */
        [data-testid="stFileUploadDropzone"] {
            min-height: 120px !important;
            padding: 15px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            border: 2px dashed rgba(124, 58, 237, 0.3) !important;
            background-color: rgba(124, 58, 237, 0.05) !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stFileUploadDropzone"]:hover,
        [data-testid="stFileUploadDropzone"]:active {
            border-color: rgba(124, 58, 237, 0.6) !important;
            background-color: rgba(124, 58, 237, 0.08) !important;
        }
    }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)
    
    # Use multiple detection methods for reliability
    platform_string = platform.platform().lower()
    platform_is_mobile = any(mobile_os in platform_string for mobile_os in ['android', 'ios', 'iphone'])
    
    user_agent = os.environ.get('HTTP_USER_AGENT', '').lower()
    ua_is_mobile = any(mobile_kw in user_agent for mobile_kw in ['android', 'iphone', 'ipad', 'mobile'])
    
    session_is_mobile = st.session_state.get("is_mobile", False)
    
    # Combine all detection methods
    is_mobile = platform_is_mobile or ua_is_mobile or session_is_mobile
    
    # Store in session state
    st.session_state.is_mobile = is_mobile
    
    return is_mobile

# =============================================================================
# UTILITY & PROCESSING FUNCTIONS
# =============================================================================
def check_and_rotate_input_image(board_image: np.ndarray, detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Check if image needs rotation based on card orientation
    Returns rotated image and rotation flag
    """
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
    """Restore original image orientation if it was rotated"""
    if was_rotated:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def predict_color(shape_image: np.ndarray) -> str:
    """Predict the color of a card shape using HSV color filtering"""
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for SET colors
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
    purple_mask = cv2.inRange(hsv_image, np.array([120, 50, 50]), np.array([160, 255, 255]))
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Count pixels matching each color
    color_counts = {
        "green": cv2.countNonZero(green_mask),
        "purple": cv2.countNonZero(purple_mask),
        "red": cv2.countNonZero(red_mask)
    }
    
    # Return the color with the most matching pixels
    return max(color_counts, key=color_counts.get)

def predict_card_features(card_image: np.ndarray, shape_detector: YOLO,
                          fill_model: tf.keras.Model, shape_model: tf.keras.Model,
                          box: List[int]) -> Dict:
    """
    Extract and predict all features of a card: count, color, fill, shape
    Returns a dictionary with all card features
    """
    # Detect shapes within the card
    shape_results = shape_detector(card_image)
    card_h, card_w = card_image.shape[:2]
    card_area = card_w * card_h
    
    # Filter boxes to remove small detections (noise)
    filtered_boxes = [
        [int(x1), int(y1), int(x2), int(y2)]
        for x1, y1, x2, y2 in shape_results[0].boxes.xyxy.cpu().numpy()
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area
    ]
    
    # If no shapes detected, return default values
    if not filtered_boxes:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': box}
    
    # Get input shapes for models
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    
    # Process each detected shape
    fill_imgs, shape_imgs, color_list = [], [], []
    for fb in filtered_boxes:
        x1, y1, x2, y2 = fb
        shape_img = card_image[y1:y2, x1:x2]
        
        # Resize images for model input
        fill_img = cv2.resize(shape_img, tuple(fill_input_shape)) / 255.0
        shape_img_resized = cv2.resize(shape_img, tuple(shape_input_shape)) / 255.0
        
        fill_imgs.append(fill_img)
        shape_imgs.append(shape_img_resized)
        color_list.append(predict_color(shape_img))
    
    # Convert to numpy arrays for batch prediction
    fill_imgs = np.array(fill_imgs)
    shape_imgs = np.array(shape_imgs)
    
    # Make predictions
    fill_preds = fill_model.predict(fill_imgs, batch_size=len(fill_imgs))
    shape_preds = shape_model.predict(shape_imgs, batch_size=len(shape_imgs))
    
    # Define labels
    fill_labels_list = ['empty', 'full', 'striped']
    shape_labels_list = ['diamond', 'oval', 'squiggle']
    
    # Get predictions for each shape
    predicted_fill = [fill_labels_list[np.argmax(pred)] for pred in fill_preds]
    predicted_shape = [shape_labels_list[np.argmax(pred)] for pred in shape_preds]
    
    # Find most common prediction (majority vote)
    color_label = max(set(color_list), key=color_list.count)
    fill_label = max(set(predicted_fill), key=predicted_fill.count)
    shape_label = max(set(predicted_shape), key=predicted_shape.count)
    
    # Return card features
    return {
        'count': len(filtered_boxes),
        'color': color_label,
        'fill': fill_label,
        'shape': shape_label,
        'box': box
    }

def is_set(cards: List[dict]) -> bool:
    """
    Check if three cards form a valid SET
    A valid SET requires all four attributes to be either all same or all different
    """
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        feature_values = {card[feature] for card in cards}
        # Must be either all same (1 unique value) or all different (3 unique values)
        if len(feature_values) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> List[dict]:
    """
    Find all valid SETs in the detected cards
    Returns a list of dictionaries containing set information
    """
    sets_found = []
    
    # Check all possible combinations of 3 cards
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
    Detect all cards in the board image
    Returns a list of (card_image, bounding_box) tuples
    """
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    return [
        (board_image[y1:y2, x1:x2], [x1, y1, x2, y2])
        for x1, y1, x2, y2 in card_boxes
    ]

def classify_cards_from_board_image(board_image: np.ndarray, card_detector: YOLO,
                                    shape_detector: YOLO, fill_model: tf.keras.Model,
                                    shape_model: tf.keras.Model) -> pd.DataFrame:
    """
    Detect and classify all cards in the board image
    Returns a DataFrame with card features
    """
    # Detect all cards
    cards = detect_cards_from_image(board_image, card_detector)
    card_data = []
    
    # Process each card
    for card_image, box in cards:
        # Extract features
        features = predict_card_features(card_image, shape_detector, fill_model, shape_model, box)
        
        # Add to card data
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": features['box']
        })
    
    return pd.DataFrame(card_data)

def draw_sets_on_image(board_image: np.ndarray, sets_info: List[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels around SETs in the image
    Returns annotated image
    """
    # Define colors for different SETs
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255)  # Cyan
    ]
    
    # Set visual parameters
    base_thickness = 8
    base_expansion = 5
    
    # Draw each SET with a different color
    for index, set_info in enumerate(sets_info):
        color = colors[index % len(colors)]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        
        # Draw bounding box for each card in the SET
        for i, card in enumerate(set_info["cards"]):
            x1, y1, x2, y2 = card["Coordinates"]
            
            # Expand box slightly for better visibility
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(board_image.shape[1], x2 + expansion)
            y2_exp = min(board_image.shape[0], y2 + expansion)
            
            # Draw rectangle
            cv2.rectangle(board_image, (x1_exp, y1_exp), (x2_exp, y2_exp), color, thickness)
            
            # Add set number label on first card of each set
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
    """
    Main processing function to detect cards, classify features, and find SETs
    Returns (sets_found, annotated_image)
    """
    # Check if rotation is needed
    processed_image, was_rotated = check_and_rotate_input_image(board_image, card_detector)
    
    # Detect cards
    cards = detect_cards_from_image(processed_image, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], processed_image
    
    # Classify cards and find SETs
    card_df = classify_cards_from_board_image(processed_image, card_detector, shape_detector, fill_model, shape_model)
    sets_found = find_sets(card_df)
    
    # Check if SETs are found
    if not sets_found:
        st.session_state.no_sets_found = True
        return [], processed_image
    
    # Draw SETs on image
    annotated_image = draw_sets_on_image(processed_image.copy(), sets_found)
    
    # Restore original orientation if rotated
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

def reset_session_state():
    """Reset all session state variables to their initial values"""
    # Preserve the mobile detection flag
    is_mobile = st.session_state.get("is_mobile", False)
    models_loaded = st.session_state.get("models_loaded", False)
    
    # Generate a new uploader key to force reset
    uploader_key = f"uploader_{random.randint(1000, 9999)}"
    
    # Reset all session state
    for key in list(st.session_state.keys()):
        if key not in ["is_mobile", "models_loaded"]:
            if key in st.session_state:
                del st.session_state[key]
    
    # Reinitialize with clean defaults
    st.session_state.processed = False
    st.session_state.processing_complete = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.is_mobile = is_mobile  # Preserve mobile detection
    st.session_state.models_loaded = models_loaded  # Preserve model loading status
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    st.session_state.uploader_key = uploader_key
    st.session_state.image_height = 400
    st.session_state.last_reset_time = time.time()
    
    # Clear URL params
    st.query_params.clear()

# =============================================================================
# UI COMPONENTS
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

def render_loader():
    """Render a SET-themed loader without text"""
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

def render_success_message(message):
    """Render a styled success message (kept for potential future use)"""
    success_html = f"""
    <div class="success-message">
        <p>{message}</p>
    </div>
    """
    return st.markdown(success_html, unsafe_allow_html=True)

def render_process_message():
    """Render a styled system message to prompt user to process the image"""
    message_html = """
    <div class="system-message">
        <p>Click "Find Sets" to process the image</p>
    </div>
    """
    return st.markdown(message_html, unsafe_allow_html=True)

def render_image_placeholder(message="Your processed image will appear here"):
    """Render a placeholder for where the image will appear"""
    placeholder_html = f"""
    <div class="image-placeholder">
        <div class="image-placeholder-icon">🎴</div>
        <div class="image-placeholder-text">{message}</div>
    </div>
    """
    return st.markdown(placeholder_html, unsafe_allow_html=True)

# =============================================================================
# IMAGE PROCESSING
# =============================================================================
def process_image():
    """Process the uploaded image to find SETs"""
    try:
        # Check if models are loaded
        if not ensure_models_loaded():
            st.session_state.processing_error = "Failed to load models. Please check your installation."
            st.session_state.start_processing = False
            return
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
        
        # Process image with models
        sets_info, processed_image = classify_and_find_sets_from_array(
            image_cv, 
            st.session_state.card_detection_model, 
            st.session_state.shape_detection_model, 
            st.session_state.fill_model, 
            st.session_state.shape_model
        )
        
        # Update session state
        st.session_state.processed_image = processed_image
        st.session_state.sets_info = sets_info
        st.session_state.processed = True
        st.session_state.processing_complete = True
        st.session_state.start_processing = False
        
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.session_state.start_processing = False
        st.session_state.processing_complete = True
        # Log the full traceback for debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in process_image: {traceback_str}")

# =============================================================================
# MOBILE INTERFACE
# =============================================================================
def render_mobile_interface():
    """Render the mobile-optimized interface"""
    
    # Only show uploader if no file is uploaded yet
    if not st.session_state.get("uploaded_file", None):
        # Custom mobile-friendly uploader styling
        mobile_upload_css = """
        <style>
        /* Custom mobile uploader */
        [data-testid="stFileUploader"] {
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stFileUploadDropzone"] {
            min-height: 100px !important;
            padding: 10px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            border: 2px dashed rgba(124, 58, 237, 0.3) !important;
            background-color: rgba(124, 58, 237, 0.02) !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: rgba(124, 58, 237, 0.5) !important;
            background-color: rgba(124, 58, 237, 0.05) !important;
        }
        
        [data-testid="stFileUploadDropzone"] p {
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        [data-testid="stFileUploadDropzone"] svg {
            height: 28px !important;
            width: 28px !important;
            color: rgba(124, 58, 237, 0.6) !important;
        }
        
        /* Make the whole upload area more prominent */
        .mobile-upload-container {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
            border-radius: 12px;
            padding: 0.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        </style>
        """
        st.markdown(mobile_upload_css, unsafe_allow_html=True)
        
        st.markdown('<div class="mobile-upload-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; margin-top: 0; padding-top: 0;">Take a Photo of Your SET Game</h3>', unsafe_allow_html=True)
        
        # Custom instruction for mobile users
        st.markdown("""
        <div style="text-align: center; margin-bottom: 0.5rem; font-size: 0.9rem; color: #666;">
            Tap below to take a photo or upload from your gallery
        </div>
        """, unsafe_allow_html=True)
        
        # Mobile file uploader
        mobile_uploaded_file = st.file_uploader(
            label="Upload SET image (mobile)",
            type=["png", "jpg", "jpeg"],
            key=f"mobile_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            help="Upload a photo of your SET game board"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show placeholder when no image is uploaded yet
        render_image_placeholder("Your SET game will appear here")
        
        # Handle new file upload
        if mobile_uploaded_file is not None and mobile_uploaded_file != st.session_state.get("uploaded_file", None):
            # Reset state for new file
            st.session_state.processed = False
            st.session_state.processing_complete = False
            st.session_state.processed_image = None
            st.session_state.sets_info = None
            st.session_state.no_cards_detected = False
            st.session_state.no_sets_found = False
            st.session_state.processing_error = None
            
            # Set the new file
            st.session_state.uploaded_file = mobile_uploaded_file
            
            # Process the new image
            try:
                image = Image.open(mobile_uploaded_file)
                image = optimize_image(image)
                st.session_state.original_image = image
                st.session_state.image_height = image.height
                
                # Automatic processing on mobile with slight delay for UI responsiveness
                st.session_state.start_processing = True
                
                # Skip success message as requested
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load image: {str(e)}")
    
    # Show original image if available
    if st.session_state.get("original_image", None) is not None:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(
            st.session_state.original_image, 
            caption="Original Image", 
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing state
        if st.session_state.get("start_processing", False):
            # Show loading animation
            render_loader()
            
            # Process image
            process_image()
            
            # Force refresh after processing
            st.rerun()
        
        # Show processing error if any
        elif st.session_state.get("processing_error", None):
            render_error_message(f"Error during processing: {st.session_state.processing_error}")
            
            # Reset button after error
            if st.button("⟳ Try Again", key="mobile_error_reset"):
                reset_session_state()
                st.rerun()
        
        # Show results
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                render_error_message("No cards detected in this image. Are you sure this is a SET game board?")
            elif st.session_state.get("no_sets_found", False):
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(
                    processed_img, 
                    caption="Cards Detected (No SETs Found)", 
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                render_warning_message("Cards were detected but no valid SETs were found. The dealer might need to add more cards!")
            elif st.session_state.get("processed_image", None) is not None:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(
                    processed_img, 
                    caption=f"Found {len(st.session_state.sets_info)} SET{'s' if len(st.session_state.sets_info) != 1 else ''}", 
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("⟳ Analyze New Image", key="mobile_reset"):
                reset_session_state()
                st.rerun()
        
        # Show process button if needed
        elif not st.session_state.get("processed", False) and not st.session_state.get("processing_complete", False):
            # Show placeholder for the processed image
            render_image_placeholder("Processed image will appear here")
            
            # Show processing message
            render_process_message()
            
            if st.button("🔎 Find Sets", key="mobile_process"):
                st.session_state.start_processing = True
                st.rerun()

# =============================================================================
# DESKTOP INTERFACE
# =============================================================================
def render_desktop_interface():
    """Render the desktop-optimized interface"""
    
    # Setup the sidebar for desktop upload
    with st.sidebar:
        st.markdown('<h3 style="text-align: center;">Upload Your Image</h3>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            label="Upload SET image",
            type=["png", "jpg", "jpeg"],
            key=f"desktop_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            help="Upload a photo of your SET game board"
        )
        
        # Handle new file upload
        if uploaded_file is not None and uploaded_file != st.session_state.get("uploaded_file", None):
            # Reset state for new file
            st.session_state.processed = False
            st.session_state.processing_complete = False
            st.session_state.processed_image = None
            st.session_state.sets_info = None
            st.session_state.no_cards_detected = False
            st.session_state.no_sets_found = False
            st.session_state.processing_error = None
            
            # Set the new file
            st.session_state.uploaded_file = uploaded_file
            
            # Process the new image
            try:
                image = Image.open(uploaded_file)
                image = optimize_image(image)
                st.session_state.original_image = image
                st.session_state.image_height = image.height
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load image: {str(e)}")
        
        # Process button
        if uploaded_file is not None and not st.session_state.get("processed", False):
            if st.button("🔎 Find Sets", key="desktop_process"):
                st.session_state.start_processing = True
                st.rerun()
    
    # Main content area with two columns
    col1, col2 = st.columns([1, 1])
    
    # Original image column
    with col1:
        if st.session_state.get("original_image", None) is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(
                st.session_state.original_image, 
                caption="Original Image", 
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show placeholder for original image
            render_image_placeholder("Upload an image to get started")
    
    # Results column
    with col2:
        if st.session_state.get("start_processing", False):
            # Show loading animation without text
            render_loader()
            
            # Process image
            process_image()
            
            # Force refresh after processing
            st.rerun()
        elif not st.session_state.get("processed", False) and not st.session_state.get("original_image", None):
            # Show placeholder for results when no image is uploaded
            render_image_placeholder("Results will appear here")
        
        # Show processing error if any
        elif st.session_state.get("processing_error", None):
            render_error_message(f"Error during processing: {st.session_state.processing_error}")
            
            # Detailed error for debugging
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
            
            # Reset button after error
            if st.button("⟳ Try Again", key="desktop_error_reset"):
                reset_session_state()
                st.rerun()
        
        # Show results
        elif st.session_state.get("processed", False):
            if st.session_state.get("no_cards_detected", False):
                # Show the original image with error
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(
                    st.session_state.original_image, 
                    caption="Original Image", 
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show error message
                render_error_message("No cards detected in this image. Are you sure this is a SET game board?")
                
            elif st.session_state.get("no_sets_found", False):
                # Show the processed image without sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(
                    processed_img, 
                    caption="Cards Detected (No SETs Found)", 
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show warning message
                render_warning_message("Cards were detected but no valid SETs were found. The dealer might need to add more cards!")
                
            elif st.session_state.get("processed_image", None) is not None:
                # Show the processed image with detected sets
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                processed_img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(
                    processed_img, 
                    caption=f"Found {len(st.session_state.sets_info)} SET{'s' if len(st.session_state.sets_info) != 1 else ''}", 
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("⟳ Analyze New Image", key="desktop_reset"):
                reset_session_state()
                st.rerun()
        
        # Show message to process
        elif st.session_state.get("original_image", None) is not None and not st.session_state.get("processed", False):
            # Show placeholder for the processed image
            render_image_placeholder("Processed image will appear here")
            
            # Show processing message
            render_process_message()

# =============================================================================
# MAIN APP LAYOUT & LOGIC
# =============================================================================
def main():
    """Main application entry point"""
    # Load CSS styling
    load_css()
    
    # Detect mobile devices
    is_mobile = detect_mobile()
    
    # Handle potential reset
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Render header
    render_header()
    
    # Render appropriate interface based on device type
    if is_mobile:
        render_mobile_interface()
    else:
        render_desktop_interface()

if __name__ == "__main__":
    main()
