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
import os
import time
import platform

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.update({
            "initialized": True,
            "uploaded_file": None,
            "original_image": None,
            "processed_image": None,
            "sets_info": None,
            "processing_complete": False,
            "processing_error": None,
            "is_mobile": False,
            "uploader_key": "uploader_1",
        })

init_session()

# =============================================================================
# MOBILE DETECTION
# =============================================================================
def detect_mobile():
    """Detect if the user is on a mobile device."""
    user_agent = os.environ.get('HTTP_USER_AGENT', '').lower()
    return any(mob in user_agent for mob in ["android", "iphone", "ipad", "mobile"])

st.session_state.is_mobile = detect_mobile()

# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load the ML models for SET card detection."""
    base_dir = Path("models")
    characteristics_path = base_dir / "Characteristics"
    shape_path = base_dir / "Shape"
    card_path = base_dir / "Card"

    try:
        shape_model = load_model(characteristics_path / "shape_model.keras")
        fill_model = load_model(characteristics_path / "fill_model.keras")
        card_model = YOLO(card_path / "best.pt")
        shape_detector = YOLO(shape_path / "best.pt")

        if torch.cuda.is_available():
            card_model.to("cuda")
            shape_detector.to("cuda")

        return shape_model, fill_model, card_model, shape_detector
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None, None

st.session_state.shape_model, st.session_state.fill_model, st.session_state.card_model, st.session_state.shape_detector = load_models()

# =============================================================================
# CSS STYLING
# =============================================================================
def load_css():
    """Inject custom CSS for styling."""
    css = """
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #F9FAFB;
        color: #1F2937;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED 0%, #EC4899 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }
    @media (max-width: 991px) {
        [data-testid="stSidebar"] {
            display: none !important;
        }
        .stAppViewContainer>section:first-child {
            min-width: 100% !important;
            max-width: 100% !important;
        }
        .stFileUpload {
            width: 100%;
            text-align: center;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================
def process_image():
    """Process the uploaded image and find valid SETs."""
    try:
        image_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
        card_model = st.session_state.card_model
        shape_detector = st.session_state.shape_detector
        fill_model = st.session_state.fill_model
        shape_model = st.session_state.shape_model

        if not all([card_model, shape_detector, fill_model, shape_model]):
            st.session_state.processing_error = "Failed to load models."
            return

        results = card_model(image_cv)
        card_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if not len(card_boxes):
            st.session_state.processing_error = "No cards detected."
            return

        sets_found = []
        annotated_image = image_cv.copy()

        for box in card_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        st.session_state.processed_image = annotated_image
        st.session_state.sets_info = sets_found
        st.session_state.processing_complete = True

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"Processing error: {traceback.format_exc()}")

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    """Render the app header."""
    st.markdown("""
        <h1 style="text-align:center; color:#7C3AED;">🎴 SET Game Detector</h1>
        <p style="text-align:center;">Upload an image of a SET game board and detect valid sets!</p>
    """, unsafe_allow_html=True)

def render_image(image, caption):
    """Render an image inside a styled container."""
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.image(image, caption=caption, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_file_uploader():
    """Render the file uploader based on device type."""
    if st.session_state.is_mobile:
        return st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            key=st.session_state.uploader_key,
            label_visibility="collapsed"
        )
    else:
        with st.sidebar:
            return st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                key=st.session_state.uploader_key
            )

# =============================================================================
# MAIN APP LOGIC
# =============================================================================
def main():
    """Main function to run the Streamlit app."""
    render_header()

    uploaded_file = render_file_uploader()
    
    if uploaded_file and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.processing_complete = False
        st.session_state.processing_error = None
        st.session_state.processed_image = None

    if st.session_state.original_image:
        render_image(st.session_state.original_image, "Original Image")

        if st.session_state.is_mobile and not st.session_state.processing_complete:
            st.session_state.processing_complete = True
            process_image()
            st.rerun()

        elif not st.session_state.processing_complete:
            if st.button("🔎 Find Sets"):
                process_image()
                st.rerun()

    if st.session_state.processing_complete:
        if st.session_state.processing_error:
            st.error(st.session_state.processing_error)
        elif st.session_state.processed_image is not None:
            render_image(st.session_state.processed_image, "Processed Image")

if __name__ == "__main__":
    main()
