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
#       INITIALIZE SESSION STATE
# =============================================================================

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.is_mobile = False

# =============================================================================
#       CONFIGURATION & STYLE LOADING
# =============================================================================

st.set_page_config(layout="wide", page_title="SET Game Detector")

# Detect if user is on mobile
is_mobile = st.session_state.is_mobile

# Load correct CSS file
css_file = "mobile.css" if is_mobile else "desktop.css"

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Could not load custom CSS.")

load_css(css_file)

# =============================================================================
#       HEADER SECTION
# =============================================================================

st.markdown(
    """
    <div class="header-container">
        <h1>🎴 SET Game Detector</h1>
        <p class="subtitle">Upload an image of a Set game board and detect valid sets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
#       FILE UPLOADER & DEVICE DETECTION
# =============================================================================

if is_mobile:
    uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"], key="main_uploader")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your image", type=["png", "jpg", "jpeg"], key="sidebar_uploader")

# Save uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.is_mobile = is_mobile
    try:
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.processed_image = None
        st.session_state.sets_info = None
    except Exception as e:
        st.error("Failed to load image. Please try another file.")

# Desktop: Show "Find Sets" button
if not is_mobile:
    st.sidebar.info("After uploading, click **Find Sets** to start processing.")
    find_sets_clicked = st.sidebar.button("🔎 Find Sets", key="find_sets")
else:
    find_sets_clicked = True  # Mobile auto-processes

# =============================================================================
#       IMAGE PROCESSING & DISPLAY
# =============================================================================

if st.session_state.uploaded_file is None:
    st.info("Please upload an image.")
else:
    original_image = st.session_state.original_image.copy()

    if find_sets_clicked:
        # Show loader while processing
        loader_placeholder = st.empty()
        loader_placeholder.markdown('<div class="center-loader">Detecting sets...</div>', unsafe_allow_html=True)

        # Simulated processing (replace with actual model calls)
        processed_image = original_image.copy()  # Replace with detection logic
        st.session_state.processed_image = processed_image

        # Display feedback
        st.success("Sets detected!" if processed_image else "No sets found.")
        loader_placeholder.empty()

    # Layout based on device type
    if is_mobile:
        # Mobile Layout (Stacked)
        st.subheader("Original Image")
        st.image(original_image, width=400, output_format="JPEG")
        st.markdown('<div class="mobile-arrow">⬇️</div>', unsafe_allow_html=True)
        st.subheader("Detected Sets")
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, width=400, output_format="JPEG")
        else:
            st.info("Processed image will appear here after detection.")
    else:
        # Desktop Layout (Side by Side)
        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            st.subheader("Original Image")
            st.image(original_image, width=400, output_format="JPEG")
        with col2:
            st.markdown("<div class='desktop-arrow'>➡️</div>", unsafe_allow_html=True)
        with col3:
            st.subheader("Detected Sets")
            if st.session_state.processed_image:
                st.image(st.session_state.processed_image, width=400, output_format="JPEG")
            else:
                st.info("Processed image will appear here after detection.")
