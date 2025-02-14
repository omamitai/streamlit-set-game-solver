"""
Set Game Detector Streamlit App
================================

This app detects valid sets from an uploaded image of a Set game board.
It uses computer vision and machine learning models for card detection
and feature classification, then highlights the detected sets on the image.

Instructions:
    - Place your pre-trained models under the models/ directory as indicated.
    - Run the app with: streamlit run app.py
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

# =============================================================================
#                               CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 📜 Inject Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# =============================================================================
#                              MODEL LOADING
# =============================================================================

# Define base model directory
base_dir = Path("models")

# Define model subdirectories
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

# Load classification models (Keras)
shape_model = load_model(str(characteristics_path / "shape_model.keras"))
fill_model = load_model(str(characteristics_path / "fill_model.keras"))

# Load YOLO detection models
shape_detection_model = YOLO(str(shape_path / "best.pt"))
shape_detection_model.conf = 0.5

card_detection_model = YOLO(str(card_path / "best.pt"))
card_detection_model.conf = 0.5

# Move YOLO models to GPU if available
if torch.cuda.is_available():
    card_detection_model.to("cuda")
    shape_detection_model.to("cuda")

# =============================================================================
#                           STREAMLIT INTERFACE
# =============================================================================

# 🎴 **Title & Description**
st.markdown("<h1 class='title'>🎴 SET Game Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image of a Set game board and detect valid sets!</p>", unsafe_allow_html=True)

# 🔹 **Centered File Upload Section**
st.markdown("<h3 class='upload-title'>📤 Upload Your Image</h3>", unsafe_allow_html=True)
file_container = st.empty()
uploaded_file = file_container.file_uploader(
    "Drag & Drop or Browse Files",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Ensure styling remains even after file is uploaded
if uploaded_file:
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploader"] {
            border: 2px dashed #6A0DAD;
            padding: 15px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.2);
            text-align: center;
            width: 50%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file:
    image = Image.open(uploaded_file)

    # 🔹 **Two-Column Layout for Original & Processed Images**
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("### 📷 Original Photo")
        st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    with col2:
        st.markdown("### 🖼️ Processed Result")
        st.write("Click **Find Sets** to process the image.")

    # 🔘 **Button Row**
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        refresh_clicked = st.button("🔄 Refresh", use_container_width=True)
    
    with col_btn2:
        find_sets_clicked = st.button("🔎 Find Sets", use_container_width=True)

    # Refresh Page on Click
    if refresh_clicked:
        st.experimental_rerun()

    if find_sets_clicked:
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("🔄 Processing... Please wait."):
                sets_info, final_image = classify_and_find_sets_from_array(
                    image_cv,
                    card_detection_model,
                    shape_detection_model,
                    fill_model,
                    shape_model,
                )

            # Convert processed image for display
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(final_image_rgb, caption="✅ Detected Sets", use_column_width=True, output_format="JPEG")

            # 🏆 Success Message
            st.toast("🎉 Sets detected successfully!", icon="✅")

            # 📌 Expandable Results Section
            with st.expander("📜 View Detected Sets Details"):
                st.json(sets_info)

        except Exception as e:
            st.error("⚠️ An error occurred during processing:")
            st.text(traceback.format_exc())
