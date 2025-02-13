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

# --------------------------------------------------------------------------
# 🎨 Streamlit Page Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 📜 Inject Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# --------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------
base_dir = Path("models")
characteristics_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

shape_model = load_model(str(characteristics_path / "shape_model.keras"))
fill_model = load_model(str(characteristics_path / "fill_model.keras"))

shape_detection_model = YOLO(str(shape_path / "best.pt"))
shape_detection_model.yaml = str(shape_path / "data.yaml")
shape_detection_model.conf = 0.5

card_detection_model = YOLO(str(card_path / "best.pt"))
card_detection_model.yaml = str(card_path / "data.yaml")
card_detection_model.conf = 0.5

if torch.cuda.is_available():
    card_detection_model.to('cuda')
    shape_detection_model.to('cuda')

# --------------------------------------------------------------------------
# Utility & Processing Functions
# (functions remain unchanged; these handle image processing and set detection)
# --------------------------------------------------------------------------
# ... (your utility functions remain unchanged) ...

# --------------------------------------------------------------------------
# 🎲 Streamlit App UI
# --------------------------------------------------------------------------

# 🌟 Title & Description (centered and themed)
st.markdown("<h1 style='text-align:center; color:#FFD700;'>🎴 SET Game Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image of a SET game board and click <b>🔎 Find Sets</b> to detect valid sets.</p>", unsafe_allow_html=True)

# 🔹 Layout: Two Equal Columns with tight spacing
col1, col2 = st.columns([1, 1], gap="small")

# 📥 Left Column: Upload & Show Original Image
with col1:
    st.markdown("### 📥 Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🎴 Original Image", use_column_width=True, output_format="JPEG")

# 🔍 Right Column: Process & Show Results
with col2:
    st.markdown("### 🔍 Processed Result")
    if uploaded_file:
        # Align the action button inline with the description
        if st.button("🔎 Find Sets", use_container_width=True):
            try:
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                with st.spinner("🔄 Processing... Please wait."):
                    sets_info, final_image = classify_and_find_sets_from_array(
                        image_cv,
                        card_detection_model,
                        shape_detection_model,
                        fill_model,
                        shape_model
                    )
                final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                st.image(final_image_rgb, caption="✅ Detected Sets", use_column_width=True, output_format="JPEG")
                st.toast("🎉 Sets detected successfully!", icon="✅")
                with st.expander("📜 View Detected Sets Details"):
                    st.json(sets_info)
            except Exception as e:
                st.error("⚠️ An error occurred during processing:")
                st.text(traceback.format_exc())
