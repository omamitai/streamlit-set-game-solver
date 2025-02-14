import streamlit as st
from PIL import Image
import numpy as np
import cv2
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple, List, Dict

# =============================================================================
#                           CONFIGURATION & STYLE
# =============================================================================

st.set_page_config(layout="wide", page_title="SET Game Detector")

# Inject CSS styles
with open("styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main header (compact & modern)
st.markdown(
    """
    <div class="main-header">
        <h1>🎴 SET Game Detector</h1>
        <p>Upload an image of a Set game board from the sidebar and click "Find Sets"</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
#                              MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[YOLO, YOLO, tf.keras.Model, tf.keras.Model]:
    base_dir = Path("models")
    shape_model = load_model(base_dir / "Characteristics" / "11022025" / "shape_model.keras")
    fill_model = load_model(base_dir / "Characteristics" / "11022025" / "fill_model.keras")
    card_detector = YOLO(base_dir / "Card" / "16042024" / "best.pt")
    shape_detector = YOLO(base_dir / "Shape" / "15052024" / "best.pt")

    if torch.cuda.is_available():
        card_detector.to("cuda")
        shape_detector.to("cuda")

    return card_detector, shape_detector, shape_model, fill_model

card_detector, shape_detector, shape_model, fill_model = load_models()

# =============================================================================
#                           SIDEBAR: File Upload
# =============================================================================

st.sidebar.markdown('<div class="sidebar-header">📤 Upload Your Image</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

# =============================================================================
#                           MAIN INTERFACE: Button & Image Display
# =============================================================================

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        max_width = 450  # Adjusted for compact display
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)

        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔎 Find Sets"):
                try:
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    with st.spinner("⏳ Processing... Please wait."):
                        # Placeholder function (Assumes existing detection logic)
                        processed_image = image_cv  # Replace with actual processing function
                    st.session_state.processed_image = processed_image
                except Exception as e:
                    st.error("⚠️ An error occurred:")
                    st.text(traceback.format_exc())

        left_col, mid_col, right_col = st.columns([3, 1, 3])
        with left_col:
            st.image(image, use_container_width=True)
        with mid_col:
            st.markdown("<div class='arrow'>➡️</div>", unsafe_allow_html=True)
        with right_col:
            if "processed_image" in st.session_state:
                processed_image_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_image_rgb, use_container_width=True)
            else:
                st.info("Processed image will appear here after clicking 'Find Sets'.")
    except Exception as e:
        st.error("Failed to load image. Please try again.")
        st.exception(e)
else:
    st.info("Please upload an image to begin.")
