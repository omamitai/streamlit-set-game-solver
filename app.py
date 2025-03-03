import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from itertools import combinations
from pathlib import Path
import random

# Streamlit Configuration
st.set_page_config(
    page_title="SET Detector",
    page_icon="🃏",
    layout="wide"
)

# SET Theme Colors
SET_THEME = {
    "primary": "#7C3AED",     # Purple
    "secondary": "#10B981",   # Green
    "accent": "#EC4899",      # Pink
    "red": "#EF4444",         # Red
    "green": "#10B981",       # Green
    "purple": "#8B5CF6",      # Light Purple
    "background": "#F4F1FA",  # Light Purple Background
    "card": "#FFFFFF",        # White
    "text": "#1F2937",        # Dark Text
    "text_muted": "#6B7280",  # Gray Text
}

# Session State Initialization
if "app_view" not in st.session_state:
    st.session_state.app_view = "upload"
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "initial"

# Custom CSS
def load_custom_css():
    css = """
    <style>
    :root {
        --set-purple: #7C3AED;
        --set-green: #10B981;
        --set-red: #EF4444;
        --set-background: #F4F1FA;
        --set-card: #FFFFFF;
        --set-text: #1F2937;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--set-background);
        color: var(--set-text);
    }
    .main .block-container {
        padding: 0.5rem;
        max-width: 100%;
    }
    .ios-header {
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(236, 72, 153, 0.15));
        backdrop-filter: blur(12px);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.2);
        text-align: center;
    }
    .ios-header h1 {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple), #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ios-image-container {
        margin: 0.4rem auto;
        border-radius: 16px;
        max-height: 220px;
        width: 90%;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(236, 72, 153, 0.08));
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25);
    }
    .ios-button-primary > button {
        background: linear-gradient(135deg, var(--set-purple), #8B5CF6) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.4rem !important;
        min-height: 44px !important;
        box-shadow: 0 3px 12px rgba(124, 58, 237, 0.4) !important;
        width: 100% !important;
    }
    .ios-button-secondary > button {
        background: linear-gradient(135deg, var(--set-red), #F87171) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.4rem !important;
        min-height: 44px !important;
        box-shadow: 0 3px 12px rgba(239, 68, 68, 0.4) !important;
        width: 100% !important;
    }
    .find-sets-button > button {
        background: linear-gradient(135deg, var(--set-green), #34D399) !important;
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.4) !important;
    }
    .ios-loader-container {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(244, 241, 250, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 16px;
    }
    .ios-loader {
        width: 40px;
        height: 40px;
        border: 3px solid rgba(124, 58, 237, 0.2);
        border-top: 3px solid var(--set-purple);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-bottom: 0.5rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .ios-loader-text {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--set-purple);
        background: rgba(255, 255, 255, 0.95);
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }
    .ios-alert {
        padding: 0.5rem 0.8rem;
        border-radius: 14px;
        margin: 0.4rem auto;
        font-size: 0.85rem;
        font-weight: 600;
        width: 90%;
        text-align: center;
    }
    .ios-alert-error {
        background-color: rgba(239, 68, 68, 0.12);
        color: var(--set-red);
    }
    .ios-alert-warning {
        background-color: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
    }
    .ios-instruction {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--set-text-muted);
        margin: 0.4rem auto;
        max-width: 90%;
        background: rgba(255, 255, 255, 0.6);
        padding: 0.5rem 0.8rem;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(124, 58, 237, 0.12);
    }
    .ios-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, var(--set-purple), #8B5CF6);
        color: white;
        border-radius: 16px;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0.3rem auto;
        box-shadow: 0 3px 12px rgba(124, 58, 237, 0.4);
    }
    footer, header { display: none !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Model Paths & Loading
base_dir = Path("models")
char_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024" 
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models():
    """
    Loads the Keras models for 'shape' and 'fill' classification from disk.
    Returns (shape_model, fill_model).
    """
    try:
        model_shape = load_model(str(char_path / "shape_model.keras"))
        model_fill = load_model(str(char_path / "fill_model.keras"))
        return model_shape, model_fill
    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models():
    """
    Loads the YOLO detection models for cards and shapes from disk.
    Returns (card_detector, shape_detector).
    """
    try:
        detector_shape = YOLO(str(shape_path / "best.pt"))
        detector_shape.conf = 0.5
        detector_card = YOLO(str(card_path / "best.pt"))
        detector_card.conf = 0.5
        if torch.cuda.is_available():
            detector_card.to("cuda")
            detector_shape.to("cuda")
        return detector_card, detector_shape
    except Exception as e:
        st.error(f"Error loading detection models: {e}")
        return None, None

# Attempt to load all models
try:
    model_shape, model_fill = load_classification_models()
    detector_card, detector_shape = load_detection_models()
    models_loaded = all([model_shape, model_fill, detector_card, detector_shape])
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Utility Functions
def optimize_image_size(img_pil, max_dim=450):
    width, height = img_pil.size
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        return img_pil.resize((new_width, new_height), Image.LANCZOS)
    return img_pil

def verify_and_rotate_image(board_image, card_detector):
    """
    Checks if the detected cards are oriented primarily vertically or horizontally.
    If they're vertical, rotates the board_image 90 degrees clockwise for consistent processing.
    """
    detection = card_detector(board_image)
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0:
        return board_image, False

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Rotate if average height > average width
    if np.mean(heights) > np.mean(widths):
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False

def restore_orientation(img, was_rotated):
    """
    Restores original orientation if the image was previously rotated.
    """
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr):
    """
    Rough color classification using HSV thresholds to differentiate 'red', 'green', 'purple'.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))

    # Red can wrap around hue=0, so we combine both ends
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    counts = {
        "green": cv2.countNonZero(mask_green),
        "purple": cv2.countNonZero(mask_purple),
        "red": cv2.countNonZero(mask_red),
    }
    return max(counts, key=counts.get)

def detect_cards(board_img, card_detector):
    """
    Runs YOLO on the board_img to detect card bounding boxes.
    Returns a list of (card_image, [x1, y1, x2, y2]) for each detected card.
    """
    result = card_detector(board_img)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    detected_cards = []

    for x1, y1, x2, y2 in boxes:
        detected_cards.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2]))
    return detected_cards

def predict_card_features(card_img, shape_detector, fill_model, shape_model, card_box):
    """
    Predicts the 'count', 'color', 'fill', 'shape' features for a single card.
    It uses a shape_detector YOLO model to locate shapes, then passes them to fill_model and shape_model.
    """
    # Detect shapes on the card
    shape_detections = shape_detector(card_img)
    c_h, c_w = card_img.shape[:2]
    card_area = c_w * c_h

    # Filter out spurious shape detections
    shape_boxes = []
    for coords in shape_detections[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = coords.astype(int)
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area:
            shape_boxes.append([x1, y1, x2, y2])

    if not shape_boxes:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': card_box
        }

    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs = []
    shape_imgs = []
    color_candidates = []

    # Prepare each detected shape region for classification
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        shape_crop = card_img[sy1:sy2, sx1:sx2]

        fill_crop = cv2.resize(shape_crop, fill_input_size) / 255.0
        shape_crop_resized = cv2.resize(shape_crop, shape_input_size) / 255.0

        fill_imgs.append(fill_crop)
        shape_imgs.append(shape_crop_resized)
        color_candidates.append(predict_color(shape_crop))

    fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=len(fill_imgs), verbose=0)
    shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=len(shape_imgs), verbose=0)

    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']

    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]

    # Take the most common color/fill/shape across all shape detections for the card
    final_color = max(set(color_candidates), key=color_candidates.count)
    final_fill = max(set(fill_result), key=fill_result.count)
    final_shape = max(set(shape_result), key=shape_result.count)

    return {
        'count': len(shape_boxes),
        'color': final_color,
        'fill': final_fill,
        'shape': final_shape,
        'box': card_box
    }

def classify_cards_on_board(board_img, card_detector, shape_detector, fill_model, shape_model):
    """
    Detects cards on the board, then classifies each card's features.
    Returns a DataFrame with columns: 'Count', 'Color', 'Fill', 'Shape', 'Coordinates'.
    """
    detected_cards = detect_cards(board_img, card_detector)
    card_rows = []

    for (card_img, box) in detected_cards:
        card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
        card_rows.append({
            "Count": card_feats['count'],
            "Color": card_feats['color'],
            "Fill": card_feats['fill'],
            "Shape": card_feats['shape'],
            "Coordinates": card_feats['box']
        })

    return pd.DataFrame(card_rows)

def valid_set(cards):
    """
    Checks if the given 3 cards collectively form a valid SET.
    """
    for feature in ["Count", "Color", "Fill", "Shape"]:
        if len({card[feature] for card in cards}) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df):
    """
    Finds all possible SETs from the card DataFrame.
    Each SET is a dictionary with 'set_indices' and 'cards' fields.
    """
    found_sets = []
    for combo in combinations(cards_df.iterrows(), 3):
        cards = [c[1] for c in combo]  # c is (index, row)
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [
                    {f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']}
                    for card in cards
                ]
            })
    return found_sets

def draw_detected_sets(board_img, sets_detected):
    """
    Annotates the board image with visually appealing bounding boxes for each detected SET.
    Each SET is drawn in a different color with enhanced iOS-style indicators.
    """
    # Enhanced SET-themed colors (BGR format)
    set_colors = [
        (68, 68, 239),    # Red
        (89, 199, 16),    # Green
        (246, 92, 139),   # Pink
        (0, 149, 255),    # Orange
        (222, 82, 175),   # Purple
        (85, 45, 255)     # Hot Pink
    ]
    
    # Enhanced styling parameters
    base_thickness = 4
    base_expansion = 5

    # Create a copy to avoid modifying the original
    result_img = board_img.copy()

    for idx, single_set in enumerate(sets_detected):
        color = set_colors[idx % len(set_colors)]
        thickness = base_thickness + (idx % 3)
        expansion = base_expansion + 10 * (idx % 3)

        for i, card_info in enumerate(single_set["cards"]):
            x1, y1, x2, y2 = card_info["Coordinates"]
            
            # Expand the bounding box slightly
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(result_img.shape[1], x2 + expansion)
            y2e = min(result_img.shape[0], y2 + expansion)

            # Draw premium rounded rectangle with gradient effect
            cv2.rectangle(result_img, (x1e, y1e), (x2e, y2e), color, thickness, cv2.LINE_AA)
            
            # Add iOS-style rounded corners with thicker lines
            corner_length = min(30, (x2e - x1e) // 4)
            corner_thickness = thickness + 1
            
            # Draw all four corners
            # Top-left corner
            cv2.line(result_img, (x1e, y1e + corner_length), (x1e, y1e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y1e), (x1e + corner_length, y1e), color, corner_thickness, cv2.LINE_AA)
            
            # Top-right corner
            cv2.line(result_img, (x2e - corner_length, y1e), (x2e, y1e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y1e), (x2e, y1e + corner_length), color, corner_thickness, cv2.LINE_AA)
            
            # Bottom-left corner
            cv2.line(result_img, (x1e, y2e - corner_length), (x1e, y2e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y2e), (x1e + corner_length, y2e), color, corner_thickness, cv2.LINE_AA)
            
            # Bottom-right corner
            cv2.line(result_img, (x2e - corner_length, y2e), (x2e, y2e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y2e), (x2e, y2e - corner_length), color, corner_thickness, cv2.LINE_AA)

            # Label the first card with Set number in an iOS-style pill badge
            if i == 0:
                text = f"Set {idx+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                badge_width = text_size[0] + 16
                badge_height = 26
                badge_x = x1e
                badge_y = y1e - badge_height - 5
                
                # Draw pill-shaped badge with anti-aliasing
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              color, -1, cv2.LINE_AA)
                
                # Add white border to badge for iOS style
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw text in center of badge
                cv2.putText(
                    result_img,
                    text,
                    (badge_x + 8, badge_y + badge_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    
    return result_img

def identify_sets_from_image(img_cv, card_detector, shape_detector, fill_model, shape_model):
    """
    End-to-end pipeline to classify cards on the board and detect valid sets.
    Returns a list of sets and an annotated image.
    """
    # 1. Check and fix orientation if needed
    processed, was_rotated = verify_and_rotate_image(img_cv, card_detector)

    # 2. Verify that cards are present
    cards = detect_cards(processed, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], img_cv

    # 3. Classify each card's features, then find sets
    df_cards = classify_cards_on_board(processed, card_detector, shape_detector, fill_model, shape_model)
    found_sets = locate_all_sets(df_cards)

    if not found_sets:
        st.session_state.no_sets_found = True
        # Return the processed image even without sets
        original_orientation = restore_orientation(processed.copy(), was_rotated)
        return [], original_orientation

    # 4. Draw sets on a copy of the image
    annotated = draw_detected_sets(processed.copy(), found_sets)

    # 5. Restore orientation if we rotated earlier
    final_output = restore_orientation(annotated, was_rotated)
    return found_sets, final_output

# UI Helpers
def render_header():
    st.markdown("""
    <div class="ios-header">
        <h1>SET Card Game Detector</h1>
    </div>
    """, unsafe_allow_html=True)

def render_premium_instruction(message):
    st.markdown(f'<div class="ios-instruction">{message}</div>', unsafe_allow_html=True)

def render_error(message):
    st.markdown(f'<div class="ios-alert ios-alert-error">{message}</div>', unsafe_allow_html=True)

def render_warning(message):
    st.markdown(f'<div class="ios-alert ios-alert-warning">{message}</div>', unsafe_allow_html=True)

def reset_app_state():
    st.session_state.app_view = "upload"
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    st.session_state.uploader_key = str(random.randint(1000, 9999))

# Main App
def main():
    load_custom_css()
    if not models_loaded:
        render_error("Models failed to load. Please try again later.")
        st.stop()

    render_header()

    if st.session_state.app_view == "upload":
        render_premium_instruction("Take a photo of your SET game cards")
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed")
        if uploaded_file:
            try:
                img_pil = Image.open(uploaded_file)
                img_pil = optimize_image_size(img_pil)
                st.session_state.original_image = img_pil
                st.session_state.app_view = "preview"
                st.rerun()
            except Exception:
                render_error("Failed to load the image. Please try another photo.")

    elif st.session_state.app_view == "preview":
        st.markdown('<div style="position: relative;"><div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        render_premium_instruction("Find all valid SETs in this game")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ios-button-primary find-sets-button">', unsafe_allow_html=True)
            if st.button("Find Sets", key="find_sets_btn"):
                st.session_state.app_view = "processing"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ios-button-secondary">', unsafe_allow_html=True)
            if st.button("Try Another", key="try_different_btn"):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.app_view == "processing":
        st.markdown('<div style="position: relative;"><div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown('<div class="ios-loader-container"><div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; height: 100%;"><div class="ios-loader"></div><div class="ios-loader-text">Analyzing cards...</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        try:
            img_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
            sets_info, processed_img = identify_sets_from_image(img_cv, detector_card, detector_shape, model_fill, model_shape)
            st.session_state.sets_info = sets_info
            st.session_state.processed_image = processed_img
            st.session_state.app_view = "results"
            st.rerun()
        except Exception:
            render_error("Error processing image")
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Again", key="retry_btn"):
                st.session_state.app_view = "preview"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.app_view == "results":
        if st.session_state.no_cards_detected:
            render_error("No cards detected in the image")
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            render_premium_instruction("Try taking a clearer photo with better lighting")
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Photo", key="try_again_btn"):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state.no_sets_found:
            render_warning("No valid SETs found in this game")
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            render_premium_instruction("There are no valid SET combinations in this layout")
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Photo", key="no_sets_btn"):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            num_sets = len(st.session_state.sets_info)
            st.markdown(f'<div style="text-align: center;"><div class="ios-badge">{num_sets} SET{"" if num_sets == 1 else "s"} Found</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("New Game", key="new_game_btn"):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
