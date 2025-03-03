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
from typing import Tuple, List, Dict
import time

###############################################################################
# STREAMLIT CONFIGURATION
###############################################################################
st.set_page_config(
    page_title="SET Detector",
    page_icon="🃏",
    layout="wide"
)

###############################################################################
# THEME COLORS
###############################################################################
SET_THEME = {
    "primary": "#7C3AED",     # SET Purple
    "secondary": "#10B981",   # SET Green
    "accent": "#EC4899",      # SET Pink
    "red": "#EF4444",         # SET Red
    "green": "#10B981",       # SET Green
    "purple": "#8B5CF6",      # SET Light Purple
    "background": "#F4F1FA",  # Light Purple Background
    "card": "#FFFFFF",        # White
    "text": "#1F2937",        # Dark Text
    "text_muted": "#6B7280",  # Gray Text
}

###############################################################################
# SESSION STATE INITIALIZATION
###############################################################################
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
    st.session_state.is_mobile = True  # Default to mobile
if "should_reset" not in st.session_state:
    st.session_state.should_reset = False
if "no_cards_detected" not in st.session_state:
    st.session_state.no_cards_detected = False
if "no_sets_found" not in st.session_state:
    st.session_state.no_sets_found = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "initial"
if "image_height" not in st.session_state:
    st.session_state.image_height = 400
if "app_view" not in st.session_state:
    st.session_state.app_view = "upload"  # Could be "upload", "preview", "processing", "results"
if "screen_transition" not in st.session_state:
    st.session_state.screen_transition = False


###############################################################################
# CUSTOM CSS – Updated to handle:
#   1) Loading spinner perfectly centered
#   2) Responsive image sizing
#   3) Consistent padding/layout with relative units
###############################################################################
def load_custom_css():
    """
    Loads iOS-styled CSS with enhancements:
      - Perfectly centered loader
      - Responsive image sizing
      - Relative units for padding/layout
    """
    css = """
    <style>
    /* ========================================= */
    /*         iOS FONT & BASE VARIABLES        */
    /* ========================================= */

    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Text:wght@400;500;600&display=swap');

    :root {
        --set-purple: #7C3AED;
        --set-purple-light: #8B5CF6;
        --set-purple-dark: #6D28D9;
        --set-green: #10B981;
        --set-green-light: #34D399;
        --set-green-dark: #059669;
        --set-red: #EF4444;
        --set-red-light: #F87171;
        --set-red-dark: #DC2626;
        --set-pink: #EC4899;
        --set-pink-light: #F472B6;
        --set-background: #F8F5FF;
        --set-card: #FFFFFF;
        --set-text: #1F2937;
        --set-text-light: #4B5563;
        --set-text-muted: #6B7280;
        --set-border: rgba(124, 58, 237, 0.25);
        --page-transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);

        /* Relative spacing units for consistent iOS layout */
        --spacer-xs: 1vh;
        --spacer-sm: 2vh;
        --spacer-md: 3vh;
        --spacer-lg: 5vh;
    }

    body {
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        background-color: #F2ECFD;
        color: var(--set-text);
        margin: 0;
        padding: 0;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        transition: background-color var(--page-transition);
        overscroll-behavior: none;
    }

    .stApp {
        opacity: 1;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Use relative padding in the main container for a more flexible layout */
    .main .block-container {
        padding-top: var(--spacer-sm);
        padding-bottom: var(--spacer-sm);
        padding-left: var(--spacer-xs);
        padding-right: var(--spacer-xs);
        max-width: 100%;
    }

    footer, header {
        display: none !important;
    }

    /* ========================================= */
    /*          PREMIUM IOS-STYLE HEADER         */
    /* ========================================= */
    .ios-header {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: var(--spacer-xs) var(--spacer-md);
        margin: 0 auto var(--spacer-sm);
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(236, 72, 153, 0.08));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 18px;
        box-shadow:
            0 4px 16px rgba(124, 58, 237, 0.15),
            inset 0 1px 2px rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(124, 58, 237, 0.2);
        max-width: 85%;
        position: relative;
        overflow: hidden;
    }
    .ios-header h1 {
        font-family: -apple-system, 'SF Pro Display', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple), var(--set-pink));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
        text-shadow: 0 1px 2px rgba(124, 58, 237, 0.1);
    }

    /* ========================================= */
    /*     CENTERED LOADING SPINNER OVERLAY      */
    /* ========================================= */
    .ios-loader-container {
        /* Position absolutely relative to parent container, then center it with transform */
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        background: rgba(244, 241, 250, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        z-index: 10;
        animation: fadeIn 0.4s ease-out;
        padding: 1rem; /* optional small padding inside the overlay */
    }
    .ios-loader {
        width: 42px;
        height: 42px;
        border: 3px solid rgba(124, 58, 237, 0.15);
        border-top: 3px solid var(--set-purple);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.2);
        margin-bottom: 1rem;
    }
    @keyframes spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .ios-loader-text {
        display: none;
    }

    /* ========================================= */
    /*        RESPONSIVE IMAGE CONTAINER         */
    /* ========================================= */
    .ios-image-container {
        margin: var(--spacer-sm) auto;
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        /* Instead of a fixed max-height, use a fraction of viewport height,
           and leave space for controls above/below. */
        max-height: calc(100vh - 25vh);
        min-height: 20vh;
        width: 85%;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(236, 72, 153, 0.05));
        box-shadow:
            0 10px 25px rgba(124, 58, 237, 0.15),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(124, 58, 237, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .ios-image-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        /* Remove forced scale transforms; let object-fit handle it */
        transform: none;
        transition: transform 0.3s ease;
    }

    /* Additional iOS-styled elements from the original code omitted for brevity.
       Make sure to keep your original button styles, alerts, etc. as needed. */

    /* Example for narrower button padding: */
    .ios-button-primary > button {
        padding: 0.5rem 0.8rem !important;
        min-height: 48px !important;
    }

    /* Demo: iOS-styled alert messages, short spacing. */
    .ios-alert {
        margin: var(--spacer-xs) auto;
        /* etc. */
    }

    /* FadeIn animation reused for overlays, etc. */
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

###############################################################################
# MODEL LOADING (EXAMPLE YOLO / KERAS MODELS)
###############################################################################
base_dir = Path("models")
char_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024"
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Loads Keras classification models for 'shape' and 'fill'.
    Adjust paths/names as needed.
    """
    try:
        model_shape = load_model(str(char_path / "shape_model.keras"))
        model_fill = load_model(str(char_path / "fill_model.keras"))
        return model_shape, model_fill
    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_detection_models() -> Tuple[YOLO, YOLO]:
    """
    Loads YOLO detection models for cards and shapes.
    Adjust paths/names as needed.
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

try:
    model_shape, model_fill = load_classification_models()
    detector_card, detector_shape = load_detection_models()
    models_loaded = all([model_shape, model_fill, detector_card, detector_shape])
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

###############################################################################
# IMAGE/SET DETECTION UTILS
###############################################################################
def verify_and_rotate_image(board_image: np.ndarray, card_detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Checks if the detected cards are primarily vertical or horizontal.
    Rotates 90° clockwise if cards appear vertical.
    """
    detection = card_detector(board_image)
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0:
        return board_image, False

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    if np.mean(heights) > np.mean(widths):
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False

def restore_orientation(img: np.ndarray, was_rotated: bool) -> np.ndarray:
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr: np.ndarray) -> str:
    """
    Basic color classification (red, green, purple) via HSV ranges.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    counts = {
        "green": cv2.countNonZero(mask_green),
        "purple": cv2.countNonZero(mask_purple),
        "red": cv2.countNonZero(mask_red),
    }
    return max(counts, key=counts.get)

def detect_cards(board_img: np.ndarray, card_detector: YOLO) -> List[Tuple[np.ndarray, List[int]]]:
    result = card_detector(board_img)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    detected_cards = []
    for x1, y1, x2, y2 in boxes:
        detected_cards.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2]))
    return detected_cards

def predict_card_features(
    card_img: np.ndarray,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model,
    card_box: List[int]
) -> Dict:
    shape_detections = shape_detector(card_img)
    c_h, c_w = card_img.shape[:2]
    card_area = c_w * c_h

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

def classify_cards_on_board(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> pd.DataFrame:
    detected_cards = detect_cards(board_img, card_detector)
    card_rows = []
    for (card_img, box) in detected_cards:
        feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
        card_rows.append({
            "Count": feats['count'],
            "Color": feats['color'],
            "Fill": feats['fill'],
            "Shape": feats['shape'],
            "Coordinates": feats['box']
        })
    return pd.DataFrame(card_rows)

def valid_set(cards: List[dict]) -> bool:
    for feature in ["Count", "Color", "Fill", "Shape"]:
        if len({card[feature] for card in cards}) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df: pd.DataFrame) -> List[dict]:
    found_sets = []
    from itertools import combinations
    for combo in combinations(cards_df.iterrows(), 3):
        cards = [c[1] for c in combo]
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [
                    {f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']}
                    for card in cards
                ]
            })
    return found_sets

def draw_detected_sets(board_img: np.ndarray, sets_detected: List[dict]) -> np.ndarray:
    set_colors = [
        (68, 68, 239),
        (89, 199, 16),
        (246, 92, 139),
        (0, 149, 255),
        (222, 82, 175),
        (85, 45, 255)
    ]
    base_thickness = 4
    base_expansion = 5
    result_img = board_img.copy()

    for idx, single_set in enumerate(sets_detected):
        color = set_colors[idx % len(set_colors)]
        thickness = base_thickness + (idx % 3)
        expansion = base_expansion + 10 * (idx % 3)

        for i, card_info in enumerate(single_set["cards"]):
            x1, y1, x2, y2 = card_info["Coordinates"]
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(result_img.shape[1], x2 + expansion)
            y2e = min(result_img.shape[0], y2 + expansion)

            cv2.rectangle(result_img, (x1e, y1e), (x2e, y2e), color, thickness, cv2.LINE_AA)

            corner_length = min(30, (x2e - x1e) // 4)
            corner_thickness = thickness + 2

            # top-left
            cv2.line(result_img, (x1e, y1e + corner_length), (x1e, y1e),
                     color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y1e), (x1e + corner_length, y1e),
                     color, corner_thickness, cv2.LINE_AA)
            # top-right
            cv2.line(result_img, (x2e - corner_length, y1e), (x2e, y1e),
                     color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y1e), (x2e, y1e + corner_length),
                     color, corner_thickness, cv2.LINE_AA)
            # bottom-left
            cv2.line(result_img, (x1e, y2e - corner_length), (x1e, y2e),
                     color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y2e), (x1e + corner_length, y2e),
                     color, corner_thickness, cv2.LINE_AA)
            # bottom-right
            cv2.line(result_img, (x2e - corner_length, y2e), (x2e, y2e),
                     color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y2e), (x2e, y2e - corner_length),
                     color, corner_thickness, cv2.LINE_AA)

            # optional label for Set number
            if i == 0:
                text = f"Set {idx+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                badge_width = text_size[0] + 16
                badge_height = 28
                badge_x = x1e
                badge_y = y1e - badge_height - 6

                cv2.rectangle(result_img, (badge_x, badge_y),
                              (badge_x + badge_width, badge_y + badge_height),
                              color, -1, cv2.LINE_AA)
                cv2.rectangle(result_img, (badge_x, badge_y),
                              (badge_x + badge_width, badge_y + badge_height),
                              (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    result_img, text,
                    (badge_x + 8, badge_y + badge_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA
                )

    return result_img

def identify_sets_from_image(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> Tuple[List[dict], np.ndarray]:
    processed, was_rotated = verify_and_rotate_image(board_img, card_detector)

    cards = detect_cards(processed, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], board_img

    df_cards = classify_cards_on_board(processed, card_detector, shape_detector, fill_model, shape_model)
    found_sets = locate_all_sets(df_cards)

    if not found_sets:
        st.session_state.no_sets_found = True
        original_orientation = restore_orientation(processed.copy(), was_rotated)
        return [], original_orientation

    annotated = draw_detected_sets(processed.copy(), found_sets)
    final_output = restore_orientation(annotated, was_rotated)
    return found_sets, final_output

def optimize_image_size(img_pil: Image.Image, max_dim=480) -> Image.Image:
    width, height = img_pil.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = min(max_dim, width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_dim, height)
        new_width = int(new_height * aspect_ratio)

    return img_pil.resize((new_width, new_height), Image.LANCZOS)

###############################################################################
# UI RENDERING HELPERS
###############################################################################
def render_header():
    header_html = """
    <div class="ios-header">
        <h1>SET Detector</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_loading():
    loader_html = """
    <div class="ios-loader-container">
        <div class="ios-loader"></div>
        <div class="ios-loader-text">Analyzing cards...</div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

def render_error(message: str):
    html = f"""
    <div class="ios-alert ios-alert-error">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_warning(message: str):
    html = f"""
    <div class="ios-alert ios-alert-warning">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_success_message(num_sets: int):
    if num_sets == 0:
        return
    html = f"""
    <div class="ios-alert ios-alert-success">
        Found {num_sets} SET{'' if num_sets == 1 else 's'}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def detect_mobile_device():
    """
    Inject custom JS to set viewport meta for iOS-like behavior.
    """
    js_snippet = """
    <script>
    (function() {
        if (!document.querySelector('meta[name="viewport"]')) {
            var meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
            document.getElementsByTagName('head')[0].appendChild(meta);
        }
        var statusBarMeta = document.createElement('meta');
        statusBarMeta.name = 'apple-mobile-web-app-status-bar-style';
        statusBarMeta.content = 'black-translucent';
        document.getElementsByTagName('head')[0].appendChild(statusBarMeta);

        var webAppMeta = document.createElement('meta');
        webAppMeta.name = 'apple-mobile-web-app-capable';
        webAppMeta.content = 'yes';
        document.getElementsByTagName('head')[0].appendChild(webAppMeta);

        var touchIconMeta = document.createElement('link');
        touchIconMeta.rel = 'apple-touch-icon';
        touchIconMeta.href = 'data:image/png;base64,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
    })();
    </script>
    """
    components.html(js_snippet, height=0, width=0)

###############################################################################
# MAIN APP
###############################################################################
def run_app():
    load_custom_css()
    detect_mobile_device()  # ensure iOS-friendly viewport

    render_header()

    st.title("")  # placeholder, or remove if you prefer only the custom header

    # FILE UPLOADER
    st.subheader("Upload an image of the SET game:")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key)
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        # Preview the uploaded image
        st.session_state.original_image = Image.open(uploaded_file)
        # Optionally optimize size before display
        preview = optimize_image_size(st.session_state.original_image, max_dim=600)

        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(preview, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Find SETs", key="find_sets", help="Process the board and locate valid SETs"):
            st.session_state.start_processing = True

    # PROCESSING
    if st.session_state.start_processing and st.session_state.uploaded_file:
        with st.spinner("Analyzing the board..."):
            st.session_state.start_processing = False
            st.session_state.processed = True
            # Convert PIL to OpenCV
            board_img = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
            sets_found, annotated_img = identify_sets_from_image(
                board_img,
                detector_card, detector_shape,
                model_fill, model_shape
            )
            st.session_state.sets_info = sets_found
            st.session_state.processed_image = annotated_img

    # RESULT
    if st.session_state.processed and st.session_state.processed_image is not None:
        if st.session_state.no_cards_detected:
            render_error("No cards detected in the image. Please try another photo.")
        else:
            sets_count = 0 if st.session_state.no_sets_found else len(st.session_state.sets_info)
            if sets_count == 0:
                render_warning("No valid SETs found. Try a different board.")
            else:
                render_success_message(sets_count)

            # Show the annotated image with bounding boxes
            annotated_pil = Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB))
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(annotated_pil, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Scan New Board", key="scan_new_board"):
            st.session_state.uploader_key = str(time.time())
            st.experimental_rerun()


if __name__ == "__main__":
    run_app()
