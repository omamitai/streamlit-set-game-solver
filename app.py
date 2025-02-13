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
# --------------------------------------------------------------------------
def check_and_rotate_input_image(board_image: np.ndarray, detector) -> (np.ndarray, bool):
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
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if was_rotated else image

def predict_color(shape_image: np.ndarray) -> str:
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
    purple_mask = cv2.inRange(hsv_image, np.array([120, 50, 50]), np.array([160, 255, 255]))
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    color_counts = {
        'green': cv2.countNonZero(green_mask),
        'purple': cv2.countNonZero(purple_mask),
        'red': cv2.countNonZero(red_mask)
    }
    return max(color_counts, key=color_counts.get)

def predict_card_features(card_image: np.ndarray, shape_detector, fill_model, shape_model, box: list) -> dict:
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
    return {'count': len(filtered_boxes), 'color': color_label,
            'fill': fill_label, 'shape': shape_label, 'box': box}

def is_set(cards: list) -> bool:
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        if len({card[feature] for card in cards}) not in [1, 3]:
            return False
    return True

def find_sets(card_df: pd.DataFrame) -> list:
    sets_found = []
    for combo in combinations(card_df.iterrows(), 3):
        cards = [entry[1] for entry in combo]
        if is_set(cards):
            sets_found.append({
                'set_indices': [entry[0] for entry in combo],
                'cards': [{feature: card[feature] for feature in 
                           ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} for card in cards]
            })
    return sets_found

def detect_cards_from_image(board_image: np.ndarray, detector) -> list:
    card_results = detector(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [(board_image[y1:y2, x1:x2], [x1, y1, x2, y2]) for x1, y1, x2, y2 in card_boxes]

def classify_cards_from_board_image(board_image: np.ndarray, card_detector, shape_detector, fill_model, shape_model) -> pd.DataFrame:
    cards = detect_cards_from_image(board_image, card_detector)
    card_data = []
    for card_image, box in cards:
        features = predict_card_features(card_image, shape_detector, fill_model, shape_model, box)
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"
        })
    return pd.DataFrame(card_data)

def draw_sets_on_image(board_image: np.ndarray, sets_info: list) -> np.ndarray:
    colors = [(255, 0, 0), (0, 128, 0), (128, 0, 128)]  # red, green, purple
    base_thickness = 8
    base_expansion = 5
    for index, set_info in enumerate(sets_info):
        color = colors[index % len(colors)]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        for i, card in enumerate(set_info['cards']):
            coordinates = list(map(int, card['Coordinates'].split(',')))
            x1, y1, x2, y2 = coordinates
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(board_image.shape[1], x2 + expansion)
            y2_exp = min(board_image.shape[0], y2 + expansion)
            cv2.rectangle(board_image, (x1_exp, y1_exp), (x2_exp, y2_exp), color, thickness)
            if i == 0:
                cv2.putText(board_image, f"Set {index + 1}", (x1_exp, y1_exp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    return board_image

def classify_and_find_sets_from_array(board_image: np.ndarray, card_detector, shape_detector, fill_model, shape_model) -> (list, np.ndarray):
    processed_image, was_rotated = check_and_rotate_input_image(board_image, card_detector)
    card_df = classify_cards_from_board_image(processed_image, card_detector, shape_detector, fill_model, shape_model)
    sets_found = find_sets(card_df)
    annotated_image = draw_sets_on_image(processed_image.copy(), sets_found)
    final_image = restore_original_orientation(annotated_image, was_rotated)
    return sets_found, final_image

# --------------------------------------------------------------------------
# 🎲 Streamlit App UI
# --------------------------------------------------------------------------

# 🌟 Title & Description
st.markdown("<h1 style='text-align:center; color:#FFD700;'>🎴 SET Game Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image of a SET game board and click <b>Find Sets</b> to detect valid sets.</p>", unsafe_allow_html=True)

# 🔹 Layout: Two Equal Columns
col1, col2 = st.columns([1, 1])

# 📥 Left Column: Upload & Show Original Image
with col1:
    st.markdown("### 📥 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🎴 Original Image", use_column_width=True, output_format="JPEG")

# 🔍 Right Column: Process & Show Results
with col2:
    st.markdown("### 🔍 Processed Result")
    
    if uploaded_file:
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
                # Convert image back to RGB for display
                final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                st.image(final_image_rgb, caption="✅ Detected Sets", use_column_width=True, output_format="JPEG")
                
                # 🏆 Success Message
                st.toast("🎉 Sets detected successfully!", icon="✅")
                
                # 📌 Expandable Results
                with st.expander("📜 View Detected Sets Details"):
                    st.json(sets_info)
            except Exception as e:
                st.error("⚠️ An error occurred during processing:")
                st.text(traceback.format_exc())
