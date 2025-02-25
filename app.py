import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from itertools import combinations
from pathlib import Path
import time
import base64
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================
SET_COLORS = {
    "background": "#1F2937",
    "card": "#F9FAFB",
    "primary": "#7C3AED",
    "secondary": "#10B981",
    "accent": "#EC4899",
    "text": "#F3F4F6",
    "red": "#EF4444",
    "green": "#10B981",
    "purple": "#8B5CF6",
}

CARD_PROPERTIES = {
    "Number": [1, 2, 3],
    "Color": ["red", "green", "purple"],
    "Shape": ["diamond", "oval", "squiggle"],
    "Fill": ["empty", "striped", "solid"]
}

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="SET Game Detector",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DEVICE DETECTION - Improved approach
# =============================================================================
def detect_device():
    """Detect device type using JavaScript without page reload"""
    if "device_checked" not in st.session_state:
        device_check_js = """
        <script>
        const sendDeviceInfo = () => {
            const device = (window.innerWidth < 768) ? "mobile" : "desktop";
            window.parent.postMessage({type: "streamlit:setComponentValue", value: device}, "*");
        };
        // Call immediately and also on resize
        sendDeviceInfo();
        window.addEventListener('resize', sendDeviceInfo);
        </script>
        """
        device_type = components.html(device_check_js, height=0, key="device_detector")
        st.session_state.device_type = device_type if device_type else "desktop"
        st.session_state.device_checked = True
    
    return st.session_state.device_type == "mobile"

is_mobile = detect_device()

# =============================================================================
# CSS STYLING - Added SET theme
# =============================================================================
def load_css():
    """Load the styled CSS for the app"""
    css = f"""
    <style>
    /* Global styles with SET theme */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        background-color: {SET_COLORS["background"]};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: {SET_COLORS["text"]};
    }}
    
    p, span, div, label {{
        font-family: 'Poppins', sans-serif;
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
    }}
    
    .set-header h1 {{
        font-size: 2.5rem;
        margin-bottom: 0;
        background: linear-gradient(90deg, {SET_COLORS["purple"]} 0%, {SET_COLORS["primary"]} 50%, {SET_COLORS["accent"]} 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s infinite;
    }}
    
    .set-header p {{
        color: {SET_COLORS["text"]};
        font-size: 1.1rem;
        opacity: 0.8;
    }}
    
    /* Card styles */
    .set-card {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .set-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }}
    
    .set-card h3 {{
        margin-top: 0;
        color: {SET_COLORS["text"]};
        border-bottom: 2px solid {SET_COLORS["primary"]};
        padding-bottom: 0.5rem;
        text-align: center;
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
    }}
    
    .upload-area:hover {{
        border-color: {SET_COLORS["primary"]};
        background-color: rgba(124, 58, 237, 0.1);
    }}
    
    /* Button styling */
    .find-sets-btn {{
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
        text-align: center;
    }}
    
    .find-sets-btn:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }}
    
    /* Results area */
    .results-area {{
        margin-top: 1.5rem;
    }}
    
    .results-heading {{
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .results-heading h3 {{
        margin: 0;
        margin-left: 0.5rem;
    }}
    
    /* Sidebar styles */
    .sidebar .block-container {{
        background-color: {SET_COLORS["background"]};
    }}
    
    /* Loading animation */
    .loader {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2rem 0;
    }}
    
    .loader-dot {{
        width: 12px;
        height: 12px;
        margin: 0 5px;
        border-radius: 50%;
        display: inline-block;
        animation: loader 1.5s infinite ease-in-out both;
    }}
    
    .loader-dot-1 {{
        background-color: {SET_COLORS["red"]};
        animation-delay: -0.3s;
    }}
    
    .loader-dot-2 {{
        background-color: {SET_COLORS["green"]};
        animation-delay: -0.15s;
    }}
    
    .loader-dot-3 {{
        background-color: {SET_COLORS["purple"]};
        animation-delay: 0s;
    }}
    
    @keyframes loader {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -100% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    /* Image card */
    .image-container {{
        margin-top: 0.5rem;
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }}
    
    .image-container img {{
        width: 100%;
        border-radius: 12px;
        transition: all 0.3s ease;
    }}
    
    /* Set card animations */
    .set-found-flash {{
        animation: flashSet 1.5s forwards;
    }}
    
    @keyframes flashSet {{
        0% {{ box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(124, 58, 237, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }}
    }}
    
    /* SET explanation */
    .set-explanation {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }}
    
    .set-explanation h4 {{
        margin-top: 0;
        color: {SET_COLORS["text"]};
        font-size: 1.1rem;
    }}
    
    .set-properties {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-top: 0.5rem;
    }}
    
    .property-item {{
        font-size: 0.9rem;
        padding: 0.3rem 0.5rem;
        border-radius: 6px;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.1);
    }}
    
    /* Mobile optimization */
    @media (max-width: 768px) {{
        .set-header h1 {{
            font-size: 2rem;
        }}
        
        .set-header p {{
            font-size: 0.9rem;
        }}
        
        .set-card {{
            padding: 1rem;
        }}
        
        .upload-area {{
            padding: 1rem;
        }}
        
        .property-item {{
            font-size: 0.8rem;
            padding: 0.2rem;
        }}
    }}
    
    /* Performance optimization toggle */
    .performance-toggle {{
        display: flex;
        align-items: center;
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }}
    
    .performance-toggle p {{
        margin: 0;
        margin-left: 0.5rem;
        font-size: 0.9rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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

def get_image_download_link(img, filename="set_detected.jpg", text="Download Result"):
    """Generate a download link for an image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}" style="text-decoration: none;"><div class="find-sets-btn">{text}</div></a>'
    return href

def create_set_card_image():
    """Create a SET card placeholder image"""
    card_img = Image.new('RGB', (300, 200), color='white')
    draw = ImageDraw.Draw(card_img)
    
    # Draw a simple SET card
    # This is a simplified version - in a real implementation, 
    # you'd create proper SET card designs with proper shapes
    
    return card_img

def render_loader():
    """Render a SET-themed loader"""
    loader_html = """
    <div class="loader">
        <div class="loader-dot loader-dot-1"></div>
        <div class="loader-dot loader-dot-2"></div>
        <div class="loader-dot loader-dot-3"></div>
    </div>
    """
    return st.markdown(loader_html, unsafe_allow_html=True)

# =============================================================================
# COMPONENT LAYOUTS
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

def render_desktop_layout():
    """Render the optimized desktop layout"""
    render_header()
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="set-card">', unsafe_allow_html=True)
        st.markdown('<h3>Upload Image</h3>', unsafe_allow_html=True)
        
        # File uploader
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], 
                                       help="Upload a photo of your SET game board")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance options
        st.markdown('<div class="performance-toggle">', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 3])
        with col_a:
            optimize = st.checkbox("", value=True)
        with col_b:
            st.markdown('<p>Optimize image for faster processing</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing button
        if uploaded_file is not None:
            if 'processed' not in st.session_state or not st.session_state.processed:
                if st.button("Find Sets", key="find_sets_desktop"):
                    st.session_state.start_processing = True
                    st.session_state.processed = False
            else:
                # Download button appears after processing
                st.markdown(get_image_download_link(
                    Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)),
                    text="⬇️ Download Result"
                ), unsafe_allow_html=True)
                
                # Reset button
                if st.button("Reset", key="reset_desktop"):
                    st.session_state.processed = False
                    st.session_state.original_image = None
                    st.session_state.processed_image = None
                    st.session_state.sets_info = None
                    st.session_state.uploaded_file = None
                    st.experimental_rerun()
        
        # SET game explanation
        with st.expander("How does SET work?"):
            st.markdown("""
            <div class="set-explanation">
                <h4>SET Game Rules</h4>
                <p>In SET, a valid set consists of 3 cards where each feature is either all the same or all different across the 3 cards.</p>
                <p>Every card has 4 properties:</p>
                <div class="set-properties">
                    <div class="property-item">Number: 1-3</div>
                    <div class="property-item">Color: R/G/P</div>
                    <div class="property-item">Shape: ◆/◉/~</div>
                    <div class="property-item">Fill: ◆/◇/◈</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results area
        st.markdown('<div class="set-card">', unsafe_allow_html=True)
        st.markdown('<h3>Results</h3>', unsafe_allow_html=True)
        
        # Display area - original and processed
        if uploaded_file is not None:
            if 'original_image' not in st.session_state or st.session_state.original_image is None:
                # Load and optimize the image
                image = Image.open(uploaded_file)
                if optimize:
                    image = optimize_image(image)
                st.session_state.original_image = image
                st.session_state.image_array = np.array(image)
                
            # Display image
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(st.session_state.original_image, caption="Original Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_img2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                if 'processed' in st.session_state and st.session_state.processed:
                    st.image(
                        cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
                        caption=f"Detected {len(st.session_state.sets_info)} Sets",
                        use_column_width=True
                    )
                else:
                    if 'start_processing' in st.session_state and st.session_state.start_processing:
                        render_loader()
                        # Here you would call your processing function
                        # For demonstration, let's simulate processing time
                        time.sleep(2)  # Replace with actual processing
                        
                        # This is where you'd actually process the image
                        # For now, let's just use the original as a placeholder
                        # processed_img, sets_info = process_image(st.session_state.image_array)
                        # st.session_state.processed_image = processed_img
                        # st.session_state.sets_info = sets_info
                        
                        # Placeholder - replace with actual processing
                        st.session_state.processed_image = np.array(st.session_state.original_image)  
                        st.session_state.sets_info = [{"set_id": 1}]  # Placeholder
                        
                        st.session_state.processed = True
                        st.session_state.start_processing = False
                        st.experimental_rerun()
                    else:
                        st.info("Click 'Find Sets' to process the image")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show sets information
            if 'processed' in st.session_state and st.session_state.processed:
                st.markdown("<h4>Sets Found:</h4>", unsafe_allow_html=True)
                
                # Here you would display information about the sets found
                # Placeholder for demonstration
                set_cols = st.columns(3)
                for i, _ in enumerate(st.session_state.sets_info):
                    with set_cols[i % 3]:
                        st.markdown(f"""
                        <div class="set-found-flash" style="background-color: rgba(255, 255, 255, 0.1); 
                                                          border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                            <strong>Set {i+1}</strong><br>
                            3 cards with matching properties
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Placeholder when no image is uploaded
            st.info("Please upload an image to detect sets")
            
            # Example image placeholder
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
                <img src="https://www.setgame.com/sites/default/files/set%20examples%203.png" 
                     style="max-width: 80%; border-radius: 12px;">
                <p style="margin-top: 0.5rem; font-size: 0.8rem;">Example: A valid SET</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_mobile_layout():
    """Render the optimized mobile layout"""
    render_header()
    
    # Upload card
    st.markdown('<div class="set-card">', unsafe_allow_html=True)
    st.markdown('<h3>Upload Image</h3>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], 
                                   help="Upload a photo of your SET game board")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance toggle - simpler for mobile
    optimize = st.checkbox("Optimize image for faster processing", value=True)
    
    # Processing button
    if uploaded_file is not None:
        if 'processed' not in st.session_state or not st.session_state.processed:
            if st.button("Find Sets", key="find_sets_mobile", use_container_width=True):
                st.session_state.start_processing = True
                st.session_state.processed = False
        else:
            # Download button appears after processing
            st.markdown(get_image_download_link(
                Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)),
                text="⬇️ Download Result"
            ), unsafe_allow_html=True)
            
            # Reset button
            if st.button("Reset", key="reset_mobile", use_container_width=True):
                st.session_state.processed = False
                st.session_state.original_image = None
                st.session_state.processed_image = None
                st.session_state.sets_info = None
                st.session_state.uploaded_file = None
                st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results - stacked for mobile
    if uploaded_file is not None:
        if 'original_image' not in st.session_state or st.session_state.original_image is None:
            # Load and optimize the image
            image = Image.open(uploaded_file)
            if optimize:
                image = optimize_image(image)
            st.session_state.original_image = image
            st.session_state.image_array = np.array(image)
        
        # Results card
        st.markdown('<div class="set-card">', unsafe_allow_html=True)
        st.markdown('<h3>Results</h3>', unsafe_allow_html=True)
        
        # Original image
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, caption="Original Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processed image
        st.markdown('<div class="image-container" style="margin-top: 1rem;">', unsafe_allow_html=True)
        if 'processed' in st.session_state and st.session_state.processed:
            st.image(
                cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
                caption=f"Detected {len(st.session_state.sets_info)} Sets",
                use_column_width=True
            )
            
            # Show sets information
            st.markdown("<h4>Sets Found:</h4>", unsafe_allow_html=True)
            
            # Placeholder for demonstration
            for i, _ in enumerate(st.session_state.sets_info):
                st.markdown(f"""
                <div class="set-found-flash" style="background-color: rgba(255, 255, 255, 0.1); 
                                                  border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                    <strong>Set {i+1}</strong><br>
                    3 cards with matching properties
                </div>
                """, unsafe_allow_html=True)
        else:
            if 'start_processing' in st.session_state and st.session_state.start_processing:
                render_loader()
                # Simulate processing time
                time.sleep(2)  # Replace with actual processing
                
                # Placeholder - replace with actual processing
                st.session_state.processed_image = np.array(st.session_state.original_image)  
                st.session_state.sets_info = [{"set_id": 1}]  # Placeholder
                
                st.session_state.processed = True
                st.session_state.start_processing = False
                st.experimental_rerun()
            else:
                st.info("Click 'Find Sets' to process the image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SET Game explanation - in an expander for mobile to save space
    with st.expander("How does SET work?"):
        st.markdown("""
        <div class="set-explanation">
            <h4>SET Game Rules</h4>
            <p>In SET, a valid set consists of 3 cards where each feature is either all the same or all different across the 3 cards.</p>
            <p>Every card has 4 properties:</p>
            <div class="set-properties">
                <div class="property-item">Number: 1-3</div>
                <div class="property-item">Color: R/G/P</div>
                <div class="property-item">Shape: ◆/◉/~</div>
                <div class="property-item">Fill: ◆/◇/◈</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if "processed" not in st.session_state:
    st.session_state.processed = False
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load CSS
    load_css()
    
    # Render layout based on device type
    if is_mobile:
        render_mobile_layout()
    else:
        render_desktop_layout()
    
    # Add the model loading and processing functions from the original code here
    # This is just a UI demonstration without the actual processing logic

if __name__ == "__main__":
    main()
