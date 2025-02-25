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
# DEVICE DETECTION - Simplified approach
# =============================================================================
def detect_device():
    """Detect device type based on user's browser width"""
    # Default to desktop view
    if 'is_mobile' not in st.session_state:
        st.session_state.is_mobile = False
    
    # Simple toggle in sidebar for testing
    if st.sidebar.checkbox("Mobile view", value=st.session_state.is_mobile, key="mobile_toggle"):
        st.session_state.is_mobile = True
    else:
        st.session_state.is_mobile = False
    
    return st.session_state.is_mobile

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
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
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
    }}
    
    .set-header h1 {{
        font-size: 2.5rem;
        margin-bottom: 0;
        background: linear-gradient(90deg, {SET_COLORS["purple"]} 0%, {SET_COLORS["primary"]} 50%, {SET_COLORS["accent"]} 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .set-header p {{
        font-size: 1.1rem;
        opacity: 0.8;
    }}
    
    /* Card styles */
    .set-card {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 1rem;
    }}
    
    .set-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }}
    
    .set-card h3 {{
        margin-top: 0;
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
        margin-bottom: 1rem;
    }}
    
    .upload-area:hover {{
        border-color: {SET_COLORS["primary"]};
        background-color: rgba(124, 58, 237, 0.1);
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
    }}
    
    .stButton>button:hover {{
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
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom download button */
    .download-btn {{
        display: inline-block;
        background: linear-gradient(90deg, {SET_COLORS["primary"]} 0%, {SET_COLORS["accent"]} 100%);
        color: white;
        text-decoration: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
        text-align: center;
    }}
    
    .download-btn:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# INITIALIZE MODEL FUNCTIONS
# =============================================================================
# We'll include placeholders for the model functions but we'll implement
# processing with mock data to make it work without requiring the models

@st.cache_resource(show_spinner=False)
def load_models():
    """
    This would load your models in the real implementation
    For this demo, we're just returning placeholders
    """
    # In a real app, this would be:
    # base_dir = Path("models")
    # card_detection_model = YOLO(str(base_dir / "Card" / "16042024" / "best.pt"))
    # return card_detection_model, shape_detection_model, fill_model, shape_model
    
    # For our demo:
    return "card_model", "shape_model", "fill_model", "shape_model"

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
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}" class="download-btn">{text}</a>'
    return href

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

def mock_process_image(img_array):
    """
    Mock processing function - in the real app, this would call your model
    This simulates detecting 3 sets in the image
    """
    # In the real app, this would process the image with your models
    # For our demo, we'll return the original image with some random boxes drawn
    processed_img = img_array.copy()
    
    # Draw some mock boxes for demonstration
    height, width = processed_img.shape[:2]
    mock_sets = []
    
    # Create 3 mock sets with random positions
    for i in range(3):
        # Draw three cards for each set with different colors
        color = (
            np.random.randint(150, 255),
            np.random.randint(150, 255),
            np.random.randint(150, 255)
        )
        
        cards = []
        for j in range(3):
            # Create random box positions
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(width//4, width//2)
            y2 = y1 + np.random.randint(height//4, height//2)
            
            # Draw rectangle
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            cv2.putText(processed_img, f"Set {i+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cards.append({
                "Count": np.random.randint(1, 4),
                "Color": np.random.choice(["red", "green", "purple"]),
                "Fill": np.random.choice(["empty", "striped", "solid"]),
                "Shape": np.random.choice(["diamond", "oval", "squiggle"]),
                "Coordinates": [x1, y1, x2, y2]
            })
        
        mock_sets.append({
            "set_id": i+1,
            "cards": cards
        })
    
    return processed_img, mock_sets

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
        optimize = st.checkbox("Optimize image for faster processing", value=True)
        
        # Processing button
        if uploaded_file is not None:
            if 'processed' not in st.session_state or not st.session_state.processed:
                if st.button("Find Sets", key="find_sets_desktop"):
                    st.session_state.start_processing = True
                    st.session_state.processed = False
            else:
                # Download button appears after processing
                processed_img = Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB))
                st.markdown(get_image_download_link(
                    processed_img,
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
                        
                        # Process the image (with mock processing for demonstration)
                        with st.spinner("Detecting sets..."):
                            # Simulate processing time
                            time.sleep(1.5)
                            
                            # Process image (mock processing for demo)
                            img_array = np.array(st.session_state.original_image)
                            processed_img, sets_info = mock_process_image(img_array)
                            
                            st.session_state.processed_image = processed_img
                            st.session_state.sets_info = sets_info
                            
                            st.session_state.processed = True
                            st.session_state.start_processing = False
                            st.experimental_rerun()
                    else:
                        st.info("Click 'Find Sets' to process the image")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show sets information
            if 'processed' in st.session_state and st.session_state.processed:
                st.markdown("<h4>Sets Found:</h4>", unsafe_allow_html=True)
                
                # Display information about the sets found
                set_cols = st.columns(min(3, len(st.session_state.sets_info)))
                for i, set_info in enumerate(st.session_state.sets_info):
                    with set_cols[i % len(set_cols)]:
                        # Get one card from the set to show details
                        card = set_info["cards"][0]
                        st.markdown(f"""
                        <div class="set-found-flash" style="background-color: rgba(255, 255, 255, 0.1); 
                                                          border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                            <strong>Set {i+1}</strong><br>
                            Number: {card["Count"]}<br>
                            Color: {card["Color"]}<br>
                            Fill: {card["Fill"]}<br>
                            Shape: {card["Shape"]}
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
    optimize = st.checkbox("Optimize image", value=True)
    
    # Processing button
    if uploaded_file is not None:
        if 'processed' not in st.session_state or not st.session_state.processed:
            process_button = st.button("Find Sets", key="find_sets_mobile", use_container_width=True)
            if process_button:
                st.session_state.start_processing = True
                st.session_state.processed = False
        else:
            # Download button appears after processing
            processed_img = Image.fromarray(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB))
            st.markdown(get_image_download_link(
                processed_img,
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
            
            # Display information about the sets found
            for i, set_info in enumerate(st.session_state.sets_info):
                # Get one card from the set to show details
                card = set_info["cards"][0]
                st.markdown(f"""
                <div class="set-found-flash" style="background-color: rgba(255, 255, 255, 0.1); 
                                                  border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;">
                    <strong>Set {i+1}</strong><br>
                    Number: {card["Count"]}, Color: {card["Color"]}<br>
                    Fill: {card["Fill"]}, Shape: {card["Shape"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            if 'start_processing' in st.session_state and st.session_state.start_processing:
                render_loader()
                
                # Process the image (with mock processing for demonstration)
                with st.spinner("Detecting sets..."):
                    # Simulate processing time
                    time.sleep(1.5)
                    
                    # Process image (mock processing for demo)
                    img_array = np.array(st.session_state.original_image)
                    processed_img, sets_info = mock_process_image(img_array)
                    
                    st.session_state.processed_image = processed_img
                    st.session_state.sets_info = sets_info
                    
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
    
    # Hide Streamlit elements
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    # Add note for the demo only - remove in real app
    st.sidebar.markdown("### Demo Mode")
    st.sidebar.info("This demo uses mock data instead of real AI processing. In a production app, this would connect to your AI models.")
    
    # Render layout based on device type
    if is_mobile:
        render_mobile_layout()
    else:
        render_desktop_layout()

if __name__ == "__main__":
    main()
