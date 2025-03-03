"""
SET Game Detector - iOS Design
================================

A Streamlit application that identifies valid SETs from uploaded images of SET card games.
Optimized for iOS following Apple's Human Interface Guidelines with responsive layout.
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
from typing import Tuple, List, Dict
import random
import time

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SET Detector",
    page_icon="🃏",
    layout="wide"
)

# =============================================================================
# SET THEME COLORS - Enhanced SET Card Game Palette
# =============================================================================
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

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
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
    st.session_state.app_view = "upload"  # Possible values: "upload", "preview", "processing", "results"
if "screen_transition" not in st.session_state:
    st.session_state.screen_transition = False

# =============================================================================
# CUSTOM CSS - Enhanced iOS-inspired styling with premium effects
# =============================================================================
def load_custom_css():
    """
    Loads premium iOS-styled CSS that follows Apple's Human Interface Guidelines.
    Enhanced with responsive layout and optimized sizing.
    """
    css = """
    <style>
    /* --- iOS Font Import --- */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Text:wght@400;500;600&display=swap');

    /* --- SET-themed Variables with Refined Responsive Spacing --- */
    :root {
        /* Colors */
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
        
        /* Dynamic spacing (will be updated by JS) */
        --vspace-xs: 0.35vh;
        --vspace-sm: 0.7vh;
        --vspace-md: 1.2vh;
        --vspace-lg: 1.6vh;
        --vspace-xl: 2.2vh;
        
        /* Fixed horizontal spacing */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 0.75rem;
        --space-lg: 1rem;
        --space-xl: 1.5rem;
        
        /* Dynamic component sizes (updated by JS) */
        --header-height: 2rem;
        --button-height: 2.75rem;
        --font-size-instruction: 0.9rem;
        
        /* Safe areas for notched iPhones */
        --safe-area-inset-top: env(safe-area-inset-top, 0px);
        --safe-area-inset-right: env(safe-area-inset-right, 0px);
        --safe-area-inset-bottom: env(safe-area-inset-bottom, 0px);
        --safe-area-inset-left: env(safe-area-inset-left, 0px);
    }

    /* --- Base Styles --- */
    body {
        font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont, sans-serif;
        background-color: #F2ECFD; /* SET-themed light purple background */
        color: var(--set-text);
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        transition: background-color var(--page-transition);
        overscroll-behavior: none; /* Prevent overscroll bounce effect */
        margin: 0;
        padding: 0;
    }

    /* --- Streamlit Override - Zero Padding for Mobile --- */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: 100%;
    }

    /* --- App Container Animation --- */
    .stApp {
        opacity: 1;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* --- Compact Premium Header --- */
    .ios-header {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: var(--vspace-xs) var(--space-md);
        margin: var(--vspace-xs) auto var(--vspace-xs);
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        box-shadow: 
            0 4px 12px rgba(124, 58, 237, 0.15),
            inset 0 1px 2px rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(124, 58, 237, 0.2);
        max-width: 85%;
        position: relative;
        overflow: hidden;
        height: var(--header-height); /* Dynamic height based on device */
    }
    
    /* Add subtle shimmer effect to header */
    .ios-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.1) 50%,
            rgba(255, 255, 255, 0) 100%
        );
        transform: rotate(30deg);
        animation: shimmer 6s infinite linear;
        z-index: -1;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(30deg); }
        100% { transform: translateX(100%) rotate(30deg); }
    }
    
    .ios-header h1 {
        font-family: -apple-system, 'SF Pro Display', BlinkMacSystemFont, sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-pink) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
        text-shadow: 0 1px 2px rgba(124, 58, 237, 0.1);
    }
    
    /* --- ENHANCED: Premium Image Container --- */
    .ios-image-container {
        margin: var(--vspace-xs) auto;
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        /* Width is dynamically set via JS */
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
        box-shadow: 
            0 8px 20px rgba(124, 58, 237, 0.15),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(124, 58, 237, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: var(--vspace-xs);
        /* Height is dynamically set via JS */
    }
    
    /* Scale image content to fit perfectly within container */
    .ios-image-container img {
        width: 100%;
        height: 100%;
        object-fit: contain; /* Ensures no cropping */
        display: block;
        transform: scale(0.98); /* Slight inset for aesthetic purposes */
        transition: transform 0.3s ease;
    }
    
    /* Hide empty containers */
    .ios-image-container:empty {
        display: none;
    }
    
    /* Image entering animation */
    @keyframes imageEnter {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .ios-image-container img {
        animation: imageEnter 0.3s ease-out forwards;
    }
    
    /* --- ENHANCED: Premium SET Loader with Perfect Centering --- */
    .ios-loader-container {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(244, 241, 250, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        z-index: 999;
        border-radius: 20px;
        animation: fadeIn 0.4s ease-out;
        /* Ensure alignment against parent */
        transform: translateZ(0);
    }
    
    .ios-loader {
        position: relative;
        width: 42px;
        height: 42px;
        border: 3px solid rgba(124, 58, 237, 0.15);
        border-top: 3px solid var(--set-purple);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.2);
        /* Force hardware acceleration and ensure pixel-perfect centering */
        transform: translateZ(0);
        margin: 0;
    }
    
    /* Add loading label beneath spinner for better UX */
    .ios-loader::after {
        content: "Finding Sets...";
        position: absolute;
        top: calc(100% + 15px);
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        font-size: 14px;
        font-weight: 500;
        color: var(--set-purple);
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
        letter-spacing: 0.01em;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* --- ENHANCED: Compact Instruction Text --- */
    .ios-instruction {
        text-align: center;
        font-size: var(--font-size-instruction);
        font-weight: 500;
        color: #4B5563;
        margin: var(--vspace-xs) auto;
        max-width: 85%;
        background: rgba(255, 255, 255, 0.5);
        padding: var(--vspace-xs) var(--space-md);
        border-radius: 14px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(124, 58, 237, 0.1);
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.08);
        animation: instructionEnter 0.4s ease-out;
    }
    
    @keyframes instructionEnter {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* --- ENHANCED: Compact iOS-styled Alert Messages --- */
    .ios-alert {
        padding: var(--vspace-xs) var(--space-md);
        border-radius: 14px;
        margin: var(--vspace-xs) auto;
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 600;
        min-height: 36px; /* Further reduced height */
        box-shadow: 
            0 4px 10px rgba(0, 0, 0, 0.08),
            inset 0 1px 2px rgba(255, 255, 255, 0.4);
        width: 85%;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: transform 0.2s ease, opacity 0.2s ease;
        animation: alertEnter 0.3s ease-out forwards;
    }
    
    @keyframes alertEnter {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .ios-alert-error {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--set-red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    
    .ios-alert-error::before {
        content: "⚠";
        margin-right: 0.6rem;
        font-size: 1.1rem;
    }
    
    .ios-alert-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    
    .ios-alert-warning::before {
        content: "ℹ";
        margin-right: 0.6rem;
        font-size: 1.1rem;
    }
    
    .ios-alert-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--set-green);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    .ios-alert-success::before {
        content: "✅";
        margin-right: 0.6rem;
        font-size: 1.1rem;
    }
    
    /* --- Compact Buttons --- */
    .ios-button-primary > button {
        background: linear-gradient(135deg, #7C3AED 0%, #9333EA 100%) !important;
        color: white !important;
        border: none !important;
        padding: var(--vspace-xs) var(--space-md);
        border-radius: 16px;
        font-family: -apple-system, 'SF Pro Display', BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        width: 100%;
        margin: var(--vspace-xs) 0 !important;
        min-height: var(--button-height);
        box-shadow: 
            0 4px 12px rgba(124, 58, 237, 0.25),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
        letter-spacing: 0.01em;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
        transform: translateY(0);
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem;
    }
    
    /* Button interaction states with subtle effects */
    .ios-button-primary > button:hover {
        transform: translateY(-1px);
        box-shadow: 
            0 6px 16px rgba(124, 58, 237, 0.3),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
    }
    
    .ios-button-primary > button:active {
        transform: translateY(1px);
        box-shadow: 
            0 2px 8px rgba(124, 58, 237, 0.2),
            inset 0 1px 2px rgba(255, 255, 255, 0.3);
    }
    
    /* Button icon with glow effect */
    .button-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.35rem;
        filter: drop-shadow(0 0 3px rgba(255, 255, 255, 0.5));
    }
    
    /* Find Sets button (Green) with magnifying glass icon */
    .find-sets-button > button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        box-shadow: 
            0 4px 12px rgba(16, 185, 129, 0.25),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
    }
    
    .find-sets-button > button:hover {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%) !important;
        box-shadow: 
            0 6px 16px rgba(16, 185, 129, 0.3),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
    }
    
    .find-sets-button > button:active {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        box-shadow: 
            0 2px 8px rgba(16, 185, 129, 0.2),
            inset 0 1px 2px rgba(255, 255, 255, 0.3);
    }
    
    /* Scan New Board button with camera icon */
    .scan-button > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        box-shadow: 
            0 4px 12px rgba(79, 70, 229, 0.25),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
    }
    
    .scan-button > button:hover {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%) !important;
        box-shadow: 
            0 6px 16px rgba(79, 70, 229, 0.3),
            inset 0 1px 3px rgba(255, 255, 255, 0.4);
    }
    
    .scan-button > button:active {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%) !important;
        box-shadow: 
            0 2px 8px rgba(79, 70, 229, 0.2),
            inset 0 1px 2px rgba(255, 255, 255, 0.3);
    }

    /* --- Center Button Container --- */
    .ios-center-button {
        display: flex;
        justify-content: center;
        margin: var(--vspace-xs) auto var(--vspace-sm);
        width: 85%;
        perspective: 1000px;
    }
    
    .ios-center-button > div {
        width: 100%;
        max-width: 250px;
    }
    
    /* --- Grid Layout --- */
    .ios-grid {
        display: flex;
        gap: 0.75rem;
    }
    
    .ios-column {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* --- Hide Streamlit Elements --- */
    footer {
        display: none !important;
    }
    
    header {
        display: none !important;
    }
    
    /* --- Override Streamlit Image Styles --- */
    [data-testid="stImage"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stImage"] > img {
        margin-bottom: 0 !important;
    }
    
    /* --- ENHANCED: Premium File Uploader --- */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 16px;
        padding: 1.2rem !important;
        border: 1px dashed rgba(124, 58, 237, 0.35);
        box-shadow: 
            0 4px 10px rgba(124, 58, 237, 0.1),
            inset 0 1px 2px rgba(255, 255, 255, 0.4);
        margin: 0.5rem auto !important;
        width: 85%;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 6px 14px rgba(124, 58, 237, 0.15);
        transform: translateY(-1px);
    }
    
    [data-testid="stFileUploader"] > div > button {
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-purple-light) 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        min-height: 44px;
        box-shadow: 
            0 4px 10px rgba(124, 58, 237, 0.25),
            inset 0 1px 2px rgba(255, 255, 255, 0.3) !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"] > div > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 14px rgba(124, 58, 237, 0.35) !important;
    }
    
    [data-testid="stFileUploader"] > div > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.2) !important;
    }
    
    /* --- Remove Extra Button Margins --- */
    .stButton {
        margin: 0 !important; 
    }
    
    /* --- SET Results Badge --- */
    .ios-badge {
        display: inline-block;
        padding: 0.6rem 1rem;
        background: linear-gradient(135deg, var(--set-purple) 0%, var(--set-purple-light) 100%);
        color: white;
        border-radius: 18px;
        font-size: 1rem;
        font-weight: 700;
        margin: 0.5rem auto;
        box-shadow: 
            0 4px 10px rgba(124, 58, 237, 0.35),
            inset 0 1px 2px rgba(255, 255, 255, 0.3);
        letter-spacing: 0.01em;
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
        animation: badgeEnter 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-origin: center;
    }
    
    @keyframes badgeEnter {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* --- Screen Animation --- */
    @keyframes screenEnter {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .screen-animation {
        animation: screenEnter 0.3s ease-out forwards;
    }
    
    /* --- Hide Streamlit Elements for Mobile --- */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* --- ENHANCED: Screen Animation --- */
    .ios-screen-transition {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(124, 58, 237, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        z-index: 9999;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    
    .ios-screen-transition.active {
        opacity: 1;
    }

    /* --- Extra Touches --- */
    ::selection {
        background: rgba(124, 58, 237, 0.2);
        color: var(--set-purple);
    }
    
    /* Disable iOS text selection highlight */
    * {
        -webkit-tap-highlight-color: transparent;
    }
    
    /* Force hardware acceleration for smoother animations */
    .stApp {
        -webkit-transform: translateZ(0);
        -moz-transform: translateZ(0);
        -ms-transform: translateZ(0);
        -o-transform: translateZ(0);
        transform: translateZ(0);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# MODEL PATHS & LOADING
# =============================================================================
base_dir = Path("models")
char_path = base_dir / "Characteristics" / "11022025"
shape_path = base_dir / "Shape" / "15052024" 
card_path = base_dir / "Card" / "16042024"

@st.cache_resource(show_spinner=False)
def load_classification_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
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
def load_detection_models() -> Tuple[YOLO, YOLO]:
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

# =============================================================================
# UTILITY & DETECTION FUNCTIONS
# =============================================================================
def verify_and_rotate_image(board_image: np.ndarray, card_detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Checks if the detected cards are oriented primarily vertically or horizontally.
    If they're vertical, rotates the board_image 90 degrees clockwise for consistent processing.
    Returns (possibly_rotated_image, was_rotated_flag).
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

def restore_orientation(img: np.ndarray, was_rotated: bool) -> np.ndarray:
    """
    Restores original orientation if the image was previously rotated.
    """
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr: np.ndarray) -> str:
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

def detect_cards(board_img: np.ndarray, card_detector: YOLO) -> List[Tuple[np.ndarray, List[int]]]:
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

def predict_card_features(
    card_img: np.ndarray,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model,
    card_box: List[int]
) -> Dict:
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

def classify_cards_on_board(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> pd.DataFrame:
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

def valid_set(cards: List[dict]) -> bool:
    """
    Checks if the given 3 cards collectively form a valid SET.
    """
    for feature in ["Count", "Color", "Fill", "Shape"]:
        if len({card[feature] for card in cards}) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df: pd.DataFrame) -> List[dict]:
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

def draw_detected_sets(board_img: np.ndarray, sets_detected: List[dict]) -> np.ndarray:
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
            corner_thickness = thickness + 2  # Increased thickness for more prominent corners
            
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

            # Add subtle inner glow (two concentric rectangles with different opacity)
            inner_color = (
                min(255, color[0] + 40),
                min(255, color[1] + 40),
                min(255, color[2] + 40)
            )
            
            # Inner rectangle (glow effect)
            inner_expansion = expansion - 2
            x1i = max(0, x1 - inner_expansion)
            y1i = max(0, y1 - inner_expansion)
            x2i = min(result_img.shape[1], x2 + inner_expansion)
            y2i = min(result_img.shape[0], y2 + inner_expansion)
            cv2.rectangle(result_img, (x1i, y1i), (x2i, y2i), inner_color, 1, cv2.LINE_AA)

            # Label the first card with Set number in an iOS-style pill badge
            if i == 0:
                text = f"Set {idx+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                badge_width = text_size[0] + 16
                badge_height = 28  # Slightly taller for premium look
                badge_x = x1e
                badge_y = y1e - badge_height - 6
                
                # Draw rounded pill-shaped badge with anti-aliasing
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              color, -1, cv2.LINE_AA)
                
                # Add subtle white inner border to badge for iOS style
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw text in center of badge with better positioning
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

def identify_sets_from_image(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> Tuple[List[dict], np.ndarray]:
    """
    End-to-end pipeline to classify cards on the board and detect valid sets.
    Returns a list of sets and an annotated image.
    """
    # 1. Check and fix orientation if needed
    processed, was_rotated = verify_and_rotate_image(board_img, card_detector)

    # 2. Verify that cards are present
    cards = detect_cards(processed, card_detector)
    if not cards:
        st.session_state.no_cards_detected = True
        return [], board_img

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

def optimize_image_size(img_pil: Image.Image, max_dim=800) -> Image.Image:
    """
    Intelligently resizes a PIL image using a responsive approach for mobile viewing.
    Preserves aspect ratio while optimizing for different iPhone screen sizes.
    
    Args:
        img_pil: PIL Image to resize
        max_dim: Maximum dimension for high-density retina displays (default 800px)
    
    Returns:
        Optimally resized PIL Image
    """
    width, height = img_pil.size
    aspect_ratio = width / height
    
    # First, check if image is very large (from high megapixel cameras)
    if max(width, height) > 1500:
        # Aggressive downsize for very large images to prevent memory issues
        if width > height:
            new_width = 1500
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 1500
            new_width = int(new_height * aspect_ratio)
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        width, height = img_pil.size
    
    # Standard resize for normal-sized images
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_dim
            new_width = int(new_height * aspect_ratio)
        
        # Apply high-quality resizing with antialiasing
        resized_img = img_pil.resize((new_width, new_height), Image.LANCZOS)
        return resized_img
    
    # If image is already appropriately sized, keep it as is
    return img_pil

# =============================================================================
# UI RENDERING HELPERS
# =============================================================================
def render_header():
    """
    Renders a compact SET-themed header with enhanced glassmorphism effect.
    Uses dynamic height based on device size for optimal space utilization.
    """
    header_html = """
    <div class="ios-header">
        <h1>SET Detector</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_loading():
    """
    Shows enhanced animated loading spinner perfectly centered on the image.
    Uses flexbox and absolute positioning with transform for pixel-perfect centering.
    Includes ARIA attributes for accessibility.
    """
    loader_html = """
    <div class="ios-loader-container" aria-live="polite" role="status">
        <div class="ios-loader"></div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

def render_error(message: str):
    """
    Renders a compact iOS-style error message with animation.
    """
    html = f"""
    <div class="ios-alert ios-alert-error">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_warning(message: str):
    """
    Renders a compact iOS-style warning message with animation.
    """
    html = f"""
    <div class="ios-alert ios-alert-warning">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_success_message(num_sets: int):
    """
    Renders a compact iOS-style success message with animation.
    """
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
    Sets proper viewport for iOS devices and adds dynamic image sizing JavaScript.
    Supports notched iPhones with safe area insets and prevents scrolling/zooming.
    """
    js_snippet = """
    <script>
        // Set proper viewport for iOS
        if (!document.querySelector('meta[name="viewport"]')) {
            var meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
            document.getElementsByTagName('head')[0].appendChild(meta);
        }
        
        // Add iOS status bar meta tags for full-screen web app experience
        var statusBarMeta = document.createElement('meta');
        statusBarMeta.name = 'apple-mobile-web-app-status-bar-style';
        statusBarMeta.content = 'black-translucent';
        document.getElementsByTagName('head')[0].appendChild(statusBarMeta);
        
        // Add iOS web app capable meta tag
        var webAppMeta = document.createElement('meta');
        webAppMeta.name = 'apple-mobile-web-app-capable';
        webAppMeta.content = 'yes';
        document.getElementsByTagName('head')[0].appendChild(webAppMeta);
        
        // Add apple touch icon for home screen
        var touchIconMeta = document.createElement('link');
        touchIconMeta.rel = 'apple-touch-icon';
        touchIconMeta.href = 'data:image/png;base64,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';
        document.getElementsByTagName('head')[0].appendChild(touchIconMeta);
        
        // Prevent zooming on double-tap
        document.addEventListener('touchstart', function(event) {
            if (event.touches.length > 1) {
                event.preventDefault();
            }
        }, { passive: false });
        
        // Disable rubber-band effect
        document.body.style.overflow = 'hidden';
        document.body.style.position = 'fixed';
        document.body.style.width = '100%';
        document.body.style.height = '100%';
        
        // IMPROVED: Dynamic image sizing calculation
        function calculateOptimalImageDimensions() {
            // Get viewport dimensions
            const vh = window.innerHeight;
            const vw = window.innerWidth;
            
            // Get heights of fixed UI elements
            const headerHeight = document.querySelector('.ios-header')?.offsetHeight || 0;
            const instructionHeight = document.querySelector('.ios-instruction')?.offsetHeight || 0;
            const buttonContainerHeight = document.querySelector('.ios-center-button')?.offsetHeight || 0;
            const alertHeight = document.querySelector('.ios-alert')?.offsetHeight || 0;
            
            // Calculate total UI elements height with a safety margin for potential elements not in DOM yet
            const uiElementsHeight = headerHeight + instructionHeight + buttonContainerHeight + alertHeight;
            
            // Calculate safety margins (top and bottom)
            const topMargin = vh * 0.05; // 5% of viewport height
            const bottomMargin = vh * 0.08; // 8% of viewport height
            
            // Calculate available space for image
            // Use min to ensure we don't exceed 65% of viewport height on larger screens
            const availableHeight = Math.min(
                vh - uiElementsHeight - topMargin - bottomMargin,
                vh * 0.65
            );
            
            // Enforce minimum height (for very small screens or landscape)
            const minHeight = Math.max(180, vh * 0.25); 
            const maxHeight = Math.min(vh * 0.65, Math.max(minHeight, availableHeight));
            
            // Calculate optimal width (85% of viewport width but not more than 500px)
            const optimalWidth = Math.min(vw * 0.85, 500);
            
            // Apply to all image containers
            document.querySelectorAll('.ios-image-container').forEach(container => {
                container.style.maxHeight = `${maxHeight}px`;
                container.style.width = `${optimalWidth}px`;
                container.style.margin = `${vh * 0.015}px auto`;
            });
            
            // Apply adaptive spacing based on device height
            const root = document.documentElement;
            
            // Extra small devices (iPhone SE, iPhone 8)
            if (vh < 600) {
                root.style.setProperty('--vspace-xs', '0.25vh');
                root.style.setProperty('--vspace-sm', '0.5vh');
                root.style.setProperty('--vspace-md', '0.9vh');
                root.style.setProperty('--vspace-lg', '1.2vh');
                root.style.setProperty('--header-height', '1.8rem');
                root.style.setProperty('--button-height', '2.5rem');
                root.style.setProperty('--font-size-instruction', '0.8rem');
            } 
            // Small devices (iPhone 12/13 mini)
            else if (vh < 750) {
                root.style.setProperty('--vspace-xs', '0.3vh');
                root.style.setProperty('--vspace-sm', '0.7vh');
                root.style.setProperty('--vspace-md', '1.1vh');
                root.style.setProperty('--vspace-lg', '1.5vh');
                root.style.setProperty('--header-height', '2rem');
                root.style.setProperty('--button-height', '2.75rem');
                root.style.setProperty('--font-size-instruction', '0.85rem');
            }
            // Medium/large devices (iPhone 14/15/Pro/Max)
            else {
                root.style.setProperty('--vspace-xs', '0.4vh');
                root.style.setProperty('--vspace-sm', '0.8vh');
                root.style.setProperty('--vspace-md', '1.3vh');
                root.style.setProperty('--vspace-lg', '1.8vh');
                root.style.setProperty('--header-height', '2.2rem');
                root.style.setProperty('--button-height', '3rem');
                root.style.setProperty('--font-size-instruction', '0.9rem');
            }
            
            // Support for dynamic safe areas on notched iPhones
            const safeAreaTop = getComputedStyle(document.documentElement).getPropertyValue('--safe-area-inset-top') || '0px';
            const safeAreaBottom = getComputedStyle(document.documentElement).getPropertyValue('--safe-area-inset-bottom') || '0px';
            
            // Apply safe area adjustments if available
            if (safeAreaTop !== '0px') {
                document.querySelector('.main')?.style.setProperty('padding-top', safeAreaTop);
            }
            
            // Adjust bottom margin to account for home indicator on newer iPhones
            if (safeAreaBottom !== '0px' && safeAreaBottom !== '0') {
                const buttons = document.querySelector('.ios-center-button');
                if (buttons) {
                    buttons.style.marginBottom = `calc(${safeAreaBottom} + var(--vspace-sm))`;
                }
            }
        }
        
        // Simple debounce function to avoid excessive calculations
        function debounce(func, wait) {
            let timeout;
            return function(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
        
        // Run on load and whenever window is resized
        window.addEventListener('load', () => setTimeout(calculateOptimalImageDimensions, 300));
        window.addEventListener('resize', debounce(calculateOptimalImageDimensions, 200));
        
        // Use MutationObserver to detect dynamic content changes
        const observer = new MutationObserver(debounce(calculateOptimalImageDimensions, 150));
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Track device orientation changes
        window.addEventListener('orientationchange', function() {
            // Add slight delay to ensure dimensions are updated
            setTimeout(calculateOptimalImageDimensions, 300);
        });
        
        // Add screen transition effect
        window.addEventListener('load', function() {
            var transitionDiv = document.createElement('div');
            transitionDiv.className = 'ios-screen-transition';
            document.body.appendChild(transitionDiv);
            
            // Function to trigger transition
            window.triggerScreenTransition = function() {
                transitionDiv.classList.add('active');
                setTimeout(function() {
                    transitionDiv.classList.remove('active');
                }, 300);
            };
        });
    </script>
    """
    st.markdown(js_snippet, unsafe_allow_html=True)
    return True

def reset_app_state():
    """
    Clears and reinitializes session state to reset the app.
    """
    # Preserve device type detection
    is_mobile = st.session_state.get("is_mobile", True)
    
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != "is_mobile":
            del st.session_state[key]

    # Now reinitialize with defaults
    st.session_state.processed = False
    st.session_state.start_processing = False
    st.session_state.uploaded_file = None
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.sets_info = None
    st.session_state.no_cards_detected = False
    st.session_state.no_sets_found = False
    st.session_state.image_height = 400
    st.session_state.uploader_key = str(random.randint(1000, 9999))
    st.session_state.should_reset = True
    st.session_state.is_mobile = is_mobile
    st.session_state.app_view = "upload"
    st.session_state.screen_transition = True
    
def render_premium_instruction(message: str):
    """
    Renders a compact iOS-style instruction message with animation.
    Uses viewport-relative units for consistent spacing across different devices.
    """
    html = f"""
    <div class="ios-instruction">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """
    Main application entry point with premium iOS-styled layout optimized for iPhone.
    """
    # 1. Load custom iOS-styled CSS with enhanced effects
    load_custom_css()
    
    # 2. Set proper viewport for iOS
    is_mobile = detect_mobile_device()
    st.session_state.is_mobile = is_mobile
    
    # 3. Handle app reset if needed
    if st.session_state.get("should_reset", False):
        st.session_state.should_reset = False
        st.rerun()
    
    # 4. Display SET-themed header
    render_header()
    
    # 5. APP FLOW - Zero-scrolling approach for iPhone
    
    # UPLOAD SCREEN
    if st.session_state.app_view == "upload":
        # Centered instruction with premium styling
        render_premium_instruction("Take a photo of your SET game cards")
        
        # iOS-style file uploader
        st.file_uploader(
            "Upload a SET board image",
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )
        
        if st.session_state.get("uploader_" + st.session_state.uploader_key):
            uploaded_file = st.session_state.get("uploader_" + st.session_state.uploader_key)
            
            # Reset session state for new image
            for key in ['processed', 'processed_image', 'sets_info', 'original_image',
                        'no_cards_detected', 'no_sets_found']:
                if key in st.session_state:
                    if key in ('processed', 'no_cards_detected', 'no_sets_found'):
                        st.session_state[key] = False
                    else:
                        st.session_state[key] = None

            st.session_state.uploaded_file = uploaded_file
            try:
                img_pil = Image.open(uploaded_file)
                img_pil = optimize_image_size(img_pil, max_dim=800)  # Higher quality initial resize
                st.session_state.original_image = img_pil
                st.session_state.image_height = img_pil.height
                st.session_state.app_view = "preview"
                st.session_state.screen_transition = True
                st.rerun()
            except Exception as e:
                render_error("Failed to load the image. Please try another photo.")
    
    # PREVIEW SCREEN - Show original with Find Sets button
    elif st.session_state.app_view == "preview":
        # Screen transition animation
        if st.session_state.screen_transition:
            st.markdown('<div class="screen-animation">', unsafe_allow_html=True)
            st.session_state.screen_transition = False
        
        # Premium container for the image
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Single centered Find Sets button with premium styling and icon
        st.markdown('<div class="ios-center-button">', unsafe_allow_html=True)
        st.markdown('<div class="ios-button-primary find-sets-button">', unsafe_allow_html=True)
        
        # Add magnifying glass icon to button
        find_sets_button_html = f"""
        <button type="button" data-testid="baseButton-secondary">
            <span class="button-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </span>
            Find Sets
        </button>
        """
        
        # Use the custom HTML button 
        if st.button("Find Sets", key="find_sets_btn", use_container_width=True):
            st.session_state.app_view = "processing"
            st.session_state.screen_transition = True
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        if st.session_state.screen_transition:
            st.markdown('</div>', unsafe_allow_html=True)
    
    # PROCESSING SCREEN
    elif st.session_state.app_view == "processing":
        # Screen transition animation
        if st.session_state.screen_transition:
            st.markdown('<div class="screen-animation">', unsafe_allow_html=True)
            st.session_state.screen_transition = False
            
        # Premium container for the image and loading overlay
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        # Overlay the perfectly centered premium loader
        render_loading()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # No instruction needed during processing - loading spinner is sufficient
        
        # Process the image
        try:
            img_cv = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
            sets_info, processed_img = identify_sets_from_image(
                img_cv, detector_card, detector_shape, model_fill, model_shape
            )
            st.session_state.sets_info = sets_info
            st.session_state.processed_image = processed_img
            st.session_state.processed = True
            st.session_state.app_view = "results"
            st.session_state.screen_transition = True
            st.rerun()
        except Exception as e:
            render_error("Error processing image")
            
            # Add retry button with premium style
            st.markdown('<div class="ios-center-button">', unsafe_allow_html=True)
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Again", key="retry_btn", use_container_width=True):
                st.session_state.app_view = "preview"
                st.session_state.screen_transition = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        if st.session_state.screen_transition:
            st.markdown('</div>', unsafe_allow_html=True)
    
    # RESULTS SCREEN
    elif st.session_state.app_view == "results":
        # Screen transition animation
        if st.session_state.screen_transition:
            st.markdown('<div class="screen-animation">', unsafe_allow_html=True)
            st.session_state.screen_transition = False
            
        # Handle error cases
        if st.session_state.no_cards_detected:
            render_error("No cards detected in the image")
            
            # Show original image
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            render_premium_instruction("Try taking a clearer photo with better lighting")
            
            # Try again button with premium style
            st.markdown('<div class="ios-center-button">', unsafe_allow_html=True)
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Photo", key="try_again_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif st.session_state.no_sets_found:
            render_warning("No valid SETs found in this game")
            
            # Show original image
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            render_premium_instruction("There are no valid SET combinations in this layout")
            
            # Try again button with premium style
            st.markdown('<div class="ios-center-button">', unsafe_allow_html=True)
            st.markdown('<div class="ios-button-primary">', unsafe_allow_html=True)
            if st.button("Try Another Photo", key="no_sets_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Success case - show results with premium styling but without redundant badge
            num_sets = len(st.session_state.sets_info)
            
            # Display processed image with sets highlighted
            st.markdown('<div class="ios-image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced instruction with details - more concise
            render_premium_instruction(f"Found {num_sets} valid SET{'s' if num_sets > 1 else ''}")
            
            # Single centered button with improved wording and icon
            st.markdown('<div class="ios-center-button">', unsafe_allow_html=True)
            st.markdown('<div class="ios-button-primary scan-button">', unsafe_allow_html=True)
            
            # Add camera icon to button
            scan_button_html = f"""
            <button type="button" data-testid="baseButton-secondary">
                <span class="button-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                </span>
                Scan New Board
            </button>
            """
            
            # Use the custom HTML button
            if st.button("Scan New Board", key="new_game_btn", use_container_width=True):
                reset_app_state()
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        if st.session_state.screen_transition:
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
