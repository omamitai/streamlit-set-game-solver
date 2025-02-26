import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Device Detector",
    page_icon="📱",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .device-message {
        font-size: 2.5rem;
        text-align: center;
        margin: 2rem auto;
        padding: 2rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .desktop-message {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #1E3A8A;
    }
    .mobile-message {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        color: #7B341E;
    }
    .centered-text {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Inject JavaScript to detect device type and store in sessionStorage
st.markdown("""
<script>
    // Function to detect if device is mobile
    function isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
               (window.innerWidth <= 991);
    }
    
    // Store device info in sessionStorage
    if (isMobile()) {
        sessionStorage.setItem('deviceType', 'mobile');
    } else {
        sessionStorage.setItem('deviceType', 'desktop');
    }
    
    // Attempt to communicate with Streamlit
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        value: isMobile() ? "mobile" : "desktop"
    }, "*");
</script>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='centered-text'>📱 Device Detector</h1>", unsafe_allow_html=True)

# Get query parameters to see if we have device data
query_params = st.experimental_get_query_params()
device_param = query_params.get("device", ["unknown"])[0]

# Create a placeholder for real-time updates
message_placeholder = st.empty()

# Assume desktop by default
device_type = "desktop"

# Check window size
if st.session_state.get("_is_mobile", False) or device_param == "mobile":
    device_type = "mobile"

# Function to show the appropriate message
def show_device_message(device):
    if device == "mobile":
        message_placeholder.markdown(
            "<div class='device-message mobile-message'>You're on mobile! 📱</div>", 
            unsafe_allow_html=True
        )
    else:
        message_placeholder.markdown(
            "<div class='device-message desktop-message'>You're on desktop! 💻</div>", 
            unsafe_allow_html=True
        )

# Display initial message
show_device_message(device_type)

# Add a button to refresh detection
if st.button("Refresh Detection"):
    st.rerun()

# Add some information about how the detection works
st.markdown("---")
st.markdown("""
### How it works
This app uses JavaScript to detect whether you're viewing from a mobile or desktop device. 
The detection is based on:
- User agent string (checks for mobile device identifiers)
- Screen/window width (considers screens narrower than 991px as mobile)

**Note:** For the most accurate detection, try refreshing the page if you resize your window significantly.
""")
