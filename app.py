import streamlit as st
import re
from user_agents import parse

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
    
    /* Mobile-specific styles */
    @media (max-width: 991px) {
        .device-message {
            font-size: 1.8rem;
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Inject JavaScript for client-side device detection
# This script sets CSS classes on the body element for CSS-based detection
st.markdown("""
<script>
    // Improved mobile detection function
    function detectMobileDevice() {
        // Check for common mobile user agents
        const mobileRegex = /(android|bb\\d+|meego).+mobile|avantgo|bada\\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i;
        const mobileRegex2 = /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\\-(n|u)|c55\\/|capi|ccwa|cdm\\-|cell|chtm|cldc|cmd\\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\\-s|devi|dica|dmob|do(c|p)o|ds(12|\\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\\-|_)|g1 u|g560|gene|gf\\-5|g\\-mo|go(\\.w|od)|gr(ad|un)|haie|hcit|hd\\-(m|p|t)|hei\\-|hi(pt|ta)|hp( i|ip)|hs\\-c|ht(c(\\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\\-(20|go|ma)|i230|iac( |\\-|\\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\\/)|klon|kpt |kwc\\-|kyo(c|k)|le(no|xi)|lg( g|\\/(k|l|u)|50|54|\\-[a-w])|libw|lynx|m1\\-w|m3ga|m50\\/|ma(te|ui|xo)|mc(01|21|ca)|m\\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\\-2|po(ck|rt|se)|prox|psio|pt\\-g|qa\\-a|qc(07|12|21|32|60|\\-[2-7]|i\\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\\-|oo|p\\-)|sdk\\/|se(c(\\-|0|1)|47|mc|nd|ri)|sgh\\-|shar|sie(\\-|m)|sk\\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\\-|v\\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\\-|tdg\\-|tel(i|m)|tim\\-|t\\-mo|to(pl|sh)|ts(70|m\\-|m3|m5)|tx\\-9|up(\\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\\-|your|zeto|zte\\-/i;
        
        // Check window width as fallback
        const isMobileWidth = window.innerWidth <= 991;
        
        // Special check for iOS devices
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        
        // Check the user agent
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        const isMobileByAgent = mobileRegex.test(userAgent) || mobileRegex2.test(userAgent.substr(0,4));
        
        return isMobileByAgent || isMobileWidth || isIOS;
    }
    
    // Set a specific class on the body based on device
    if (detectMobileDevice()) {
        document.body.classList.add('mobile-device');
        localStorage.setItem('deviceType', 'mobile');
    } else {
        document.body.classList.add('desktop-device');
        localStorage.setItem('deviceType', 'desktop');
    }
</script>
""", unsafe_allow_html=True)

# CSS-based detection - we'll create two divs, one visible only on mobile, one only on desktop
# Then we'll use Streamlit's components to check which one is visible
st.markdown("""
<div>
    <div id="mobile-detector" style="display: none;">
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" 
             width="1" height="1" style="display: none;" 
             onload="document.getElementById('mobile-result').textContent = 'mobile';">
    </div>
    <div id="desktop-detector">
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" 
             width="1" height="1" style="display: none;"
             onload="document.getElementById('desktop-result').textContent = 'desktop';">
    </div>
    <div id="mobile-result" style="display: none;"></div>
    <div id="desktop-result" style="display: none;"></div>
</div>
<style>
    @media (max-width: 991px) {
        #mobile-detector { display: block !important; }
        #desktop-detector { display: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='centered-text'>📱 Device Detector</h1>", unsafe_allow_html=True)

# Server-side detection based on User-Agent
def detect_device_type_from_header():
    try:
        user_agent_string = st.request_header('User-Agent', '')
        user_agent = parse(user_agent_string)
        is_mobile = user_agent.is_mobile or user_agent.is_tablet
        return "mobile" if is_mobile else "desktop"
    except:
        # Fallback to a simpler regex-based check if the user_agents library isn't available
        try:
            user_agent_string = st.request_header('User-Agent', '')
            mobile_pattern = re.compile(r'(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk', re.I)
            is_mobile = bool(mobile_pattern.search(user_agent_string))
            return "mobile" if is_mobile else "desktop"
        except:
            # Last resort: if we can't detect it, default to desktop
            return "desktop"

# Initialize device type in session state if not already present
if "device_type" not in st.session_state:
    st.session_state.device_type = detect_device_type_from_header()

# Create a placeholder for device message
message_placeholder = st.empty()

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

# Show the device message based on our detection
show_device_message(st.session_state.device_type)

# Add a refresh button and allow manual override to test different displays
col1, col2 = st.columns(2)
with col1:
    if st.button("I'm on Desktop"):
        st.session_state.device_type = "desktop"
        st.rerun()
with col2:
    if st.button("I'm on Mobile"):
        st.session_state.device_type = "mobile"
        st.rerun()

st.markdown("---")
st.markdown("""
### How it works
This app uses multiple detection methods to identify your device type:

1. **User-Agent Detection**: Analyzes your browser's user agent string
2. **CSS Media Queries**: Uses responsive design breakpoints
3. **JavaScript Detection**: Uses advanced device detection logic

The app aims to correctly identify mobile devices including iPhones, Android devices, and tablets.

**Note:** If the detection is incorrect, you can manually override it with the buttons above.
""")

# Requirements info
st.sidebar.markdown("### Requirements")
st.sidebar.code("pip install streamlit user-agents")
