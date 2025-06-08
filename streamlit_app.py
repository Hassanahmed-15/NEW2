import os
import sys
import base64
import json
import tempfile
import logging
from functools import wraps

# =============================================================================
# 0. CRITICAL RUNTIME CONFIGURATION
# =============================================================================
os.environ["STREAMLIT_SERVER_HEADLESS"] = "1"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "500"  # 500MB max upload
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable huggingface warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Error handling decorator
def catch_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            st.error(f"An error occurred: {str(e)}")
            st.stop()
    return wrapper

# =============================================================================
# 1. IMPORT STREAMLIT WITH MEMORY MANAGEMENT
# =============================================================================
import streamlit as st
import psutil

# Initialize model cache at startup
MODEL_CACHE = {}

@catch_errors
def load_model_once():
    """Load the model once and cache it"""
    if "model" not in MODEL_CACHE:
        logger.info("Loading model for the first time...")
        from processing import initialize_model
        MODEL_CACHE["model"] = initialize_model()
    return MODEL_CACHE["model"]

# =============================================================================
# 2. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Echo Clip Report",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 3. BACKGROUND IMAGE CSS
# =============================================================================
@catch_errors
def setup_background():
    bg_path = os.path.join("templates", "hello.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as img_file:
            encoded_bg = base64.b64encode(img_file.read()).decode()
        page_bg_css = f"""
        <style>
        .stApp {{
          background: url("data:image/jpg;base64,{encoded_bg}") no-repeat center center fixed;
          background-size: cover;
          font-family: 'Segoe UI', sans-serif;
          color: #fff !important;
        }}
        .upload-card {{
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 12px;
            padding: 3rem;
            width: 480px;
            margin: 10% auto;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        .upload-title {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.2rem;
            color: #fff;
        }}
        .upload-subtitle {{
            font-size: 1rem;
            color: #ddd;
            margin-bottom: 1.5rem;
        }}
        </style>
        """
        st.markdown(page_bg_css, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not find background image. Double-check the path.")

setup_background()

# =============================================================================
# 4. UPLOAD CARD
# =============================================================================
@catch_errors
def setup_upload_card():
    logo_path = "static/logo_1.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="100" style="margin-bottom: 1rem;" />'
    else:
        logo_html = '<div style="margin-bottom: 1rem;"><strong>[Logo Missing]</strong></div>'

    st.markdown(f"""
    <div class="upload-card">
        {logo_html}
        <div class="upload-title">Echo Clip Report</div>
        <div class="upload-subtitle">Upload AVI Video or DICOM Imagery</div>
    </div>
    """, unsafe_allow_html=True)

setup_upload_card()

# =============================================================================
# 5. FILE UPLOAD AND PROCESSING WITH MEMORY CHECKS
# =============================================================================
@catch_errors
def handle_file_processing():
    st.write("\n")
    uploaded_file = st.file_uploader(
        label="Upload File", 
        type=["avi", "dcm"], 
        label_visibility="collapsed"
    )

    if st.button("Upload", key="upload_button"):
        if uploaded_file is None:
            st.error("Please choose an .avi or .dcm file first.")
            return

        # Check available memory before processing
        mem = psutil.virtual_memory()
        if mem.available < 1 * 1024 * 1024 * 1024:  # 1GB threshold
            st.error("Insufficient memory available for processing")
            logger.error(f"Low memory: {mem.available/(1024**3):.1f}GB available")
            return

        # Use tempfile for cloud environments
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
            logger.info(f"Temporarily saved to: {temp_path}")

        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".avi":
                video_path = temp_path
            else:
                video_path = os.path.join(tempfile.gettempdir(), "converted_video.avi")
                with st.spinner("Converting DICOM to AVI..."):
                    logger.info("Starting DICOM conversion")
                    from processing import process_dicom
                    process_dicom(temp_path, video_path)

            with st.spinner("Running inference on video..."):
                logger.info("Starting video analysis")
                model = load_model_once()
                from processing import process_video_with_model
                process_video_with_model(video_path, model)
                st.success("✅ Report generated successfully!")
                st.session_state["report_ready"] = True

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
        finally:
            # Clean up temporary files
            for f in [temp_path, video_path if 'video_path' in locals() else None]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                        logger.info(f"Cleaned up temporary file: {f}")
                    except Exception as e:
                        logger.warning(f"Could not remove {f}: {str(e)}")

handle_file_processing()

# =============================================================================
# 6. SHOW RESULTS
# =============================================================================
@catch_errors
def display_results():
    if st.session_state.get("report_ready", False):
        st.header("🔍 Report Preview")

        try:
            report_path = "report_data.json"
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_data = json.load(f)

                for key, value in report_data.items():
                    if isinstance(value, list):
                        st.subheader(key.replace("_", " ").capitalize())
                        for item in value:
                            display_text = f"• {item[0]}" if isinstance(item, list) else f"• {item}"
                            st.write(display_text)
                    elif isinstance(value, dict):
                        st.subheader(key.replace("_", " ").capitalize())
                        st.write(f"Predicted value: {value.get('predicted_value', 'N/A')}")
                        st.write(f"Closest prompt: {value.get('closest_prompt', 'N/A')}")

                pdf_path = "final_report.pdf"
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📄 Download Final PDF",
                            data=pdf_file,
                            file_name="final_report.pdf",
                            mime="application/pdf",
                            key="pdf_download"
                        )
            else:
                st.warning("Report data not found. Please try processing again.")
                logger.warning("Report data file not found")

        except Exception as e:
            st.error(f"Failed to display report: {str(e)}")
            logger.exception("Report display failed")

display_results()

# =============================================================================
# 7. DEBUG UTILITIES
# =============================================================================
if st.secrets.get("DEBUG", False):
    st.sidebar.title("Debug Info")
    
    if st.sidebar.button("Show Logs"):
        if os.path.exists("app_debug.log"):
            with open("app_debug.log") as f:
                st.sidebar.code(f.read())
        else:
            st.sidebar.warning("No log file found")
    
    st.sidebar.write("Memory Info:", psutil.virtual_memory())
    st.sidebar.write("CPU Usage:", psutil.cpu_percent())
    if "model" in MODEL_CACHE:
        st.sidebar.write("Model loaded successfully")
