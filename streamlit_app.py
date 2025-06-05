# streamlit_app.py
import streamlit as st
import os
import base64
import json
from processing import process_dicom, process_video

# ─────────────────────────────────────────────────────────────
# 1) Page configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo Clip Report",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────
# 2) Background image CSS
# ──────────────────────────────────────────────────────────
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
    st.warning("⚠️ Could not find `static/bg/index_bg.jpg`. Double-check the path.")

# ──────────────────────────────────────────────────────────
# 3) Upload Card
# ──────────────────────────────────────────────────────────
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

# (rest of code continues unchanged)

# (rest of code continues unchanged)

# ────────────────────────────────────────────────────────────────────────
# 4) File Upload Logic
# ────────────────────────────────────────────────────────────────────────
st.write("\n")
uploaded_file = st.file_uploader(
    label="Upload File", 
    type=["avi", "dcm"], 
    label_visibility="collapsed"
)

if st.button("Upload"):
    if uploaded_file is None:
        st.error("Please choose an .avi or .dcm file first.")
    else:
        UPLOAD_DIR = "uploads"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"File saved: {uploaded_file.name}")

        ext = uploaded_file.name.lower().split(".")[-1]
        if ext == "avi":
            video_path = file_path
        else:
            video_path = os.path.join(UPLOAD_DIR, "converted_video.avi")
            st.info("Converting DICOM to AVI...")
            process_dicom(file_path, video_path)

        st.info("Running inference on video...")
        process_video(video_path)
        st.success("✅ Report generated successfully!")

        st.session_state["report_ready"] = True

# ────────────────────────────────────────────────────────────────────────
# 5) Show Results
# ────────────────────────────────────────────────────────────────────────
if st.session_state.get("report_ready", False):
    st.header("🔍 Report Preview")

    try:
        with open("report_data.json", "r") as f:
            report_data = json.load(f)

        for key, value in report_data.items():
            if isinstance(value, list):
                st.subheader(key.replace("_", " ").capitalize())
                for item in value:
                    st.write("• " + str(item[0]) if isinstance(item, list) else str(item))
            elif isinstance(value, dict):
                st.subheader(key.replace("_", " ").capitalize())
                st.write(f"Predicted value: {value.get('predicted_value')}")
                st.write(f"Closest prompt: {value.get('closest_prompt')}")

        pdf_path = os.path.join("final_report.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Final PDF",
                    data=pdf_file,
                    file_name="final_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Failed to load report: {e}")
