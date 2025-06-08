import os
import base64
import json
import tempfile
import streamlit as st
from processing import process_dicom, process_video

# Configuration
os.environ["STREAMLIT_SERVER_HEADLESS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Page setup
st.set_page_config(
    page_title="Echo Clip Report",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Background styling
bg_path = os.path.join("templates", "hello.jpg")
if os.path.exists(bg_path):
    with open(bg_path, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{bg_b64}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Segoe UI', sans-serif;
        color: #fff !important;
    }}
    .upload-card {{
        background-color: rgba(0,0,0,0.7);
        border-radius: 12px;
        padding: 3rem;
        width: 480px;
        margin: 10% auto;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# Upload card
logo_path = "static/logo_1.png"
logo_html = f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path,"rb").read()).decode()}" width="100" style="margin-bottom:1rem"/>' if os.path.exists(logo_path) else '<div style="margin-bottom:1rem"><strong>[Logo Missing]</strong></div>'

st.markdown(f"""
<div class="upload-card">
    {logo_html}
    <div class="upload-title">Echo Clip Report</div>
    <div class="upload-subtitle">Upload AVI Video or DICOM Imagery</div>
</div>
""", unsafe_allow_html=True)

# File processing
uploaded_file = st.file_uploader("Upload File", type=["avi", "dcm"], label_visibility="collapsed")

if st.button("Process"):
    if not uploaded_file:
        st.error("Please select a file first")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".avi":
                video_path = tmp_path
            else:
                video_path = os.path.join(tempfile.gettempdir(), "converted.avi")
                with st.spinner("Converting DICOM..."):
                    process_dicom(tmp_path, video_path)
            
            with st.spinner("Analyzing..."):
                process_video(video_path)
                st.success("Analysis complete!")
                st.session_state.report_ready = True
        
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
        finally:
            for f in [tmp_path, video_path if 'video_path' in locals() else None]:
                if f and os.path.exists(f):
                    try: os.remove(f)
                    except: pass

# Results display
if st.session_state.get("report_ready"):
    try:
        with open("report_data.json") as f:
            data = json.load(f)
        
        st.header("Report Preview")
        for k, v in data.items():
            if isinstance(v, list):
                st.subheader(k.replace("_", " ").title())
                for item in v:
                    st.write(f"• {item[0] if isinstance(item, list) else item}")
            elif isinstance(v, dict):
                st.subheader(k.replace("_", " ").title())
                st.write(f"Predicted value: {v.get('predicted_value', 'N/A')}")
                st.write(f"Closest prompt: {v.get('closest_prompt', 'N/A')}")
        
        if os.path.exists("final_report.pdf"):
            with open("final_report.pdf", "rb") as f:
                st.download_button("Download PDF", f, "report.pdf", "application/pdf")
    
    except Exception as e:
        st.error(f"Failed to display results: {str(e)}")
