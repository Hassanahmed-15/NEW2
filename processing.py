import os
import json
import pydicom
import cv2
import torch
import torch.nn.functional as F
from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T
from utils import zero_shot_prompts, read_avi, compute_regression_metric, compute_binary_metric
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Global model cache
_MODEL = None
_PREPROCESS = None

def get_model():
    """Initialize and cache the model once"""
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        logger.info("Initializing model...")
        _MODEL, _, _PREPROCESS = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip",
            precision="fp32",
            device="cpu"
        )
    return _MODEL, _PREPROCESS

def process_dicom(dicom_path, output_video_path):
    """Process DICOM file to AVI"""
    try:
        logger.info(f"Processing DICOM: {dicom_path}")
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract frames
        n_frames = int(dicom_data.get("NumberOfFrames", 1))
        pixel_data = dicom_data.pixel_array[:, 110:-40, 150:-150, :]
        fps = 1000 / float(dicom_data.get("FrameTime", 40))
        
        # Initialize video writer
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (224, 224)
        )
        
        # Process frames
        for i in range(n_frames if n_frames > 1 else 1):
            frame = pixel_data[i] if n_frames > 1 else pixel_data
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if len(frame.shape) == 2 else cv2.COLOR_RGB2BGR)
            writer.write(cv2.resize(frame, (224, 224)))
        
        writer.release()
        logger.info(f"Saved video to: {output_video_path}")
        
    except Exception as e:
        logger.error(f"DICOM processing failed: {str(e)}")
        raise

def process_video(video_path):
    """Main video processing function"""
    try:
        logger.info(f"Processing video: {video_path}")
        model, preprocess = get_model()
        
        # Read and preprocess video
        frames = read_avi(video_path, (224, 224))
        frames = frames[0:min(40, len(frames)):2]  # Sample frames
        frames = torch.stack([preprocess(T.ToPILImage()(f)) for f in frames]).float()
        
        # Generate embeddings
        with torch.no_grad():
            video_embed = F.normalize(model.encode_image(frames), dim=-1).unsqueeze(0)
        
        # Generate report
        report = generate_report(video_embed)
        
        # Save results
        with open("report_data.json", "w") as f:
            json.dump(report, f)
        logger.info("Report generated successfully")
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise

def generate_report(video_embed):
    """Generate report from video embeddings"""
    model, _ = get_model()
    
    def get_top_prompts(key, n=3):
        prompts = zero_shot_prompts[key]
        with torch.no_grad():
            text_embeds = F.normalize(model.encode_text(tokenize(prompts)), dim=-1)
        scores = (video_embed @ text_embeds.T).mean(dim=1)
        top_idx = scores.topk(min(n, len(prompts))).indices[0]
        return [[prompts[i], float(scores[0][i])] for i in top_idx]
    
    def get_cont_prompt(key, values):
        prompts = [p.replace("<#>", str(i)) for p in zero_shot_prompts[key] for i in range(values)]
        with torch.no_grad():
            text_embeds = F.normalize(model.encode_text(tokenize(prompts)), dim=-1)
        pred = compute_regression_metric(video_embed, text_embeds, list(range(values)))
        closest = (torch.abs(torch.arange(values) - pred.item())).argmin()
        return {
            "predicted_value": float(pred.item()),
            "closest_prompt": zero_shot_prompts[key][0].replace("<#>", str(closest.item()))
        }
    
    # Binary conditions check
    binary_conds = [
        "severe_right_ventricle_size", "moderate_right_ventricle_size", 
        "mild_right_ventricle_size", "severe_left_atrium_size",
        "moderate_left_atrium_size", "mild_left_atrium_size",
        "severe_right_atrium_size", "moderate_right_atrium_size",
        "mild_right_atrium_size", "tavr"
    ]
    
    detected = [
        cond for cond in binary_conds 
        if compute_binary_metric(
            video_embed, 
            F.normalize(model.encode_text(tokenize(zero_shot_prompts[cond])), dim=-1)
        ).item() > 0.28
    ]
    
    return {
        "mitraclip": get_top_prompts("mitraclip", 2),
        "impella": get_top_prompts("impella", 2),
        "severe_lv_dilation": get_top_prompts("severe_left_ventricle_dilation", 2),
        "moderate_lv_dilation": get_top_prompts("moderate_left_ventricle_dilation", 2),
        "mild_lv_dilation": get_top_prompts("mild_left_ventricle_dilation", 2),
        "sig_elev_ra_pressure": get_top_prompts("significantly_elevated_right_atrial_pressure", 1),
        "norm_r_atrial_pressure": get_top_prompts("normal_right_atrial_pressure", 1),
        "detected_conditions": detected,
        "ejection_fraction": get_cont_prompt('ejection_fraction', 101),
        "pulmonary_artery_pressure": get_cont_prompt('pulmonary_artery_pressure', 17)
    }
