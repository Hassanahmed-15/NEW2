import os
import sys
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
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Global model cache
MODEL_CACHE = None

def initialize_model():
    """Initialize and cache the model once"""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        logger.info("Initializing model...")
        MODEL_CACHE = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip",
            precision="fp32",
            device="cpu"
        )
    return MODEL_CACHE

def process_dicom(dicom_path, output_video_path):
    """Process DICOM file to AVI with error handling"""
    try:
        logger.info(f"Processing DICOM: {dicom_path}")
        dicom_data = pydicom.dcmread(dicom_path)

        # Get frame information
        number_of_frames = int(dicom_data.get("NumberOfFrames", 1))
        pixel_data = dicom_data.pixel_array[:, 110:-40, 150:-150, :]
        frame_time = float(dicom_data.get("FrameTime", 40))
        fps = 1000 / frame_time

        # Initialize video writer
        frame_size = (224, 224)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        # Process frames
        if number_of_frames > 1:
            for frame_index in range(number_of_frames):
                frame = pixel_data[frame_index]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if len(frame.shape) == 2 else cv2.COLOR_RGB2BGR)
                video_writer.write(cv2.resize(frame, frame_size))
        else:
            frame = cv2.cvtColor(pixel_data, cv2.COLOR_GRAY2BGR if len(pixel_data.shape) == 2 else cv2.COLOR_RGB2BGR)
            video_writer.write(cv2.resize(frame, frame_size))

        video_writer.release()
        logger.info(f"Successfully converted DICOM to: {output_video_path}")
        
    except Exception as e:
        logger.error(f"DICOM processing failed: {str(e)}")
        raise

def process_video(video_path):
    """Process video with memory-efficient approach"""
    try:
        logger.info(f"Processing video: {video_path}")
        
        # Load model from cache
        echo_clip, _, preprocess_val = initialize_model()
        
        # Read and preprocess video
        test_video = read_avi(video_path, (224, 224))
        test_video = torch.stack([
            preprocess_val(T.ToPILImage()(frame)) 
            for frame in test_video[0:min(40, len(test_video)):2]
        ], dim=0).float()

        # Generate embeddings
        with torch.no_grad():
            test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1).unsqueeze(0)

        # Helper functions
        def get_top_prompts(embedding, key, top_n=3):
            prompts = zero_shot_prompts[key]
            tokenized = tokenize(prompts)
            with torch.no_grad():
                prompt_embeds = F.normalize(echo_clip.encode_text(tokenized), dim=-1)
            similarities = (embedding @ prompt_embeds.T).mean(dim=1)
            top_indices = similarities.topk(min(top_n, len(prompts))).indices.cpu().numpy().flatten()
            return [
                [prompts[i], float(similarities[0][i].item())]
                for i in top_indices
            ]

        def get_continuous_prompt(embedding, key, value_range):
            prompts = zero_shot_prompts[key]
            generated = []
            values = []
            for p in prompts:
                for i in range(value_range):
                    generated.append(p.replace("<#>", str(i)))
                    values.append(i)
            
            tokenized = tokenize(generated)
            with torch.no_grad():
                prompt_embeds = F.normalize(echo_clip.encode_text(tokenized), dim=-1)
            
            pred_value = compute_regression_metric(embedding, prompt_embeds, values)
            closest_idx = (torch.abs(torch.tensor(values) - pred_value.item())).argmin()
            return {
                "predicted_value": float(pred_value.item()),
                "closest_prompt": zero_shot_prompts[key][0].replace("<#>", str(values[closest_idx.item()]))
            }

        # Generate report data
        report_data = {
            "mitraclip": get_top_prompts(test_video_embedding, "mitraclip", 2),
            "impella": get_top_prompts(test_video_embedding, "impella", 2),
            "severe_lv_dilation": get_top_prompts(test_video_embedding, "severe_left_ventricle_dilation", 2),
            "moderate_lv_dilation": get_top_prompts(test_video_embedding, "moderate_left_ventricle_dilation", 2),
            "mild_lv_dilation": get_top_prompts(test_video_embedding, "mild_left_ventricle_dilation", 2),
            "sig_elev_ra_pressure": get_top_prompts(test_video_embedding, "significantly_elevated_right_atrial_pressure", 1),
            "norm_r_atrial_pressure": get_top_prompts(test_video_embedding, "normal_right_atrial_pressure", 1),
            "detected_conditions": [],
            "ejection_fraction": get_continuous_prompt(test_video_embedding, 'ejection_fraction', 101),
            "pulmonary_artery_pressure": get_continuous_prompt(test_video_embedding, 'pulmonary_artery_pressure', 17)
        }

        # Check binary conditions
        binary_conditions = [
            "severe_right_ventricle_size", "moderate_right_ventricle_size", "mild_right_ventricle_size",
            "severe_left_atrium_size", "moderate_left_atrium_size", "mild_left_atrium_size",
            "severe_right_atrium_size", "moderate_right_atrium_size", "mild_right_atrium_size", "tavr"
        ]
        
        for cond in binary_conditions:
            tokenized = tokenize(zero_shot_prompts[cond])
            with torch.no_grad():
                prompt_embeds = F.normalize(echo_clip.encode_text(tokenized), dim=-1)
            pred = compute_binary_metric(test_video_embedding, prompt_embeds)
            if pred.item() > 0.28:
                report_data["detected_conditions"].append(cond)

        # Save report
        with open("report_data.json", "w") as f:
            json.dump(report_data, f)
        logger.info("Successfully generated report_data.json")
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise
