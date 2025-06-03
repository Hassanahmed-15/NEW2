import os
import json
import pydicom
import cv2
import torch
import torch.nn.functional as F
from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T

from utils import zero_shot_prompts, read_avi, compute_regression_metric, compute_binary_metric

def process_dicom(dicom_path, output_video_path):
    print(f"Processing DICOM file: {dicom_path}")
    dicom_data = pydicom.dcmread(dicom_path)

    number_of_frames = dicom_data.get("NumberOfFrames", 1)
    if isinstance(number_of_frames, (pydicom.valuerep.IS, str)):
        number_of_frames = int(number_of_frames)
    print(f"Number of Frames: {number_of_frames}")

    pixel_data = dicom_data.pixel_array
    print(f"Original pixel data shape: {pixel_data.shape}")
    pixel_data = pixel_data[:, 110:-40, 150:-150, :]

    frame_time = dicom_data.get("FrameTime", 40)
    if isinstance(frame_time, (pydicom.valuerep.DSfloat, str)):
        frame_time = float(frame_time)
    fps = 1000 / frame_time
    print(f"Frame time: {frame_time}, FPS: {fps}")

    frame_size = (224, 224)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    print(f"Video writer initialized with size: {frame_size}")

    if number_of_frames > 1:
        for frame_index in range(number_of_frames):
            frame = pixel_data[frame_index]
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_rescaled = cv2.resize(frame, frame_size)
            video_writer.write(frame_rescaled)
    else:
        frame = pixel_data
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rescaled = cv2.resize(frame, frame_size)
        video_writer.write(frame_rescaled)

    video_writer.release()
    print(f"Video saved at {output_video_path}")

def process_video(video_path):
    print(f"Processing video file: {video_path}")
    echo_clip, _, preprocess_val = create_model_and_transforms("hf-hub:mkaichristensen/echo-clip", precision="fp32", device="cpu")

    test_video = read_avi(video_path, (224, 224))
    test_video = torch.stack([preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0)
    test_video = test_video[0:min(40, len(test_video)):2]
    test_video = test_video.to(torch.float32)

    test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1)
    test_video_embedding = test_video_embedding.unsqueeze(0)

    def compute_prompt_similarities(video_embeddings, prompt_embeddings):
        per_frame_similarities = video_embeddings @ prompt_embeddings.T
        predictions = per_frame_similarities.mean(dim=1)
        return predictions

    def find_top_n_matching_prompts(video_embedding, prompt_key, top_n=3):
        prompts = zero_shot_prompts[prompt_key]
        tokenized_prompts = tokenize(prompts)
        with torch.no_grad():
            prompt_embeddings = F.normalize(echo_clip.encode_text(tokenized_prompts), dim=-1)
        similarities = compute_prompt_similarities(video_embedding, prompt_embeddings)
        top_n = min(top_n, len(prompts))
        top_n_indices = similarities.topk(top_n, largest=True).indices.cpu().numpy().flatten()
        top_n_similarities = similarities.topk(top_n, largest=True).values.detach().cpu().numpy().flatten()
        top_n_prompts = [prompts[idx] for idx in top_n_indices]
        return list(zip(top_n_prompts, top_n_similarities))

    def find_top_cont_prompts(t_video_embedding, key, value_range):
        prompts = zero_shot_prompts[key]
        gen_prompts = []
        prompt_values = []
        for prompt in prompts:
            for i in range(value_range):
                gen_prompts.append(prompt.replace("<#>", str(i)))
                prompt_values.append(i)
        tokenized = tokenize(gen_prompts)
        with torch.no_grad():
            encoded = echo_clip.encode_text(tokenized)
            norm_prompts = F.normalize(encoded, dim=-1)
        predicted_value = compute_regression_metric(t_video_embedding, norm_prompts, prompt_values)
        closest_prompt_index = (torch.abs(torch.tensor(prompt_values) - predicted_value.item())).argmin()
        closest_prompt_original = zero_shot_prompts[key][0].replace("<#>", str(closest_prompt_index.item()))
        return {
            "predicted_value": float(predicted_value.item()),
            "closest_prompt": closest_prompt_original
        }

    report_data = {
        "mitraclip": find_top_n_matching_prompts(test_video_embedding, "mitraclip", top_n=2),
        "impella": find_top_n_matching_prompts(test_video_embedding, "impella", top_n=2),
        "severe_lv_dilation": find_top_n_matching_prompts(test_video_embedding, "severe_left_ventricle_dilation", top_n=2),
        "moderate_lv_dilation": find_top_n_matching_prompts(test_video_embedding, "moderate_left_ventricle_dilation", top_n=2),
        "mild_lv_dilation": find_top_n_matching_prompts(test_video_embedding, "mild_left_ventricle_dilation", top_n=2),
        "sig_elev_ra_pressure": find_top_n_matching_prompts(test_video_embedding, "significantly_elevated_right_atrial_pressure", top_n=1),
        "norm_r_atrial_pressure": find_top_n_matching_prompts(test_video_embedding, "normal_right_atrial_pressure", top_n=1),
        "detected_conditions": []
    }

    bins = [
        "severe_right_ventricle_size", "moderate_right_ventricle_size", "mild_right_ventricle_size",
        "severe_left_atrium_size", "moderate_left_atrium_size", "mild_left_atrium_size",
        "severe_right_atrium_size", "moderate_right_atrium_size", "mild_right_atrium_size", "tavr"
    ]
    for b in bins:
        bin_prompt = zero_shot_prompts[b]
        tokenized = tokenize(bin_prompt)
        with torch.no_grad():
            prompt_embeddings = F.normalize(echo_clip.encode_text(tokenized), dim=-1)
        prediction = compute_binary_metric(test_video_embedding, prompt_embeddings)
        if prediction.item() > 0.28:
            report_data["detected_conditions"].append(b)

    report_data["ejection_fraction"] = find_top_cont_prompts(test_video_embedding, 'ejection_fraction', 101)
    report_data["pulmonary_artery_pressure"] = find_top_cont_prompts(test_video_embedding, 'pulmonary_artery_pressure', 17)

    for key in [
        "mitraclip", "impella", "severe_lv_dilation", "moderate_lv_dilation", "mild_lv_dilation",
        "sig_elev_ra_pressure", "norm_r_atrial_pressure"]:
        report_data[key] = [[prompt, float(sim)] for prompt, sim in report_data[key]]

    with open("report_data.json", "w") as outfile:
        json.dump(report_data, outfile)
    print("report_data.json written")
