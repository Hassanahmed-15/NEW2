from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import json
import os
import torch
import torch.nn.functional as F
from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T
from utils import zero_shot_prompts, read_avi, compute_regression_metric, compute_binary_metric
from weasyprint import HTML
import pydicom
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'avi', 'dcm'}
app.config['REPORT_FILE'] = os.path.join(app.root_path, 'report_data.json')
app.config['PDF_FILE'] = os.path.join(app.root_path, 'final_report.pdf')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def ensure_report_file():
    if not os.path.exists(app.config['REPORT_FILE']):
        default_report_data = {
            'summary': [],
            'mitraclip': [],
            'impella': [],
            'severe_lv_dilation': [],
            'moderate_lv_dilation': [],
            'mild_lv_dilation': [],
            'sig_elev_ra_pressure': [],
            'norm_r_atrial_pressure': [],
            'detected_conditions': [],
            'ejection_fraction': {"predicted_value": None, "closest_prompt": None},
            'pulmonary_artery_pressure': {"predicted_value": None, "closest_prompt": None}
        }
        with open(app.config['REPORT_FILE'], 'w') as outfile:
            json.dump(default_report_data, outfile)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_dicom_file(file_path):
    """Check if a file is a DICOM file by attempting to read it with pydicom."""
    try:
        pydicom.dcmread(file_path)
        return True
    except pydicom.errors.InvalidDicomError:
        return False

def process_dicom(dicom_path, output_video_path):
    app.logger.info(f'Processing DICOM file: {dicom_path}')
    dicom_data = pydicom.dcmread(dicom_path)

    number_of_frames = dicom_data.get("NumberOfFrames", 1)
    if isinstance(number_of_frames, (pydicom.valuerep.IS, str)):
        number_of_frames = int(number_of_frames)
    app.logger.info(f'Number of Frames: {number_of_frames}')

    pixel_data = dicom_data.pixel_array
    app.logger.info(f'Original pixel data shape: {pixel_data.shape}')
    pixel_data = pixel_data[:, 110:-40, 150:-150, :]

    frame_time = dicom_data.get('FrameTime', 40)
    if isinstance(frame_time, (pydicom.valuerep.DSfloat, str)):
        frame_time = float(frame_time)
    fps = 1000 / frame_time
    app.logger.info(f'Frame time: {frame_time}, FPS: {fps}')

    frame_shape = pixel_data[0].shape if number_of_frames > 1 else pixel_data.shape
    frame_size = (224, 224)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    app.logger.info(f'Video writer initialized with size: {frame_size}')

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
    app.logger.info(f'Video saved at {output_video_path}')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        if allowed_file(file.filename):
            if file.filename.rsplit('.', 1)[1].lower() == 'avi':
                video_path = filename
            else:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted_video.avi')
                process_dicom(filename, video_path)
        else:
            if is_dicom_file(filename):
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted_video.avi')
                process_dicom(filename, video_path)
            else:
                return "Unsupported file type", 400
        process_video(video_path)
        return redirect(url_for('report'))
    return render_template('upload.html')

@app.route('/save_dropdown', methods=['POST'])
def save_dropdown():
    dropdown_data = request.form.to_dict()
    key = list(dropdown_data.keys())[0]
    selected_option = dropdown_data[key]

    with open(app.config['REPORT_FILE'], 'r') as infile:
        report_data = json.load(infile)
    
    report_data[key] = [[selected_option, 0]]
    
    with open(app.config['REPORT_FILE'], 'w') as outfile:
        json.dump(report_data, outfile)
    
    return jsonify({'status': 'success'})

@app.route('/report')
def report():
    with open(app.config['REPORT_FILE'], 'r') as infile:
        report_data = json.load(infile)

    app.logger.info(f"Report Data: {report_data}")

    for key in ['mitraclip', 'impella', 'severe_lv_dilation', 'moderate_lv_dilation', 'mild_lv_dilation', 'sig_elev_ra_pressure', 'norm_r_atrial_pressure']:
        report_data[key] = [item[0] for item in report_data[key]]
    return render_template('report_form.html', data=report_data)

@app.route('/edit', methods=['GET', 'POST'])
def edit_report():
    ensure_report_file()
    if request.method == 'GET':
        with open(app.config['REPORT_FILE'], 'r') as infile:
            report_data = json.load(infile)

        app.logger.info(f"Edit Report Data Before Processing: {report_data}")

        default_keys = {
            'summary': [],
            'mitraclip': [],
            'impella': [],
            'severe_lv_dilation': [],
            'moderate_lv_dilation': [],
            'mild_lv_dilation': [],
            'sig_elev_ra_pressure': [],
            'norm_r_atrial_pressure': [],
            'detected_conditions': [],
            'ejection_fraction': {"predicted_value": None, "closest_prompt": None},
            'pulmonary_artery_pressure': {"predicted_value": None, "closest_prompt": None}
        }
        for key, default_value in default_keys.items():
            if key not in report_data or report_data[key] is None:
                report_data[key] = default_value

        app.logger.info(f"Edit Report Data After Processing: {report_data}")
        return render_template('edit_report.html', data=report_data)
    elif request.method == 'POST':
        return save_report()

@app.route('/save', methods=['GET','POST'])
def save_report():
    ensure_report_file()
    with open(app.config['REPORT_FILE'], 'r') as infile:
        report_data = json.load(infile)

    form_data = request.form.to_dict()

    app.logger.info(f"Form Data: {form_data}")

    def safe_load_json(data, default):
        try:
            return json.loads(data) if data else default
        except json.JSONDecodeError:
            return default

    report_data['summary'] = form_data.get('summary', '').split('\n')
    report_data['mitraclip'] = [[item, 0] for item in form_data.get('mitraclip', '').split('\n') if item]
    report_data['impella'] = [[item, 0] for item in form_data.get('impella', '').split('\n') if item]
    report_data['severe_lv_dilation'] = [[item, 0] for item in form_data.get('severe_lv_dilation', '').split('\n') if item]
    report_data['moderate_lv_dilation'] = [[item, 0] for item in form_data.get('moderate_lv_dilation', '').split('\n') if item]
    report_data['mild_lv_dilation'] = [[item, 0] for item in form_data.get('mild_lv_dilation', '').split('\n') if item]
    report_data['sig_elev_ra_pressure'] = [[item, 0] for item in form_data.get('sig_elev_ra_pressure', '').split('\n') if item]
    report_data['norm_r_atrial_pressure'] = [[item, 0] for item in form_data.get('norm_r_atrial_pressure', '').split('\n') if item]
    report_data['detected_conditions'] = form_data.get('detected_conditions', '').split('\n')
    report_data['ejection_fraction'] = safe_load_json(form_data.get('ejection_fraction', '{}'), {"predicted_value": None, "closest_prompt": None})
    report_data['pulmonary_artery_pressure'] = safe_load_json(form_data.get('pulmonary_artery_pressure', '{}'), {"predicted_value": None, "closest_prompt": None})

    report_data['patient_name'] = form_data.get('patient_name', report_data.get('patient_name', ''))
    report_data['study_date'] = form_data.get('study_date', report_data.get('study_date', ''))
    report_data['patient_id'] = form_data.get('patient_id', report_data.get('patient_id', ''))
    report_data['birth_date'] = form_data.get('birth_date', report_data.get('birth_date', ''))
    report_data['gender'] = form_data.get('gender', report_data.get('gender', ''))
    report_data['height'] = form_data.get('height', report_data.get('height', ''))
    report_data['weight'] = form_data.get('weight', report_data.get('weight', ''))
    report_data['ordering_physician'] = form_data.get('ordering_physician', report_data.get('ordering_physician', ''))
    report_data['performed_by'] = form_data.get('performed_by', report_data.get('performed_by', ''))
    report_data['indication'] = form_data.get('indication', report_data.get('indication', ''))
    report_data['bsa'] = form_data.get('bsa', report_data.get('bsa', ''))

    app.logger.info(f"Updated Report Data: {report_data}")

    with open(app.config['REPORT_FILE'], 'w') as outfile:
        json.dump(report_data, outfile)

    if 'save_and_download' in request.form:
        return redirect(url_for('download_final_pdf'))
    elif 'go_to_final' in request.form:
        return redirect(url_for('final_report'))
    return redirect(url_for('final_report'))

@app.route('/download')
def download_report():
    return send_file(app.config['REPORT_FILE'], as_attachment=True)

@app.route('/final_report')
def final_report():
    ensure_report_file()
    with open(app.config['REPORT_FILE'], 'r') as infile:
        report_data = json.load(infile)

    app.logger.info(f"Final Report Data: {report_data}")
    return render_template('final_report.html', data=report_data)

@app.route('/download_final_pdf', methods=['GET', 'POST'])
def download_final_pdf():
    with open(app.config['REPORT_FILE'], 'r') as infile:
        report_data = json.load(infile)

    rendered = render_template('final_report_view.html', data=report_data)
    base_url = request.url_root
    pdf_path = os.path.join(app.root_path, 'final_report.pdf')
    HTML(string=rendered, base_url=base_url).write_pdf(pdf_path)

    return send_file(pdf_path, as_attachment=True, download_name='final_report.pdf')

def process_video(video_path):
    app.logger.info(f'Processing video file: {video_path}')
    # Use CPU instead of CUDA
    echo_clip, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip",
        precision="fp32",
        device="cpu"
    )

    # Read & preprocess video frames
    test_video = read_avi(video_path, (224, 224))
    test_video = torch.stack([preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0)

    # Sample up to 40 frames, every 2nd frame; keep on CPU
    test_video = test_video[0:min(40, len(test_video)):2]
    test_video = test_video.to(torch.float32)

    # Compute video embedding on CPU
    with torch.no_grad():
        image_embeddings = echo_clip.encode_image(test_video)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
    test_video_embedding = image_embeddings.unsqueeze(0)

    def compute_prompt_similarities(video_embeddings, prompt_embeddings):
        per_frame_similarities = video_embeddings @ prompt_embeddings.T
        predictions = per_frame_similarities.mean(dim=1)
        return predictions

    def find_top_n_matching_prompts(video_embedding, prompt_key, top_n=3):
        prompts = zero_shot_prompts[prompt_key]
        tokenized_prompts = tokenize(prompts)
        with torch.no_grad():
            prompt_embeddings = echo_clip.encode_text(tokenized_prompts)
            prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        similarities = compute_prompt_similarities(video_embedding, prompt_embeddings)
        top_n = min(top_n, len(prompts))
        topk = similarities.topk(top_n, largest=True)
        top_n_indices = topk.indices.cpu().numpy().flatten()
        top_n_similarities = topk.values.cpu().numpy().flatten()
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
        closest_prompt_value = prompt_values[closest_prompt_index]
        # Reconstruct original prompt template
        original_template = zero_shot_prompts[key][0]
        closest_prompt_text = original_template.replace("<#>", str(closest_prompt_index))
        return {
            "predicted_value": float(predicted_value.item()),
            "closest_prompt": closest_prompt_text
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
        tokenized_bin = tokenize(bin_prompt)
        with torch.no_grad():
            prompt_embeddings = echo_clip.encode_text(tokenized_bin)
            prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
            prediction = compute_binary_metric(test_video_embedding, prompt_embeddings)
        f1_calibrated_threshold = 0.28
        if prediction.item() > f1_calibrated_threshold:
            report_data["detected_conditions"].append(b)

    report_data["ejection_fraction"] = find_top_cont_prompts(test_video_embedding, 'ejection_fraction', 101)
    report_data["pulmonary_artery_pressure"] = find_top_cont_prompts(test_video_embedding, 'pulmonary_artery_pressure', 17)

    # Convert lists of tuples to lists of [prompt, score]
    for key in [
        "mitraclip", "impella", "severe_lv_dilation", "moderate_lv_dilation",
        "mild_lv_dilation", "sig_elev_ra_pressure", "norm_r_atrial_pressure"
    ]:
        report_data[key] = [[prompt, float(similarity)] for prompt, similarity in report_data[key]]

    with open(app.config['REPORT_FILE'], 'w') as outfile:
        json.dump(report_data, outfile)

if __name__ == '__main__':
    app.run(debug=True)
