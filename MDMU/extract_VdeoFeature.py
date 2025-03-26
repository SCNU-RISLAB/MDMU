import os
import torch
from transformers import TimesformerModel, AutoImageProcessor
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm


def load_video_frames(video_path, num_frames=64):
    """Load frames from a video file and return a list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    frames = []
    frame_id = 0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == frame_indices[idx]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image)
            idx += 1
            if idx >= len(frame_indices):
                break
        frame_id += 1
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


def extract_video_features(video_path):
    """Extract video features using a pre-trained TimeSformer model."""
    model_name = 'timesformer1'  # Update to the correct model name
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TimesformerModel.from_pretrained(model_name)

    frames = load_video_frames(video_path, num_frames=64)
    inputs = processor(frames, return_tensors="pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    sequence_length = last_hidden_state.shape[1]
    hidden_size = last_hidden_state.shape[2]
    num_frames = inputs['pixel_values'].shape[1]
    num_patches_per_frame = (sequence_length - 1) // num_frames

    patch_embeddings = last_hidden_state[:, 1:, :].view(-1, num_frames, num_patches_per_frame, hidden_size)
    frame_embeddings = patch_embeddings.mean(dim=2)

    features = frame_embeddings.cpu().numpy()
    return features

def extract_features_from_folder(raw_folder, feature_folder):
    """Extract features from all videos in the specified folder."""
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    for root, _, files in tqdm(os.walk(raw_folder)):
        print(root)
        for file in tqdm(files):
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, raw_folder)
                feature_subfolder = os.path.join(feature_folder, relative_path)

                if not os.path.exists(feature_subfolder):
                    os.makedirs(feature_subfolder)

                try:
                    features = extract_video_features(video_path)
                    feature_file_path = os.path.join(feature_subfolder, f"{os.path.splitext(file)[0]}.pt")
                    torch.save(torch.tensor(features), feature_file_path)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue  # Skip the current file and continue to the next file

    print("All features have been extracted and saved.")

if __name__ == "__main__":
    raw_folder = "data/MOSI/Raw"
    feature_folder = "data/MOSI/Video_Features"
    extract_features_from_folder(raw_folder, feature_folder)
    print("All features have been extracted and saved.")
