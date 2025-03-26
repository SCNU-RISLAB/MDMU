# -*- coding: gbk -*-
import os
import torch
import torchaudio
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from tqdm import tqdm
import numpy as np

def extract_audio_features(audio_path, processor, model, target_seq_length=96):
    """
    Use the data2vec-audio to extract audio features from a given audio file
    and adjust the sequence length to targets_seq length.    """
    sound, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        sound = resampler(sound)
    if sound.shape[0] > 1:
        sound = torch.mean(sound, dim=0)
    else:
        sound = sound.squeeze(0)
    sound = sound.numpy()

    inputs = processor(sound, sampling_rate=16000, return_tensors="pt", padding=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state.squeeze(0)  # 形状：(序列长度, 特征维度)

    # Adjust the length of the feature sequence to target_seq_length
    current_seq_length = last_hidden_state.shape[0]
    if current_seq_length > target_seq_length:
        last_hidden_state = torch.nn.functional.interpolate(
            last_hidden_state.unsqueeze(0).permute(0, 2, 1),  # 形状：(1, 特征维度, 序列长度)
            size=target_seq_length,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1).squeeze(0)
    elif current_seq_length < target_seq_length:
        last_hidden_state = torch.nn.functional.interpolate(
            last_hidden_state.unsqueeze(0).permute(0, 2, 1),  # 形状：(1, 特征维度, 序列长度)
            size=target_seq_length,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1).squeeze(0)

    return last_hidden_state.cpu()

def process_audio_files(audio_root, feature_root):
    """
    Process all audio files under audio_root and save features to feature_root
    """
    processor = Wav2Vec2Processor.from_pretrained('data2vec-audio-large')
    model = Data2VecAudioModel.from_pretrained('data2vec-audio-large')

    for root, dirs, files in os.walk(audio_root):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, audio_root)
                feature_subdir = os.path.join(feature_root, relative_path)
                os.makedirs(feature_subdir, exist_ok=True)
                feature_file = os.path.join(feature_subdir, file.replace('.wav', '.pt'))

                try:
                    features = extract_audio_features(audio_path, processor, model, target_seq_length=96)
                    torch.save(features, feature_file)
                except Exception as e:
                    print(f"Error occurred while processing file {audio_math}: {e}")

if __name__ == "__main__":
    audio_root = 'data/MOSI/wav'
    feature_root = 'data/MOSI/Audio_Features_data2vec'
    os.makedirs(feature_root, exist_ok=True)
    process_audio_files(audio_root, feature_root)
    print("Audio features have been extracted and saved.")
