import os
import random
import numpy as np
import librosa
import csv

base_dir = "../../preprocessed_data"
target_timesteps = 1000
sr = 8000

def create_mel_spectrogram(audio, sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    win_length = 512
    n_fft = 512
    hop_length = 128
    n_mels = 128

    # Create Spectrogram, convert to db and transpose to match expected input shape (time_steps, features)
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels
    )
    S = librosa.power_to_db(S, ref=np.max)
    S = S.T

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    return S

digit = 0
timesteps = 0
spectrograms = []
amount = 0
used_files_paths = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    print(folder_name)

    # Find correct digit and make spectrogram from it
    for filename in os.listdir(folder_path):
        if int(filename[0]) == digit:
            file_path = os.path.join(base_dir, folder_name, filename)
            audio, sr = librosa.load(file_path, sr=None)
            used_files_paths.append(file_path)

            # Create and store spectrogram
            S = create_mel_spectrogram(audio, sr)
            spectrograms.append(S)

            timesteps += S.shape[0]

            print(filename)
            break

    print(timesteps)

    if timesteps >= target_timesteps:
        break

    if digit < 9:
        digit += 1
    elif digit == 9:
        digit = 0

    amount += 1

print(amount)

with open("used_files_for_IP.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for item in used_files_paths:
        writer.writerow([item])

spectrograms = np.concatenate(spectrograms)
np.save("../../datasets/dataset_IP.npy", spectrograms)
