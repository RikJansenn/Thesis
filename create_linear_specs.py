import numpy as np
import librosa
import os
import csv

folder_path = "preprocessed_data"

def load_training_data(folder_path):
    samples = []
    targets = []

    with open("used_files_for_IP.csv", "r", newline="") as f:
        reader = csv.reader(f)
        files_used_for_IP = [row[0] for row in reader]

    total = 0

    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)

                if file_path in files_used_for_IP:
                    continue

                print(file_path)
                audio, sr = librosa.load(file_path, sr=None)

                # Create and store spectrogram
                S = create_linear_spectrogram(audio, sr)

                samples.append(S)

                # Create lables per timestep for each spectrogram
                label = create_label(S, filename)
                targets.append(label)

                total += 1

                # if total >= 10:
                #     break

    return samples, targets

def create_label(S, filename):
    # Create label for silence
    silence_label = np.zeros(11)
    silence_label[10] = 1

    # Create one-hot vector representing digit
    digit = int(filename[0])
    label = np.eye(11)[digit].reshape(1, -1)

    # Create silence and digit labels per timestep
    time_steps = S.shape[0]
    labels_expanded = np.zeros((time_steps, 11))
    for t in range(time_steps):
        if np.all(S[t] == 0):
            labels_expanded[t] = silence_label
        else:
            labels_expanded[t] = label

    return labels_expanded

def create_linear_spectrogram(audio, sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    win_length = 256
    n_fft = 256
    hop_length = 128

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    # Create Spectrogram, conver to db and transpose to match expected input shape (time_steps, features)
    S = np.abs(librosa.stft(y=audio, win_length=win_length, n_fft=n_fft, hop_length=hop_length))
    S = librosa.amplitude_to_db(S, ref=np.max)
    S = S.T

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    print(f"Linear shape: {S.shape}")

    return S

def trim_or_pad(audio, sr, fixed_length):
    target_len = int(sr * fixed_length)
    if len(audio) < target_len:
        pad_len = target_len - len(audio)

        # Choose random split to pad before and after signal
        left = np.random.randint(0, pad_len + 1)
        right = pad_len - left

        audio = np.pad(audio, (left, right), mode='constant')  # pad
    else:
        audio = audio[:target_len]  # trim

    return audio


if __name__ == "__main__":
    specs, targets = load_training_data(folder_path)

    np.savez("datasets/specs.npz", specs=specs, targets=targets)
