from utils import plot_pdf, plot_waveform, plot_spectrogram, plot_weights
from biological_constraints import apply_ip, create_tonotopic_mapping

import numpy as np
import librosa
import os

folder_path = "Dataset/data"

def load_training_data(folder_path):
    samples = []
    mel_samples = []

    targets = []

    # Create label for silence
    silence_label = np.zeros(11)
    silence_label[10] = 1

    total = 0

    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)
                print(file_path)
                audio, sr = librosa.load(file_path, sr=None)

                # Create and store spectrogram
                S_mel = create_mel_spectrogram(audio, sr)
                S = create_spectrogram(audio, sr)

                mel_samples.append(S_mel)
                samples.append(S)

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

                # Append final target array
                targets.append(labels_expanded)

                total += 1
                print(total)

                # if total >= 10:
                #     break


    return samples, mel_samples, targets

def create_mel_spectrogram(audio, orig_sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    sr = 8000
    win_length = 512
    n_fft = 512
    hop_length = 128
    n_mels = 128

    plot = False

    # Resample audio
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    # Apply RMS Normalization
    audio = rms_normalize(audio)

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

    S = (S - S.min()) / (S.max() - S.min())
    # print(f"Spectrogram shape: {S.shape}")

    if plot:
        plot_waveform(audio, sr, title="After RMS")
        plot_spectrogram(S.T, sr, hop_length)

    return S

def create_spectrogram(audio, orig_sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    sr = 8000
    win_length = 256
    n_fft = 256
    hop_length = 128

    plot = False

    # Resample audio
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    # Apply RMS Normalization
    audio = rms_normalize(audio)

    # Create Spectrogram, conver to db and transpose to match expected input shape (time_steps, features)
    S = np.abs(librosa.stft(y=audio, win_length=win_length, n_fft=n_fft, hop_length=hop_length))
    S = librosa.amplitude_to_db(S, ref=np.max)
    S = S.T

    # Normalize Spectrogram
    # S = librosa.util.normalize(S)
    S = (S - S.min()) / (S.max() - S.min())
    # print(f"Spectrogram shape: {S.shape}")

    # plt.hist(S.flatten(), bins=100)
    # plt.title("Spec distribution")
    # plt.show()

    if plot:
        plot_waveform(audio, sr, title="After RMS")
        plot_spectrogram(S.T, sr, hop_length)

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

def rms_normalize(audio):
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / rms

    return audio


if __name__ == "__main__":
    specs, melspecs, targets = load_training_data(folder_path)

    specs = np.array(specs, dtype=np.float32)
    melspecs = np.array(melspecs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    np.savez(
        "dataset_train.npz",
        specs=specs,
        melspecs=melspecs,
        targets=targets
    )
