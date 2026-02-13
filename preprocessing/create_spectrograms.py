import pycochleagram.cochleagram as cgram
import numpy as np
import librosa
import os
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

folder_path = "../preprocessed_data"


def load_training_data(folder_path):
    samples = []
    mel_samples = []
    cochlea_samples = []

    targets_linear = []
    targets_mel = []
    targets_cochlea = []

    total = 0

    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)

                print(file_path)
                audio, sr = librosa.load(file_path, sr=None)

                # Create and store spectrogram
                S = create_spectrogram(audio, sr)
                S_mel = create_mel_spectrogram(audio, sr)
                S_coch = create_cochleagram(audio, sr)

                samples.append(S)
                mel_samples.append(S_mel)
                cochlea_samples.append(S_coch)

                # Create lables per timestep for each spectrogram
                label_linear = create_label(S, filename)
                targets_linear.append(label_linear)
                label_mel = create_label(S_mel, filename)
                targets_mel.append(label_mel)
                label_cochlea = create_label(S_coch, filename)
                targets_cochlea.append(label_cochlea)

                total += 1

                # if total >= 10:
                #     break

    return samples, mel_samples, cochlea_samples, targets_linear, targets_mel, targets_cochlea


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


def create_spectrogram(audio, sr):
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


def create_mel_spectrogram(audio, sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    win_length = 512
    n_fft = 512
    hop_length = 128
    n_mels = 128

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

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
    print(f"Mel shape: {S.shape}")

    return S


def create_cochleagram(audio, sr, sample_factor=2, nonlinearity="db"):
    # Length to pad/trim to
    fixed_length = 1
    n = 64

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    S = cgram.human_cochleagram(audio,
                                sr,
                                n=n,
                                downsample=downsampler,
                                sample_factor=sample_factor,
                                nonlinearity=nonlinearity,
                                strict=False).T

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    print(f"Coch shape: {S.shape}")

    return S


def downsampler(envs):
    return cgram.apply_envelope_downsample(
        envs,
        mode='poly',
        audio_sr=8000,
        env_sr=64  # Amount of timesteps
    )


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
    specs, melspecs, cochs, targets_linear, targets_mel, targets_cochlea = load_training_data(folder_path)

    specs = np.array(specs, dtype=np.float32)
    melspecs = np.array(melspecs, dtype=np.float32)
    cochs = np.array(cochs, dtype=np.float32)
    targets_linear = np.array(targets_linear, dtype=np.float32)
    targets_mel = np.array(targets_mel, dtype=np.float32)
    targets_cochlea = np.array(targets_cochlea, dtype=np.float32)

    np.savez(
        "../datasets/dataset_train_all.npz",
        specs=specs,
        melspecs=melspecs,
        cochs=cochs,
        targets_linear=targets_linear,
        targets_mel=targets_mel,
        targets_cochlea=targets_cochlea
    )
