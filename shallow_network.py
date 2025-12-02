import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm, mode

from utils import plot_pdf, plot_waveform, plot_spectrogram, plot_weights
from biological_constraints import apply_ip, create_tonotopic_mapping

import numpy as np
import librosa
import os

# folder_path = "C:/Users/rikki/Documents/Uni/Thesis/Dataset/data/01"
folder_path = "C:/Users/rikki/Uni/data/01"

IP = True
TONOTOPIC = False

def load_training_data(folder_path):
    samples = []
    targets = []

    # Create label for silence
    silence_label = np.zeros(11)
    silence_label[10] = 1

    test = True

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=None)

            # Create and store spectrogram
            S = create_spectrogram(audio, sr)
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

    return samples, targets

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

    # Create Spectrogram, transpose to match expected input shape (time_steps, features)
    S = np.abs(librosa.stft(y=audio, win_length=win_length, n_fft=n_fft, hop_length=hop_length)).T

    # Normalize Spectrogram
    S = librosa.util.normalize(S)
    print(f"Shape: {S.shape}")

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
    print(f"RMS before: {rms}")

    audio = audio / rms

    rms = np.sqrt(np.mean(audio ** 2))
    print(f"RMS after: {rms}")
    return audio

def create_model(N, sr, lr):
    if IP:
        reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=0.1, activation="tanh", epochs=4)
    else:
        reservoir = Reservoir(500, sr=sr, lr=lr)
    readout = Ridge(ridge=1e-6, output_dim=11)

    return reservoir, readout

def train_model(X, Y, reservoir, readout):
    X_list = []
    Y_list = []

    total = len(X)
    i = 1

    for spec, labels in zip(X, Y):
        print(f"{i} out of {total}")

        states = reservoir.run(spec)

        X_list.append(states)
        Y_list.append(labels)

        i += 1

    X_all = np.vstack(X_list)  # shape = (sum_T, units)
    Y_all = np.vstack(Y_list)  # shape = (sum_T, 10)

    readout.fit(X_all, Y_all)

    return readout

def test_model(reservoir, readout, X_test, Y_test):
    y_pred = []

    for spec in X_test:
        states = reservoir.run(spec)
        predictions = readout.run(states)                               # Get raw prediction per timestep
        pred_per_timestep = np.argmax(predictions, axis=1)              # Get one-hot winner at each timestep
        non_silence_preds = pred_per_timestep[pred_per_timestep != 10]  # Remove silence as category
        final_pred = mode(non_silence_preds, keepdims=False).mode       # Get winning digit with majority voting

        y_pred.append(final_pred)

    # Take the middle one-hot vectors as digits, as there is never silence there
    y_true = np.array([np.argmax(Y[len(Y)//2]) for Y in Y_test])

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")

def create_training_data():
    X, Y = load_training_data(folder_path)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    return X, Y, X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    N = 500
    sr = 0.8
    lr = 1

    X, Y, X_train, X_test, Y_train, Y_test = create_training_data()

    reservoir, readout = create_model(N, sr, lr)

    if TONOTOPIC:
        W_in, W = create_tonotopic_mapping(N, sr)

        reservoir.Win = W_in
        reservoir.W = W

    if IP:
        apply_ip(reservoir, X)

        states = reservoir.run(X_test[0])
        states = np.vstack(states)

        plot_pdf(states, 0.1, title="IP on single band narma")

    readout = train_model(X_train, Y_train, reservoir, readout)

    # plot_weights(reservoir.W)

    test_model(reservoir, readout, X_test, Y_test)
