import random
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input
from reservoirpy.datasets import narma
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm

import numpy as np
import matplotlib
import librosa
import os

matplotlib.use('tkagg')
folder_path = "C:/Users/rikki/Documents/Uni/Thesis/Dataset/data/01"
# folder_path = "C:/Users/rikki/Uni/data/01"

IP = True

# Load audio and convert to spectrograms, and store target labels
def load_training_data(folder_path):
    samples = []
    targets = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=None)

            # Create Spectrogram
            S = create_spectrogram(audio, sr)

            samples.append(S)
            digit = int(filename[0])
            targets.append(np.eye(10)[digit].reshape(1, -1))

    return samples, targets


def create_spectrogram(audio, orig_sr):
    # Length to pad/trim to
    fixed_length = 1

    # Spectrogram parameters
    sr = 8000
    win_length = 256
    n_fft = 256
    hop_length = 128

    eps = 1e-6

    # Resample audio
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Trim/pad to fixed length
    target_len = int(sr * fixed_length)
    if len(audio) < target_len:
        pad_len = target_len - len(audio)

        # Choose random split to pad before and after signal
        left = np.random.randint(0, pad_len + 1)
        right = pad_len - left

        audio = np.pad(audio, (left, right), mode='constant')
    else:
        audio = audio[:target_len]

    # Create Spectrogram, transpose to match expected input shape
    S = np.abs(librosa.stft(y=audio,  win_length=win_length, n_fft=n_fft, hop_length=hop_length)).T

    # Normalize Spectrogram?
    S = librosa.util.normalize(S)

    # S = S.astype(np.float32)
    # mean = S.mean()
    # std = S.std() + eps
    # S = (S - mean) / std

    print(S.shape)

    return S

# Create model
def create_model():
    if IP:
        reservoir = IPReservoir(500, mu=0.0, sigma=0.1, sr=0.8, lr=0.9, activation="tanh", epochs=4)
    else:
        reservoir = Reservoir(500, lr=0.5, sr=0.9)
    readout = Ridge(ridge=1e-7 )
    return reservoir, readout

def pretrain_model(reservoir, X):
    stream = []
    total = 0

    while total < 1000:
        spec = random.choice(X)
        stream.append(spec)
        total += spec.shape[0]
    stream = np.concatenate(stream, axis=0)

    reservoir.fit(stream, warmup=100)

    # features = X[0].shape[1]
    # T = 1000
    #
    # multi_d_narma = np.zeros((T, features))
    #
    # for i in range(features):
    #     _, X_narma = narma(T)
    #     multi_d_narma[:, i] = X_narma.ravel()
    #
    # _ = reservoir.fit(multi_d_narma, warmup=100)

    # T = 1000
    # _, X_narma = narma(T)
    # _ = reservoir.fit(X_narma, warmup=100)
    #
    #
    # # Set input matrix to correct shape for spectrograms
    # reservoir.Win = np.random.uniform(-1, 1, (reservoir.units, 129))
    # reservoir.input_dim = 129

    return reservoir

def train_model(X, Y, reservoir, readout):
    # Run spectrogram through reservoir and collect final state
    final_states = []
    for x in X:
        states = reservoir.run(x)                               # states = (samples, timesteps, neurons)
        final_states.append(states[-1].reshape(1, -1))          # final_state = (samples, 1 (timestep), neurons)

    # Train readout on last layers of reservoirs
    readout = readout.fit(final_states, Y)

    return readout, final_states

# Test model
def test_model(reservoir, readout, X_test, Y_test):
    final_states_test = []

    for x in X_test:
        states = reservoir.run(x)
        final_states_test.append(states[-1].reshape(1, -1))

    predictions = readout.run(final_states_test)
    predictions = np.vstack(predictions)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(np.vstack(Y_test), axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")

def create_training_data():
    X, Y = load_training_data(folder_path)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    return X, Y, X_train, X_test, Y_train, Y_test

def heavyside(x):
    return 1.0 if x >= 0 else 0.0

def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den

def plot_pdf(states, sigma):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0, 16)
    for s in range(states.shape[1]):
        hist, edges = np.histogram(states[:, s], density=True, bins=200)
        points = [np.mean([edges[i], edges[i + 1]]) for i in range(len(edges) - 1)]
        ax1.scatter(points, hist, s=0.2, color="gray", alpha=0.25)
    ax1.hist(
        states.flatten(),
        density=True,
        bins=200,
        histtype="step",
        label="Global activation",
        lw=3.0,
    )
    x = np.linspace(-1.0, 1.0, 200)
    pdf = [bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in x]
    ax1.plot(x, pdf, label="Target distribution", linestyle="--", lw=3.0)
    ax1.set_xlabel("Reservoir activations")
    ax1.set_ylabel("Probability density")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, Y, X_train, X_test, Y_train, Y_test = create_training_data()

    reservoir, readout = create_model()

    if IP:
        pretrain_model(reservoir, X)

        states = reservoir.run(X_test[0])
        states = np.vstack(states)

        plot_pdf(states, 0.1)

    readout, final_states = train_model(X_train, Y_train, reservoir, readout)
    test_model(reservoir, readout, X_test, Y_test)
