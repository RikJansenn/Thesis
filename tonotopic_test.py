import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input
from reservoirpy.datasets import narma
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import librosa
import os

matplotlib.use('tkagg')

# folder_path = "C:/Users/rikki/Documents/Uni/Thesis/Dataset/data/01"
folder_path = "C:/Users/rikki/Uni/data/01"
rpy.set_seed(42)

N_RESERVOIR = 500
SR = 0.9
N_FREQ = 129  # Update when updating spectrogram parameters


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
    sr = 8000
    fixed_length = 1
    n_fft = 256
    hop_length = 128
    eps = 1e-6

    # Resample audio
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Trim/pad to fixed length
    target_len = int(sr * fixed_length)
    if len(audio) < target_len:
        # pad center (or end) with zeros
        pad_len = target_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')
    else:
        audio = audio[:target_len]

    # Create Spectrogram
    S = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)).T

    # Normalize Spectrogram?
    S = S.astype(np.float32)
    mean = S.mean()
    std = S.std() + eps
    S = (S - mean) / std

    print(S.shape)

    return S


def create_tonotopic_mapping():
    neuron_positions = np.linspace(0, 1, N_RESERVOIR)
    freq_positions = np.linspace(0, 1, N_FREQ)

    tuning_width = 0.05     # Determines standard deviation/selectivity of input connections
    input_scaling = 0.5
    connectivity = 0.05     # Determines sparsity
    sigma = 0.05            # Determines neighborhood width

    ### Create input matrix ###
    W_in = np.zeros((N_RESERVOIR, N_FREQ))
    for i, pos in enumerate(neuron_positions):
        W_in[i, :] = np.exp(-0.5 * ((freq_positions - pos) / tuning_width) ** 2)
        W_in[i, :] *= np.random.uniform(0.5, 1.0) * input_scaling

    ### Create reservoir weight matrix ###
    # First create normal sparse random weights
    mask = np.random.randn(N_RESERVOIR, N_RESERVOIR) < connectivity
    W = np.random.uniform(-1, 1, (N_RESERVOIR, N_RESERVOIR)) * mask

    # Apply a gaussian weighing based on distance
    distance = np.abs(neuron_positions[:, None] - neuron_positions[None, :])  # Compute distance between each neuron
    locality = np.exp(-0.5 * (distance / sigma) ** 2)  # Compute locality weighing
    W *= locality  # Apply weighing to matrix

    # Normalize spectral radius
    eigvals = np.linalg.eigvals(W)
    W *= SR / np.max(np.abs(eigvals))

    return W_in, W


# Create model
def create_model():
    reservoir = Reservoir(N_RESERVOIR, lr=0.5, sr=SR)
    readout = Ridge(ridge=1e-7)

    W_in, W = create_tonotopic_mapping()

    reservoir.Win = W_in
    reservoir.W = W

    return reservoir, readout


def train_model(X, Y, reservoir, readout):
    # Run spectrogram through reservoir and collect final state
    final_states = []
    for x in X:
        states = reservoir.run(x)                       # (samples, timesteps, neurons)
        final_states.append(states[-1].reshape(1, -1))  # (samples, 1 (timestep), neurons)

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

def plot_weights(W):
    plt.figure(figsize=(6, 6))
    plt.imshow(W, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Weight')
    plt.title('Neuron Connection Weights (-1 to 1)')
    plt.show()

if __name__ == "__main__":
    X, Y = load_training_data(folder_path)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    reservoir, readout = create_model()

    readout, final_states = train_model(X_train, Y_train, reservoir, readout)
    plot_weights(reservoir.W)

    test_model(reservoir, readout, X_test, Y_test)
