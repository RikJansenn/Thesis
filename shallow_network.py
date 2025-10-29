import random

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('tkagg')

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
folder_path = "C:/Users/rikki/Documents/Uni/Thesis/Dataset/data/01"
rpy.set_seed(44)

IP = True
TIMESTEPS_IP = 1000

# Load audio and convert to spectrograms, and store target labels
def load_training_data(folder_path):
    samples = []
    targets = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=None)

            # Create Spectrogram
            S = np.abs(librosa.stft(audio))
            S = S.T

            samples.append(S)
            digit = int(filename[0])
            targets.append(np.eye(10)[digit].reshape(1, -1))

            print(S.shape)

    return samples, targets

# Create model
def create_model():
    if IP:
        reservoir = IPReservoir(1000, mu=0.0, sigma=0.3, sr=0.95, activation="tanh", epochs=10)
    else:
        reservoir = Reservoir(500, lr=0.5, sr=0.9)
    readout = Ridge(ridge=1e-7 )
    return reservoir, readout

def pretrain_model(reservoir, X):
    stream = []
    total = 0

    while total < TIMESTEPS_IP:
        spec = random.choice(X)
        stream.append(spec)
        total += spec.shape[0]
    stream = np.concatenate(stream, axis=0)

    reservoir.fit(stream, warmup=100)
    return reservoir

def train_model(X, Y, reservoir, readout):
    # Run spectrogram through reservoir and collect final state
    final_states = []
    for x in X:
        states = reservoir.run(x)                               # (samples, timesteps, neurons)
        final_states.append(states[-1].reshape(1, -1))          # (samples, 1 (timestep), neurons)

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


if __name__ == "__main__":
    X, Y = load_training_data(folder_path)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    reservoir, readout = create_model()

    if IP:
        states_before = reservoir.run(X_test[0])
        pretrain_model(reservoir, X)
        states_after = reservoir.run(X_test[0])

        states_before = np.vstack(states_before)
        states_after = np.vstack(states_after)

    readout, final_states = train_model(X_train, Y_train, reservoir, readout)
    test_model(reservoir, readout, X_test, Y_test)
