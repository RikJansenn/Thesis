import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input
import numpy as np
import matplotlib
import librosa
import os

matplotlib.use('tkagg')
folder_path = "C:/Users/rikki/Documents/Audacity"
rpy.set_seed(44)

# Load audio and convert to spectrograms
def load_audio(folder_path):
    samples = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=None)
            # Create Spectrogram
            S = np.abs(librosa.stft(audio))
            S = S.T

            samples.append(S)
    return samples

def create_training_data(samples):
    X = samples                                                 # (samples, timesteps, features)

    # Create training targets
    Y = []
    for i in range(len(X)):
        Y.append(np.eye(3)[i].reshape(1, -1))                   # (samples, timesteps, outputs)

    return X, Y

# Create model
def create_model():
    reservoir = Reservoir(500, lr=0.5, sr=0.9)
    readout = Ridge(ridge=1e-7)
    return reservoir, readout

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
def test_model(readout, test_data):
    return readout.run(test_data)

if __name__ == "__main__":
    samples = load_audio(folder_path)
    X, Y = create_training_data(samples)
    reservoir, readout = create_model()
    readout, final_states = train_model(X, Y, reservoir, readout)
    output = test_model(readout, final_states[0])
    print(output)


