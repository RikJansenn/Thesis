import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import librosa
import os

rpy.set_seed(44)

# Load audio
folder_path = "C:/Users/rikki/Documents/Audacity"
audio_series_list = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        audio, sr = librosa.load(file_path, sr=None)
        audio_series = audio.reshape(-1, 1)
        audio_series_list.append(audio_series)

# Create training stuff
X = audio_series_list                                       # (samples, timesteps, features)     right now: (3, T, 1)

Y = []
for i in range(len(X)):
    Y.append(np.eye(3)[i].reshape(1, -1))                   # (samples, timesteps, outputs)      right now: (3, 1, 3)

print(Y)

# Create model
data = Input()
reservoir = Reservoir(500, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
esn_model = data >> reservoir >> readout

# Run timeseries through reservoir
final_states = []
for x in X:
    states = reservoir.run(x)                               # (samples, timesteps, neurons)      right now: (3, T, 500)
    final_states.append(states[-1].reshape(1, -1))          # (samples, 1, neurons)              right now: (3, 1, 500) (just last timestep)


# Train  readout on last layer of reservoir
readout = readout.fit(final_states, Y)

# Prediction of the model based on last layer of reservoir
print(final_states[0])
output = readout.run(final_states[1])

print(output)


