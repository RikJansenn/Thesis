import numpy as np
from reservoirpy.datasets import narma
import random

def apply_ip(reservoir, X):
    ### Using spectrograms ###
    # stream = []
    # total = 0
    #
    # while total < 1000:
    #     spec = random.choice(X)
    #     stream.append(spec)
    #     total += spec.shape[0]
    # stream = np.concatenate(stream, axis=0)
    #
    # reservoir.fit(stream, warmup=100)

    ### Multi-band narma ###
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

    ### Single band narma ###
    T = 1000
    _, X_narma = narma(T)
    _ = reservoir.fit(X_narma, warmup=100)

    # Set input matrix to correct shape for spectrograms
    reservoir.Win = np.random.uniform(-1, 1, (reservoir.units, 129))
    reservoir.input_dim = 129

    return reservoir

def create_tonotopic_mapping(N_reservoir, sr, N_freq=129):
    neuron_positions = np.linspace(0, 1, N_reservoir)
    freq_positions = np.linspace(0, 1, N_freq)

    tuning_width = 0.05     # Determines standard deviation/selectivity of input connections
    input_scaling = 1
    connectivity = 0.05     # Determines sparsity
    sigma = 0.05            # Determines neighborhood width

    ### Create input matrix ###
    W_in = np.zeros((N_reservoir, N_freq))
    for i, pos in enumerate(neuron_positions):
        W_in[i, :] = np.exp(-0.5 * ((freq_positions - pos) / tuning_width) ** 2)
        W_in[i, :] *= np.random.uniform(0.5, 1.0) * input_scaling

    ### Create reservoir weight matrix ###
    # First create normal sparse random weights
    mask = np.random.randn(N_reservoir, N_reservoir) < connectivity
    W = np.random.uniform(-1, 1, (N_reservoir, N_reservoir)) * mask

    # Apply a gaussian weighing based on distance
    distance = np.abs(neuron_positions[:, None] - neuron_positions[None, :])  # Compute distance between each neuron
    locality = np.exp(-0.5 * (distance / sigma) ** 2)  # Compute locality weighing
    W *= locality  # Apply weighing to matrix

    # Normalize spectral radius
    eigvals = np.linalg.eigvals(W)
    W *= sr / np.max(np.abs(eigvals))

    return W_in, W