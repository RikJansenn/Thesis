import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from reservoirpy.datasets import narma
from scipy.stats import pearsonr
from scipy.stats import entropy, norm
import numpy as np
import matplotlib
import librosa
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
import pandas as pd
from utils import plot_pdf
from biological_constraints import apply_ip_specs
matplotlib.use('tkAgg')

rpy.set_seed(42)

def heavyside(x):
    return 1.0 if x >= 0 else 0.0

def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den

def get_KL_divergence_and_entropy(states, sigma):
    # Get all state activations and their min and max
    all_activations = states.flatten()
    x_min = all_activations.min()
    x_max = all_activations.max()

    # Estimate PDF with a histogram from all activations
    hist, edges = np.histogram(all_activations, density=True, bins=200, range=(x_min, x_max))

    # Use bin centers, so estimated PDF and target PDF are aligned
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Target PDF
    pdf = np.array([bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in bin_centers])

    kl = entropy(hist, pdf)
    ent = entropy(hist)

    return kl, ent

def get_KL_divergence_and_entropy_per_neuron(states, sigma):
    kls = []
    ents = []

    for state in states:
        x_min = state.min()
        x_max = state.max()

        # Estimate PDF with a histogram from all activations
        hist, edges = np.histogram(state, density=True, bins=200, range=(x_min, x_max))

        # Use bin centers, so estimated PDF and target PDF are aligned
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Target PDF
        pdf = np.array([bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in bin_centers])

        kls.append(entropy(hist, pdf))
        # ents.append(entropy(hist))

    return np.mean(kls)


# Create model
def create_model(N, lr, sr, sigma, epochs=10):
    reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=epochs, learning_rate=2e-4)

    return reservoir

if __name__ == "__main__":
    data_len = 1000
    iterations = 5

    results = []

    # Parameters to test
    N_values = [600, 700, 800, 900, 1000, 1100, 1200]       # 7
    lr_values = [0.85, 0.88, 0.91, 0.94, 0.97, 1]    # 6
    sr_values = [0.8, 0.9, 1, 1.1, 1.2]      # 5
    sigma = [0.1, 0.2, 0.3]

    print(lr_values)

    # Testing amount of epochs
    X = np.load("datasets/dataset_IP.npy")
    X_test = np.load("datasets/IP_testset.npz")["melspecs"]

    for epochs in range(1, 8):
        kls = []
        for i in range(iterations):
            reservoir = create_model(N=500, lr=1, sr=0.8, sigma=0.1, epochs=epochs)

            _ = reservoir.fit(X, warmup=100)
            states = reservoir.run(X_test[i])

            kl, ent = get_KL_divergence_and_entropy(states, sigma)
            kls.append(kl)

        print(np.mean(kls))
        results.append({
            "epochs": epochs,
            "KL_mean": np.mean(kls),
        })

    # Loop through all parameter combinations and test each 100 times
    # X = np.load("datasets/dataset_IP.npy")
    # Y = np.load("datasets/dataset_param_search.npz")
    # specs = Y["melspecs"]
    # set = 1
    #
    # for N in N_values:
    #     for lr in lr_values:
    #         lr = round(lr, 2)
    #         for sr in sr_values:
    #             print(set)
    #             for i in range(iterations):
    #                 # Create model
    #                 reservoir = create_model(N, lr, sr, sigma)
    #
    #                 # Apply intrinsic plasticity
    #                 _ = reservoir.fit(X, warmup=100)
    #
    #                 # Get activations
    #                 idx = np.random.randint(len(specs))
    #                 random_spec = specs[idx]
    #                 states = reservoir.run(random_spec)
    #
    #                 # Get the KL-divergence and entropy for neuron activations
    #                 kl = get_KL_divergence_and_entropy(states, sigma)
    #
    #                 print(f"{set}: Total KL: {kl}")
    #
    #                 results.append({
    #                     "N": N,
    #                     "lr": lr,
    #                     "sr": sr,
    #                     "KL": kl,
    #                 })
    #             set += 1
    #
    # df = pd.DataFrame(results)
    # df.to_csv(f"results_parameters_ip_specs_{iterations}_iter.csv", index=False)
    # df.to_csv(f"avg_kl_vs_kl_per_neuron.csv", index=False)
