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
matplotlib.use('tkAgg')
rpy.set_seed(52)


def heavyside(x):
    return 1.0 if x >= 0 else 0.0

def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den

def get_KL_divergence_and_entropy(states, sigma):
    # Estimated probability distribution of state activation
    all_activations = states.flatten()
    x_min = all_activations.min()
    x_max = all_activations.max()

    # Estimate PDF with a histogram from all activations
    hist, edges = np.histogram(all_activations, density=True, bins=200, range=(x_min, x_max))

    # Convert PDF to PMF
    dx = edges[1] - edges[0] # width of a bin
    P = hist * dx
    P /= P.sum()

    # Use bin centers, so estimated PDF and target PDF are aligned
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Target PDF
    pdf = np.array([bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in bin_centers])

    # Convert to PMF
    Q = pdf * dx
    Q /= Q.sum()

    # Add epsilon for stability
    eps = 1e-12
    P += eps
    Q += eps

    kl = entropy(P, Q)
    ent = entropy(P)

    return kl, ent

def plot_pdf(states, sigma):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0, 16)

    x_min = states.min()
    x_max = states.max()

    for s in range(states.shape[1]):
        hist, edges = np.histogram(states[:, s], density=True, bins=200, range=(x_min, x_max))
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

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    pdf = [bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in bin_centers]
    ax1.plot(bin_centers, pdf, label="Target distribution", linestyle="--", lw=3.0)
    ax1.set_xlabel("Reservoir activations")
    ax1.set_ylabel("Probability density")
    plt.legend()
    plt.show()


# Create model
def create_model(N, lr, sr, sigma):
    reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=10)
    readout = Ridge(ridge=1e-6, input_dim=500)

    return reservoir, readout

if __name__ == "__main__":
    data_len = 1000
    entropies = []
    kls = []

    # Parameters to test
    N_values = np.arange(400, 1001, 100)
    lr_values = np.arange(0.85, 1, 0.03)
    sr_values = np.arange(0.8, 1.21, 0.1)
    sigma = 0.1

    # Create narma series
    _, X = narma(data_len)

    for N in N_values:
        for lr in lr_values:
            for sr in sr_values:
                reservoir, readout = create_model(N, lr, sr, sigma)

                # Apply intrinsic plasticity
                _ = reservoir.fit(X, warmup=100)
                states_after = reservoir.run(X)

                # Get activations
                states = reservoir.run(X[100:])

                kl, ent = get_KL_divergence_and_entropy(states, sigma)

                kls.append(kl)
                entropies.append(ent)
                #plot_pdf(states, sigma)

    print(kls)
    print(entropies)
