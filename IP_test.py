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
    plt.savefig("Activations")
    plt.show()


# Create model
def create_model(N, lr, sr, sigma, epochs=4):
    reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=epochs)

    return reservoir

if __name__ == "__main__":
    data_len = 1000
    iterations = 1

    results = []

    # Parameters to test
    N_values = np.arange(400, 1001, 100)        # 7
    lr_values = np.arange(0.85, 1.01, 0.03)     # 6
    sr_values = np.arange(0.8, 1.21, 0.1)       # 5
    sigma = 0.1

    print(lr_values)

    # Create narma series
    _, X = narma(data_len)

    # Testing amount of epochs
    # for epochs in range(1, 13):
    #     kls = []
    #     for i in range(iterations):
    #         reservoir = create_model(500, 1, 0.95, 0.1, epochs)
    #
    #         _ = reservoir.fit(X, warmup=100)
    #         states = reservoir.run(X)
    #
    #         kl, ent = get_KL_divergence_and_entropy(states, sigma)
    #         kls.append(kl)
    #
    #     print(np.mean(kls))
    #     results.append({
    #         "epochs": epochs,
    #         "KL_mean": np.mean(kls),
    #     })

    # Loop through all parameter combinations and test each 100 times

    set = 1

    for N in N_values:
        for lr in lr_values:
            lr = round(lr, 2)
            for sr in sr_values:
                print(set)
                for i in range(iterations):
                    # Create model
                    reservoir = create_model(600, 1, 0.8, sigma)

                    states_before = reservoir.run(X[100:])
                    # Apply intrinsic plasticity
                    _ = reservoir.fit(X, warmup=100)
                    # Get activations
                    states = reservoir.run(X[100:])

                    plot_pdf(states_before, sigma)
                    plot_pdf(states, sigma)

                    # Get the KL-divergence and entropy for neuron activations
                    kl, ent = get_KL_divergence_and_entropy(states, sigma)

                    results.append({
                        "N": N,
                        "lr": lr,
                        "sr": sr,
                        "set_n": set,
                        "KL": kl,
                        "entropy": ent,
                    })
                set += 1

    df = pd.DataFrame(results)
    df.to_csv(f"results_parameters_ip_{iterations}_iter_v2.csv", index=False)
