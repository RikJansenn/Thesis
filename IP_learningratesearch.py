import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from scipy.stats import entropy, norm
import numpy as np
import matplotlib
from scipy.stats import entropy
import pandas as pd

from biological_constraints import apply_ip, apply_ip_specs, apply_IP_multiband
# from my_biological_constraints import apply_ip

from utils import plot_pdf

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
    # ent = entropy(hist)

    return kl


# Create model
def create_model(N, lr, sr, learn_rate):
    reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=0.1, activation="tanh", epochs=4, learning_rate=learn_rate)

    return reservoir


def create_training_data():
    data = np.load("datasets/dataset_IP.npy")

    return data

def create_test_data():
    data = np.load("datasets/dataset_param_search.npz")
    X_test = data["melspecs"]

    return X_test


def create_input_weights(reservoir, input_d, p):
    n = 1 / input_d
    # reservoir.Win = np.random.uniform(n, n, (reservoir.units, input_d))

    reservoir.Win = np.random.uniform(0.5 * n, n, (reservoir.units, input_d))

    mask = np.random.rand(reservoir.units, input_d) < p
    reservoir.Win *= mask
    reservoir.input_dim = input_d

    return reservoir.Win

def create_spec_weights(reservoir, input_d, p):
    reservoir.Win = np.random.uniform(1, 1, (reservoir.units, input_d))
    mask = np.random.rand(reservoir.units, input_d) < p
    # reservoir.Win *= mask

    return reservoir.Win


if __name__ == "__main__":
    data_len = 1000

    N = 500
    sr = 0.95
    lr = 1
    p = 0.1

    learn_rates = [9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4]

    for learn_rate in learn_rates:
        # Create model
        reservoir = create_model(N, sr, lr, learn_rate)

        X = create_training_data()
        input_d = X.shape[1]

        # reservoir = apply_ip_specs(reservoir, input_d, p, X)
        # reservoir = apply_ip(reservoir, X)
        reservoir = apply_IP_multiband(reservoir, input_d)
        reservoir.Win = create_spec_weights(reservoir, input_d, p)

        digit = 0
        i = 0

        X_test = create_test_data()[:300]
        kls = []
        results = []

        # For each spec, plot the pdf and get KL divergence/entropy
        for spec in X_test:
            states = reservoir.run(spec)
            kl = get_KL_divergence_and_entropy(states, 0.1)
            kls.append(kl)
            # plot_pdf(states, 0.1, f"PDF for digit {digit}, number {i+1}")

            i += 1
            if i % 10 == 0:
                digit += 1

            reservoir.reset()

        kl_mean = np.mean(kls)
        print(f"{learn_rate} mean: {kl_mean}")

        results.append({
            "learn_rate": learn_rate,
            "KL_mean": kl_mean,
        })

        df = pd.DataFrame(results)
        df.to_csv("learning_rate_results/trainmultinarma_testspecs_sparsity.csv", index=False)
