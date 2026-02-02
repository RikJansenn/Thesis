import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from scipy.stats import entropy, norm
import numpy as np
import matplotlib
from scipy.stats import entropy

from biological_constraints import apply_ip, apply_ip_specs
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
def create_model(N, lr, sr):
    reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=0.1, activation="tanh", epochs=4, learning_rate=5e-5)

    return reservoir


def create_training_data():
    data = np.load("datasets/dataset_IP.npy")

    return data

def create_test_data():
    data = np.load("datasets/IP_testset.npz")
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


if __name__ == "__main__":
    data_len = 1000

    N = 500
    sr = 0.95
    lr = 1
    p = 0.1

    # Create model
    reservoir = create_model(N, sr, lr)

    X = create_training_data()  # Should return 10 specs for each digit
    input_d = X.shape[1]

    reservoir = apply_ip_specs(reservoir, X)
    # reservoir = apply_ip(reservoir, X)
    # reservoir.Win = create_input_weights(reservoir, input_d, p)

    digit = 0
    i = 0

    X_test = create_test_data()
    kls = []
    # For each spec, plot the pdf and get KL divergence/entropy
    for spec in X_test:
        states = reservoir.run(spec)
        kl = get_KL_divergence_and_entropy(states, 0.1)
        print(kl)
        kls.append(kl)
        plot_pdf(states, 0.1, f"PDF for digit {digit}, number {i+1}")

        i += 1
        if i % 10 == 0:
            digit += 1

        # reservoir.reset()

    print(f"mean: {np.mean(kls)}")
