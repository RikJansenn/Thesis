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
matplotlib.use('Qt5Agg')
rpy.set_seed(52)

# Create model
def create_model():
    reservoir = IPReservoir(500, mu=0.0, sigma=0.3, sr=0.9, activation="tanh", epochs=3)
    # reservoir = Reservoir(units=500, sr=0.9, lr=0.2, input_scaling=0.5)
    readout = Ridge(ridge=1e-6, input_dim=500)

    esn = reservoir >> readout

    return reservoir, readout, esn

def memory_capacity(reservoir, readout, u, n_train=5000, max_delay=200, warmup=100):
    X = reservoir.run(u)
    MCs = []

    for delay in range(1, max_delay + 1):
        # Target = input delayed by k
        target_train = u[warmup + delay : n_train]
        features_train = X[warmup : n_train - delay]

        readout.fit(features_train, target_train)

        # Test phase
        target_test = u[n_train + delay :]
        features_test = X[n_train : len(u) - delay]
        y_pred = readout.run(features_test)

        # Compute squared correlation coefficient
        r, _ = pearsonr(target_test.flatten(), y_pred.flatten())
        MCs.append(r**2)

    total_MC = np.sum(MCs)
    return total_MC, MCs

def plot_pdf(reservoir, states):
    # Compute obsererved PDF
    flat_activations = states.flatten()
    kde = gaussian_kde(flat_activations)

    # Evaluate PDF with values
    x = np.linspace(flat_activations.min(), flat_activations.max(), 500)
    observed_pdf = kde(x)

    # Compute target PDF
    mu = reservoir.mu
    sigma = reservoir.sigma
    target_pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    target_pdf /= np.trapezoid(target_pdf, x)

    # Compute Entropy
    epsilon = 1e-12
    entropy = -np.trapezoid(observed_pdf * np.log(observed_pdf + epsilon), x)

    print("Entropy: ", entropy)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, observed_pdf, color="royalblue", label='Observed Activation Density')
    plt.plot(x, target_pdf, "--", color="red", label='Target Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Probability Density')
    plt.title('Probability Density of Neuron Activations (Gaussian KDE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pdf_per_neuron(reservoir, states, n):
    x = np.linspace(-1, 1, 200)

    mu = reservoir.mu
    sigma = reservoir.sigma
    target_pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    target_pdf /= np.trapezoid(target_pdf, x)

    plt.plot(x, target_pdf, "--", color="#1f77b4", lw=3, label='Target Distribution')

    for i in range(n):  # example: plot first 5 neurons
        kde = gaussian_kde(states[:, i])
        plt.plot(x, kde(x), label=f'Neuron {i}')
    plt.legend()
    plt.show()

def compare_ip_effect():
    n_units = 100
    max_delay = 200
    input_length = 6000
    warmup = 100
    n_train = 5000

    # Generate input
    u = np.random.uniform(-0.8, 0.8, size=(input_length, 1))

    # Create reservoirs
    reservoir_plain = Reservoir(n_units, sr=0.9, lr=0.55, activation="tanh")
    readout_plain = Ridge(ridge=1e-7)
    mc_plain, mc_list_plain = memory_capacity(reservoir_plain, readout_plain, u, n_train=n_train, max_delay=max_delay, warmup=warmup)

    reservoir_ip = IPReservoir(100, mu=0.0, sigma=0.1, sr=0.9, lr=0.55, learning_rate=1e-5, activation="tanh", epochs=10)
    readout_ip = Ridge(ridge=1e-7)

    # Pretrain with IP
    _ = reservoir_ip.fit(u, warmup=warmup)

    mc_ip, mc_list_ip = memory_capacity(reservoir_ip, readout_ip, u, n_train=n_train, max_delay=max_delay, warmup=warmup)

    # Print results
    print(f"Total Memory Capacity (no IP): {mc_plain:.3f}")
    print(f"Total Memory Capacity (with IP): {mc_ip:.3f}")

def kl_divergence_and_entropy(observed_samples, mu, sigma):
    """
    Compute both the KL divergence between the observed activation
    distribution and a target Gaussian N(mu, sigma^2), and the
    Shannon entropy of the observed distribution.
    """
    # Estimate observed PDF via KDE
    kde = gaussian_kde(observed_samples)
    x = np.linspace(min(-3, observed_samples.min()), max(3, observed_samples.max()), 500)
    p = kde(x)
    q = norm.pdf(x, loc=mu, scale=sigma)

    # Normalize both PDFs to ensure proper distributions
    p /= np.trapezoid(p, x)
    q /= np.trapezoid(q, x)

    # Compute entropy (Shannon)
    epsilon = 1e-12
    entropy = -np.trapezoid(p * np.log(p + epsilon), x)

    # Compute KL divergence KL(P || Q)
    kl = np.trapezoid(p * np.log((p + epsilon) / (q + epsilon)), x)

    return kl, entropy

if __name__ == "__main__":
    data_len = 5000
    train_len = 4000
    mses = []
    entropies = []
    kls = []

    for i in range(10):
        # Create model and datasets
        reservoir, readout, esn = create_model()
        u, y = narma(data_len)
        u = u[-len(y):]

        # Create train and test sets
        u_train, y_train = u[:train_len], y[:train_len]
        u_test, y_test = u[train_len:], y[train_len:]


        # Do Intrinsic Plasticity stuff
        states_before = reservoir.run(u)
        reservoir = reservoir.fit(u_train, warmup=100)
        states_after = reservoir.run(u)
        print("hey")
        # #kl1 = kl_divergence(states_before.flatten(), mu=reservoir.mu, sigma=reservoir.sigma)
        kl, entropy = kl_divergence_and_entropy(states_after.flatten(), mu=reservoir.mu, sigma=reservoir.sigma)

        # #print("KL divergence:", kl1)
        print("KL divergence:", kl)
        print("Entropy:", entropy)

        # Train readout layer
        X_train = reservoir.run(u_train)
        readout = readout.fit(X_train, y_train)

        # Test and evaluate model
        reservoir.reset()
        X_test = reservoir.run(u_test)
        y_pred = readout.run(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        print("Test MSE:", mse)

        mses.append(mse)
        entropies.append(entropy)
        kls.append(kl)

    print("Average MSE over 10 runs:", np.mean(mses))
    print("Average Entropy over 10 runs:", np.mean(entropies))
    print("Average KL divergence over 10 runs:", np.mean(kls))
