import random

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('tkagg')

import reservoirpy as rpy
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge
from reservoirpy.datasets import narma

import numpy as np
import matplotlib
import librosa
import os

rpy.set_seed(70)

# Create model
def create_model():
    reservoir = IPReservoir(500, mu=0.0, sigma=0.3, sr=0.95, activation="tanh", epochs=10)
    readout = Ridge(ridge=1e-7)
    return reservoir, readout
def plot_pdf(states):
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

def plot_pdf_per_neuron(states, n):
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

if __name__ == "__main__":
    reservoir, readout = create_model()

    u, y = narma(1000)
    states_before = reservoir.run(u)
    _ = reservoir.fit(u, warmup=100)
    states_after = reservoir.run(u)

    theoretical_entropy = 0.5 * np.log(2 * np.pi * np.e * reservoir.sigma ** 2)
    print("Theoretical entropy: ", theoretical_entropy)

    # plot_pdf(states_before)
    plot_pdf(states_after)
