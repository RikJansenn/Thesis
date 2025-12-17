import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import librosa

matplotlib.use('tkagg')

def heavyside(x):
    return 1.0 if x >= 0 else 0.0


def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den


def plot_pdf(states, sigma, title):
    if isinstance(states, np.ndarray):
        states = [states[i, :] for i in range(states.shape[0])]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0, 16)

    # Plot per-neuron PDFs
    for neuron_states in states:
        hist, edges = np.histogram(neuron_states, density=True, bins=200)
        points = 0.5 * (edges[:-1] + edges[1:])
        ax1.scatter(points, hist, s=0.2, alpha=0.25)

    # Global activation distribution
    all_states = np.concatenate(states)
    ax1.hist(
        all_states,
        density=True,
        bins=200,
        histtype="step",
        label="Global activation",
        lw=3.0,
    )

    # Target distribution
    x = np.linspace(-1.0, 1.0, 200)
    pdf = [bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in x]
    ax1.plot(x, pdf, label="Target distribution", linestyle="--", lw=3.0)

    ax1.set_xlabel("Reservoir activations")
    ax1.set_ylabel("Probability density")
    ax1.set_title(title)
    ax1.legend()

    plt.savefig(f"plots/{title}")
    plt.show()

def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(S, sr, hop_length, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear'
    )
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_weights(W):
    plt.figure(figsize=(6, 6))
    plt.imshow(W, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Weight')
    plt.title('Neuron Connection Weights (-1 to 1)')
    plt.show()


