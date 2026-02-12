import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import librosa
from scipy.stats import entropy

matplotlib.use('tkagg')

def heavyside(x):
    return 1.0 if x >= 0 else 0.0


def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den


def plot_pdf(states, sigma, title):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0, 16)
    for s in range(states.shape[1]):
        hist, edges = np.histogram(states[:, s], density=True, bins=50)
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
    plt.title(title)
    plt.legend()
    plt.savefig(f"../plots/{title}")
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

def create_training_data(spec="Mel"):
    data_train = np.load("../datasets/dataset_train.npz")
    data_test = np.load("../datasets/dataset_param_search.npz")
    if spec == "Linear":
        X_train = data_train["specs"]
        Y_train = data_train["targets_linear"]
        X_test = data_test["specs"]
        Y_test = data_test["targets_linear"]
    elif spec == "Mel":
        X_train = data_train["melspecs"]
        Y_train = data_train["targets_mel"]
        X_test = data_test["melspecs"]
        Y_test = data_test["targets_mel"]
    elif spec == "Cochlea":
        X_train = data_train["cochs"]
        Y_train = data_train["targets_cochlea"]
        X_test = data_test["cochs"]
        Y_test = data_test["targets_cochlea"]

    return X_train, X_test, Y_train, Y_test
