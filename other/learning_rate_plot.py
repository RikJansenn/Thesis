import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa

matplotlib.use('tkagg')

data = np.load("../datasets/IP_testset.npz")
S = data["specs"][0]
S_mel = data["melspecs"][0]
S_coch = data["cochs"][0]

def plot_all_spectrograms(S, S_mel, S_coch, sr, n_fft=256, hop_length_stft=128):
    """
    Plots STFT spectrogram, Mel spectrogram, and cochleagram
    with frequency (Hz) on the y-axis and time (s) on the x-axis.
    Assumes S, S_mel, S_coch are time × freq (transposed) arrays.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # -----------------------
    # 1. Linear STFT spectrogram
    # -----------------------
    freqs = np.fft.rfftfreq(n_fft, 1/sr)  # Hz
    time_stft = np.arange(S.shape[0]) * hop_length_stft / sr  # seconds

    im0 = axes[0].imshow(
        S.T,
        origin='lower',
        aspect='auto',
        extent=[time_stft[0], time_stft[-1], freqs[0], freqs[-1]],
        cmap='magma'
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Linear Spectrogram")
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")

    # -----------------------
    # 2. Mel Spectrogram
    # -----------------------
    n_mels = S_mel.shape[1]
    time_mel = np.arange(S_mel.shape[0]) * hop_length_stft / sr  # seconds
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)

    im1 = axes[1].imshow(
        S_mel.T,
        origin='lower',
        aspect='auto',
        extent=[time_mel[0], time_mel[-1], mel_freqs[0], mel_freqs[-1]],
        cmap='magma'
    )
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

    # -----------------------
    # 3. Cochleagram
    # -----------------------
    n_ch = S_coch.shape[1]
    time_coch = np.arange(S_coch.shape[0]) * hop_length_stft / sr  # seconds

    # Approximate cochlear frequencies if not given
    coch_freqs = np.geomspace(50, sr/2, n_ch)

    im2 = axes[2].imshow(
        S_coch.T,
        origin='lower',
        aspect='auto',
        extent=[time_coch[0], time_coch[-1], coch_freqs[0], coch_freqs[-1]],
        cmap='magma'
    )
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Cochleagram")
    fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()
    

plot_all_spectrograms(S, S_mel, S_coch, 8000)


