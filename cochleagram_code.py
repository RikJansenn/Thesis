def create_cochleagram(audio, sr, sample_factor=2, nonlinearity="db"):
    # Length to pad/trim to
    fixed_length = 1
    n = 64

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    S = cgram.human_cochleagram(audio,
                                sr,
                                n=n,
                                downsample=downsampler,
                                sample_factor=sample_factor,
                                nonlinearity=nonlinearity,
                                strict=False).T

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    print(f"Coch shape: {S.shape}")

    return S


def downsampler(envs):
    return cgram.apply_envelope_downsample(
        envs,
        mode='poly',
        audio_sr=8000,
        env_sr=64  # Amount of timesteps
    )