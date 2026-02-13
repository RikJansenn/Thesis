import pycochleagram.cochleagram as cgram
import numpy as np
import librosa
import os
import csv

folder_path = "../preprocessed_data"

def load_training_data(folder_path):
    cochlea_samples = []

    targets = []

    # Create label for silence
    silence_label = np.zeros(11)
    silence_label[10] = 1

    with open("used_files_for_IP.csv", "r", newline="") as f:
        reader = csv.reader(f)
        files_used_for_IP = [row[0] for row in reader]

    total = 0

    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)

                if file_path in files_used_for_IP:
                    continue

                print(file_path)
                audio, sr = librosa.load(file_path, sr=None)

                S = create_cochleagram(audio, sr)
                cochlea_samples.append(S)

                # Create one-hot vector representing digit
                digit = int(filename[0])
                label = np.eye(11)[digit].reshape(1, -1)

                # Create silence and digit labels per timestep
                time_steps = S.shape[0]
                labels_expanded = np.zeros((time_steps, 11))
                for t in range(time_steps):
                    if np.all(S[t] == 0):
                        labels_expanded[t] = silence_label
                    else:
                        labels_expanded[t] = label

                # Append final target array
                targets.append(labels_expanded)

                total += 1

                # if total >= 10:
                #     break

    return cochlea_samples, targets

def create_cochleagram(audio, sr, sample_factor=2, nonlinearity="db"):
    # Length to pad/trim to
    fixed_length = 1
    n = 63

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    S = cgram.human_cochleagram(audio,
                                         sr,
                                         n=n,
                                         low_lim=50,
                                         high_lim=3800,
                                         downsample=downsampler,
                                         sample_factor=sample_factor,
                                         nonlinearity=nonlinearity,
                                         strict=False).T

    # plt.subplot(222)
    # plt.title('Cochleagram with poly downsampling')
    # plt.ylabel('filter #')
    # plt.xlabel('time')
    # cu.cochshow(np.flipud(human_coch.T), interact=False)
    # plt.gca().invert_yaxis()
    # plt.show()

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

def trim_or_pad(audio, sr, fixed_length):
    target_len = int(sr * fixed_length)
    if len(audio) < target_len:
        pad_len = target_len - len(audio)

        # Choose random split to pad before and after signal
        left = np.random.randint(0, pad_len + 1)
        right = pad_len - left

        audio = np.pad(audio, (left, right), mode='constant')  # pad
    else:
        audio = audio[:target_len]  # trim

    return audio


if __name__ == "__main__":
    cochs, targets = load_training_data(folder_path)

    cochs = np.array(cochs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    np.savez(
        "../datasets/dataset_train_cochs.npz",
        cochs=cochs,
        targets=targets
    )
