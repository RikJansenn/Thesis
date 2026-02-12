import numpy as np

data = np.load("datasets/dataset_train.npz", allow_pickle=True)

print("Keys:", data.files)

specs = data["specs"]
melspecs = data["melspecs"]
cochs = data["cochs"]
targets_linear = data["targets_linear"]
targets_mel = data["targets_mel"]
targets_cochlea = data["targets_cochlea"]

# Count each class
labels = np.argmax(targets_linear, axis=2)  # (N, T)

digit_counts = np.zeros(10, dtype=int)

for utt_labels in labels:
    # remove silence (label 10)
    non_silence = utt_labels[utt_labels != 10]

    if len(non_silence) == 0:
        continue  # just in case

    # spoken digit = most frequent non-silence label
    digit = np.bincount(non_silence).argmax()
    digit_counts[digit] += 1

# Print results
for d, c in enumerate(digit_counts):
    print(f"Digit {d}: {c}")


print("Type specs:", type(specs))
print("Type targets:", type(targets_linear))
print("Type cochs:", type(cochs))

print("Length specs:", len(specs))
print("Length targets:", len(targets_linear))
print("Length cochs:", len(cochs))

print("Unique spec shapes:")
print({s.shape for s in specs})

print("Unique target shapes:")
print({t.shape for t in targets_linear})

print("Unique coch shapes:")
print({c.shape for c in cochs})

i = 0

spec = specs[i]
target = targets_linear[i]

print("Spec shape:", spec.shape)
print("Target shape:", target.shape)

# Time dimension check
print("Spec timesteps:", spec.shape[0])
print("Target timesteps:", target.shape[0])

print("Spec min/max:", spec.min(), spec.max())
print("Target unique values:", np.unique(target))