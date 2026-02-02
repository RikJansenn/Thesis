import numpy as np

INPUT_FILE = "../datasets/dataset_train_all.npz"
SUBSET_FILE = "../datasets/IP_testset.npz"
REMAINING_FILE = "../datasets/dataset_train_remaining.npz"

SILENCE_LABEL = 10
N_PER_DIGIT = 10
RANDOM_SEED = 42

data = np.load(INPUT_FILE)
specs = data["specs"]      # (N, T, F)
melspecs = data["melspecs"]
cochs = data["cochs"]
targets_linear = data["targets_linear"]
targets_mel = data["targets_mel"]
targets_cochlea = data["targets_cochlea"]

N = len(specs)

# Find the full digit label for each utterance
labels = np.argmax(targets_mel, axis=2)  # (N, T)
digit_labels = np.full(N, -1, dtype=int)

for i, label in enumerate(labels):
    non_silence = label[label != SILENCE_LABEL]
    digit_labels[i] = np.bincount(non_silence).argmax()

# Select 10 random utterances per digit
rng = np.random.default_rng(RANDOM_SEED)
selected_indices = []
for digit in range(10):
    idxs = np.where(digit_labels == digit)[0]
    chosen = rng.choice(idxs, size=N_PER_DIGIT, replace=False)
    selected_indices.extend(chosen)

selected_indices = np.array(selected_indices)

# Create new dataset
subset_specs = specs[selected_indices]
subset_melspecs = melspecs[selected_indices]
subset_cochs = cochs[selected_indices]
subset_targets = digit_labels[selected_indices]

# Remove from original dataset
mask = np.ones(N, dtype=bool)
mask[selected_indices] = False

remaining_specs = specs[mask]
remaining_melspecs = melspecs[mask]
remaining_cochs = cochs[mask]
remaining_targets_linear = targets_linear[mask]
remaining_targets_mel = targets_mel[mask]
remaining_targets_cochlea = targets_cochlea[mask]

# Save both datasets
np.savez(
    SUBSET_FILE,
    specs=subset_specs,
    melspecs=subset_melspecs,
    cochs=subset_cochs,
    targets=subset_targets
)

np.savez(
    REMAINING_FILE,
    specs=remaining_specs,
    melspecs=remaining_melspecs,
    cochs=remaining_cochs,
    targets_linear=remaining_targets_linear,
    targets_mel=remaining_targets_mel,
    targets_cochlea=remaining_targets_cochlea
)

print("Subset size:", len(subset_specs))
print("Remaining size:", len(remaining_specs))
print("Total:", len(subset_specs) + len(remaining_specs))
