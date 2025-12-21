import numpy as np

data = np.load("dataset_train.npz", allow_pickle=True)

print("Keys:", data.files)

specs = data["specs"]
melspecs = data["melspecs"]
targets = data["targets"]

print("Type specs:", type(specs))
print("Type melspecs:", type(melspecs))
print("Type targets:", type(targets))

print("Length specs:", len(specs))
print("Length melspecs:", len(melspecs))
print("Length targets:", len(targets))

print("Unique spec shapes:")
print({s.shape for s in specs})

print("Unique mel shapes:")
print({m.shape for m in melspecs})

print("Unique target shapes:")
print({t.shape for t in targets})

i = 0

spec = specs[i]
mel = melspecs[i]
target = targets[i]

print("Spec shape:", spec.shape)
print("Mel shape:", mel.shape)
print("Target shape:", target.shape)

# Time dimension check
print("Spec timesteps:", spec.shape[0])
print("Target timesteps:", target.shape[0])

print("Spec min/max:", spec.min(), spec.max())
print("Mel min/max:", mel.min(), mel.max())
print("Target unique values:", np.unique(target))