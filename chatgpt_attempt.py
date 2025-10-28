import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# 1. Create synthetic time series data
# -----------------------------
# Suppose we have 3 classes (0,1,2)
n_classes = 3
n_features = 1     # raw time series has 1 feature
seq_length = 50    # number of time steps

# Generate 5 sequences per class
X = []
y_labels = []

for label in range(n_classes):
    for _ in range(5):
        # create a simple synthetic sequence
        seq = np.sin(np.linspace(0, 3.14*label, seq_length)).reshape(-1, 1)
        seq += 0.1 * np.random.randn(seq_length, 1)  # add small noise
        X.append(seq)
        y_labels.append(label)

X = np.array(X, dtype=object)  # list of sequences
y_labels = np.array(y_labels)

# Convert labels to one-hot
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y_labels.reshape(-1, 1))

# -----------------------------
# 2. Create reservoir and readout
# -----------------------------
reservoir = Reservoir(units=100,
                      input_dim=n_features,
                      spectral_radius=0.9,
                      input_scaling=0.5,
                      leaking_rate=0.3)

readout = Ridge(ridge=1e-6)

# Pipeline
model = reservoir >> readout

# -----------------------------
# 3. Train the model
# -----------------------------
model.fit(X, y_onehot)

# -----------------------------
# 4. Test on a new sequence
# -----------------------------
test_seq = np.sin(np.linspace(0, 3.14*1, seq_length)).reshape(-1,1)  # class 1
y_pred = model.predict([test_seq])
pred_class = np.argmax(y_pred[0])

print("Predicted one-hot:", y_pred[0])
print("Predicted class:", pred_class)