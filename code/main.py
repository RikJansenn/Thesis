import numpy as np
from sklearn.metrics import f1_score
from models import ShallowNetwork
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import time


start_time = time.time()

# Experiment settings
IP = True
TONOTOPIC = False
SPEC = "Mel"

# Parameter ranges
N = 1000
sr = 0.8
lr = 0.94
sigma = 0.1

# Probability for input sparsity
p = 0.1

def create_training_data():
#     data_train = np.load("../datasets/dataset_train.npz")
#     data_test = np.load("../datasets/dataset_param_search.npz")

    # X_train = data_train["melspecs"]
    # Y_train = data_train["targets_mel"]
    # X_test = data_test["melspecs"]
    # Y_test = data_test["targets_mel"]

    data = np.load("../datasets/mel_specs_63x40(128).npz")
    #
    X = data["melspecs"]
    Y = data["targets_mel"]

    X_remain, X_all, Y_remain, Y_all = train_test_split(
        X, Y, test_size=0.2, random_state=42,  shuffle=True
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.1, random_state=42, shuffle=True
    )

    # X_train, X_remain, Y_train, Y_remain = train_test_split(
    #     X, Y, test_size=0.2, random_state=42,  shuffle=True
    # )
    #
    # X_test, X_discard, Y_test, Y_discard = train_test_split(
    #     X_remain, Y_remain, test_size=0.5, random_state=42, shuffle=True
    # )

    return X_train, X_test, Y_train, Y_test

def ridge_search(model, X, Y, ridge_values, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ridge_f1_scores = {}

    for ridge in ridge_values:
        fold_f1s = []

        model.ridge = ridge  # update ridge on the existing model

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            # Train readout
            model.train(X_tr, Y_tr)

            # Validate
            _, y_true, y_pred, *_ = model.test(X_val, Y_val)
            f1 = f1_score(y_true, y_pred, average="macro")
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        ridge_f1_scores[ridge] = mean_f1

    best_ridge = max(ridge_f1_scores, key=ridge_f1_scores.get)
    print("\nBest ridge:", best_ridge, "with F1:", ridge_f1_scores[best_ridge])

    return best_ridge

if __name__ == "__main__":
    # Load data
    X_train, X_test, Y_train, Y_test = create_training_data()

    ridge_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e3]

    print("Creating model...")
    model = ShallowNetwork(N=N, sr=sr, lr=lr, input_scaling=1, sigma=sigma, ridge=1e-6, input_dim=X_train[0].shape[1], IP=IP)

    if IP:
        print("Applying IP...")
        model.apply_ip(p)

    model.create_input_weights(p)

    # model.ridge = ridge_search(model, X_test, Y_test, ridge_values)

    print("Training model...")
    model.train(X_train, Y_train)

    print("Testing model...")
    acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)
    # acc, y_true, y_pred = model.test(X_test, Y_test)
    f1 = f1_score(y_true, y_pred, average="macro")

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
