from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import IPReservoir
from reservoirpy.nodes import Ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import mode
import pandas as pd
import pickle
from reservoirpy.observables import effective_spectral_radius
import numpy as np

from utils import *
from biological_constraints import apply_ip, create_tonotopic_mapping

IP = True
TONOTOPIC = False
SPEC = "Mel"  # Choose "Cochlea", "Mel" or "Linear"

def create_model(N, sr, lr, sigma):
    if IP:
        reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=10, learning_rate=2e-4)
    else:
        reservoir = Reservoir(N, sr=sr, lr=lr)
    readout = Ridge(ridge=1e-7, output_dim=11)

    return reservoir, readout

def train_model(X, Y, reservoir, readout):
    X_list = []
    Y_list = []

    total = len(X)
    i = 1

    for spec, labels in zip(X, Y):
        states = reservoir.run(spec)

        X_list.append(states)
        Y_list.append(labels)

        i += 1

    X_all = np.vstack(X_list)  # shape = (sum_T, units)
    Y_all = np.vstack(Y_list)  # shape = (sum_T, 10)

    readout.fit(X_all, Y_all)

    return readout

def test_model(reservoir, readout, X_test, Y_test):
    y_pred = []
    timestep_predictions = []

    for spec in X_test:
        states = reservoir.run(spec)
        predictions = readout.run(states)                               # Get raw prediction per timestep
        pred_per_timestep = np.argmax(predictions, axis=1)              # Get one-hot winner at each timestep
        timestep_predictions.append(pred_per_timestep)
        non_silence_preds = pred_per_timestep[pred_per_timestep != 10]  # Remove silence as category
        final_pred = mode(non_silence_preds, keepdims=False).mode       # Get winning digit with majority voting

        y_pred.append(final_pred)

    y_pred = np.array(y_pred)

    # Get the targets
    y_per_timestep = np.argmax(Y_test, axis=2)
    y_true = []

    for i in range(y_per_timestep.shape[0]):
        non_silence = y_per_timestep[i][y_per_timestep[i] != 10]
        y_true.append(non_silence[0])  # or np.unique(non_silence)[0]

    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")

    return accuracy, y_true, y_pred, timestep_predictions, y_per_timestep

def create_training_data():
    data_train = np.load("datasets/dataset_train.npz")
    data_test = np.load("datasets/dataset_param_search.npz")
    if SPEC == "Linear":
        X_train = data_train["specs"]
        Y_train = data_train["targets_linear"]
        X_test = data_test["specs"]
        Y_test = data_test["targets_linear"]
    elif SPEC == "Mel":
        X_train = data_train["melspecs"]
        Y_train = data_train["targets_mel"]
        X_test = data_test["melspecs"]
        Y_test = data_test["targets_mel"]
    elif SPEC == "Cochlea":
        X_train = data_train["cochs"]
        Y_train = data_train["targets_cochlea"]
        X_test = data_test["cochs"]
        Y_test = data_test["targets_cochlea"]

    return X_train, X_test, Y_train, Y_test

def create_input_weights(reservoir, input_d, p):
    n = 1/input_d
    reservoir.Win = np.random.uniform(0.5 * n, n, (reservoir.units, input_d))

    # Apply mask for sparsity
    mask = np.random.rand(reservoir.units, input_d) < p
    reservoir.Win *= mask

    reservoir.input_dim = input_d

    return reservoir.Win

def run_single_fold(X_train, Y_train, X_test, Y_test, N, sr, lr, p):
    # Create fresh model
    reservoir, readout = create_model(N, sr, lr)

    input_d = X_train[0].shape[1]

    # Tonotopic mapping
    if TONOTOPIC:
        W_in, W = create_tonotopic_mapping(N, sr, input_d)
        reservoir.Win = W_in
        reservoir.W = W

    # Intrinsic plasticity (train ONLY on training data)
    if IP:
        reservoir = apply_ip(reservoir, X_train, input_d)
        reservoir.Win = create_input_weights(reservoir, input_d, p)
    else:
        reservoir.Win = create_input_weights(reservoir, input_d, p)

    # Train readout
    readout = train_model(X_train, Y_train, reservoir, readout)

    # Test
    _, y_true, y_pred = test_model(reservoir, readout, X_test, Y_test)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, cm

# def cross_validate(X, Y, k=5):
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#
#     accuracies = []
#     f1_scores = []
#     confusion_matrices = []
#
#     fold = 1
#     for train_idx, test_idx in kf.split(X):
#         print(f"\n=== Fold {fold}/{k} ===")
#
#         X_train = X[train_idx]
#         Y_train = Y[train_idx]
#         X_test = X[test_idx]
#         Y_test = Y[test_idx]
#
#         acc, f1, cm = run_single_fold(
#             X_train, Y_train, X_test, Y_test,
#             N=N, sr=sr, lr=lr, p=p
#         )
#
#         accuracies.append(acc)
#         f1_scores.append(f1)
#         confusion_matrices.append(cm)
#
#         print(f"Accuracy: {acc:.3f} | F1 (macro): {f1:.3f}")
#         fold += 1
#
#     accuracies = np.array(accuracies)
#     f1_scores = np.array(f1_scores)
#
#     print("\n=== Cross-validation summary ===")
#     print(f"Accuracy: {accuracies.mean():.3f} ± {accuracies.std():.3f}")
#     print(f"F1 score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
#
#     # Sum confusion matrices across folds
#     cm_total = np.sum(confusion_matrices, axis=0)
#
#     return accuracies, f1_scores, cm_total

def plot_confusion_matrix(cm, class_names):
    # Normalize per row (true labels)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_percent)
    plt.colorbar(label="Percentage")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (%)")

    # Add text annotations
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%",
                     ha="center", va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters to test
    N_values = [800, 1000, 1200]
    lr_values = [0.93, 0.97, 1]
    sr_values = [0.8, 1, 1.2]
    sigmas = [0.1, 0.2, 0.3]
    iterations = 50

    p = 0.1  # Probabilty of a connection existing

    X_train, X_test, Y_train, Y_test = create_training_data()
    input_d = X_train[0].shape[1]

    set = 0
    results = []
    for N in N_values:
        for lr in lr_values:
            set += 1
            print(f"\n=== Parameter set {set} ===")
            for i in range(iterations):
                print("Creating model and applying IP...")
                # Create model and apply IP
                reservoir, readout = create_model(N=N, sr=0.9, lr=lr, sigma=0.1)
                reservoir = apply_ip(reservoir, input_d)
                reservoir.Win = create_input_weights(reservoir, input_d, p)

                eff_sr = effective_spectral_radius(reservoir.W, lr=lr)

                with open(f"reservoirs/ip_reservoir_N{N}_lr{lr}_{i}.pkl", "wb") as f:
                    pickle.dump(reservoir, f)

                print("Computing KL...")
                # Compute KL with a random spectrogram
                idx = np.random.randint(len(X_test))
                random_spec = X_test[idx]
                states = reservoir.run(random_spec)
                kl, ent = get_KL_divergence_and_entropy(states, 0.1)

                print("Training model...")
                # Train and test model
                readout = train_model(X_train, Y_train, reservoir, readout)
                print("Testing model...")
                _, y_true, y_pred, timestep_predictions, y_per_timestep = test_model(reservoir, readout, X_test,
                                                                                             Y_test)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="macro")

                print("Saving results...")
                results.append({
                    "set": set,
                    "N": N,
                    "lr": lr,
                    "Effective_sr": eff_sr,
                    "KL": kl,
                    "Entropy": ent,
                    "Accuracy": acc,
                    "F1": f1,
                })

                df = pd.DataFrame(timestep_predictions)
                df.to_csv(f"timestep_predictions/timestep_predictions_N{N}_lr{lr}_{i}'.csv", index=False)

                df = pd.DataFrame(y_per_timestep)
                df.to_csv(f"timestep_predictions/true_labels_N{N}_lr{lr}_{i}.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(f"results_parameter_sweep_ipreservoir_N_lr", index=False)

    # Create fresh model
    # reservoir, readout = create_model(N, sr, lr)
    #
    # input_d = X_train[0].shape[1]
    #
    # # Tonotopic mapping
    # if TONOTOPIC:
    #     W_in, W = create_tonotopic_mapping(N, sr, input_d)
    #     reservoir.Win = W_in
    #     reservoir.W = W
    #
    # # Intrinsic plasticity
    # if IP:
    #     print("Applying IP...")
    #     reservoir = apply_ip(reservoir, input_d)
    #     reservoir.Win = create_input_weights(reservoir, input_d, p)
    # else:
    #     reservoir.Win = create_input_weights(reservoir, input_d, p)
    #
    # # Train readout
    # print("Training model...")
    # readout = train_model(X_train, Y_train, reservoir, readout)
    #
    # # Test
    # print("Testing model...")
    # _, y_true, y_pred, timestep_predictions, y_per_timestep = test_model(reservoir, readout, X_test, Y_test)
    #
    # acc = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, average="macro")
    # cm = confusion_matrix(y_true, y_pred)
    #
    # print(f"Accuracy: {acc:.3f}")
    # print(f"F1 score: {f1}")
    # plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
    #
    # df = pd.DataFrame(timestep_predictions)
    # df.to_csv(f"timestep_predictions_{SPEC}{'_IP' if IP else ''}{'_Tonotopic' if TONOTOPIC else ''}.csv", index=False)
    #
    # df = pd.DataFrame(y_per_timestep)
    # df.to_csv(f"true_labels_{SPEC}{'_IP' if IP else ''}{'_Tonotopic' if TONOTOPIC else ''}.csv", index=False)
