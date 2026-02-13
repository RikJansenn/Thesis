from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import f1_score
from utils import get_KL_divergence_and_entropy
from models import ShallowNetwork
from reservoirpy.observables import effective_spectral_radius
import pandas as pd

def run_experiment(params, X_train, Y_train, X_test, Y_test, IP=True, TONOTOPIC=False, p=0.1):
    N, sr, lr, sigma, iteration = params
    print("Creating model...")
    model = ShallowNetwork(N=N, sr=sr, lr=lr, sigma=sigma, input_dim=X_train[0].shape[1], IP=IP)

    if IP:
        print("Applying IP...")
        model.apply_ip(p)

    # model.create_input_weights(p)

    # Compute effective sr
    eff_sr = effective_spectral_radius(model.reservoir.W, lr=lr)

    # Compute KL/entropy
    idx = np.random.randint(len(X_test))
    states = model.reservoir.run(X_test[idx])
    kl, ent = get_KL_divergence_and_entropy(states, sigma)

    # Train
    print("Training model...")
    model.train(X_train, Y_train)

    # Save reservoir
    with open(f"../outputs2/reservoirs/reservoir_N{N}_sr{sr}_lr{lr}_sigma{sigma}_{iteration}.pkl", "wb") as f:
        pickle.dump(model.reservoir, f)

    # Test
    print("Testing model...")
    acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)
    f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "N": N, "sr": sr, "lr": lr, "sigma": sigma, "iteration": iteration,
        "Effective_sr": eff_sr, "KL": kl, "Entropy": ent, "Accuracy": acc, "F1": f1
    }

    df = pd.DataFrame(timestep_predictions)
    df.to_csv(f"../outputs2/timestep_predictions_{SPEC}{'_IP' if IP else ''}{'_Tonotopic' if TONOTOPIC else ''}.csv", index=False)

    df = pd.DataFrame(y_per_timestep)
    df.to_csv(f"../outputs2/true_labels_{SPEC}{'_IP' if IP else ''}{'_Tonotopic' if TONOTOPIC else ''}.csv", index=False)

def parameter_sweep(N_values, lr_values, sr_values, sigmas, iterations, X_train, Y_train, X_test, Y_test, IP, TONOTOPIC, n_jobs):
    param_combinations = list(product(N_values, lr_values, sr_values, sigmas, range(iterations)))

    results = [
        run_experiment(params, X_train, Y_train, X_test, Y_test, IP, TONOTOPIC)
        for params in tqdm(param_combinations, desc="Parameter sweep")
    ]
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(run_experiment)(params, X_train, Y_train, X_test, Y_test)
    #     for params in tqdm(param_combinations, desc="Parameter sweep")
    # )
    return results

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