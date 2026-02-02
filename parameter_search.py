from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
import numpy as np
from reservoirpy.datasets import to_forecasting
from reservoirpy.hyper import research
import json
from reservoirpy.datasets import mackey_glass
from reservoirpy.datasets import narma
import random
from scipy.stats import gaussian_kde, norm
from shallow_network import *

hyperopt_config = {
    "exp": "ridge_search_1",    # the experimentation name
    "hp_max_evals": 1,             # the number of different sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to choose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 4,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 1000],
        "sr": ["choice", 0.73],
        "lr": ["choice", 0.95],
        "mu": ["choice", 0.0],
        "sigma": ["choice", 0.1],
        "epochs": ["choice", 2],
        "input_scaling": ["choice", 1.0],
        "input_connectivity": ["choice", 0.73],
        "ridge": ["choice", 1e-2],
        "seed": ["choice", 1234]
    }
}

IP = False
TONOTOPIC = False

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

def objective(dataset, config, *, input_scaling, input_connectivity, N, sr, lr, ridge, mu, sigma, epochs, seed):
    # This step may vary depending on what you put inside 'dataset'
    X_train, X_test, Y_train, Y_test = dataset

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    accuracies = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = IPReservoir(
            units=int(N),
            sr=sr,
            lr=lr,
            mu=mu,
            sigma=sigma,
            epochs=epochs,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            activation="tanh",
            seed=variable_seed
        )
        readout = Ridge(ridge=ridge, input_dim=N)

        # Train your model and test your model.
        print(ridge)
        input_d = X_train[0].shape[1]
        p = 0.1
        if TONOTOPIC:
            W_in, W = create_tonotopic_mapping(N, sr)

            reservoir.Win = W_in
            reservoir.W = W

        if IP:
            reservoir = apply_ip(reservoir, input_d)

            # Set input matrix to correct shape for spectrograms
            reservoir.Win = create_input_weights(reservoir, input_d, p)

        # Reshape input matrix to have only positive values
        if not IP and not TONOTOPIC:
            reservoir.Win = create_input_weights(reservoir, input_d, p)

        readout = train_model(X_train, Y_train, reservoir, readout)

        # Test model
        accuracy, _, _, _, _ = test_model(reservoir, readout, X_test, Y_test)

        # Change the seed between instances
        variable_seed += 1

        accuracies.append(accuracy)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.

    return {
        'loss': accuracy,
    }

def create_training_data_paramsearch():
    data = np.load("dataset_train.npz")
    X = data["specs"]
    Y = data["targets"]

    # Take a random 10% of the data
    X_sample, X_remaining, Y_sample, Y_remaining = train_test_split(
        X, Y, test_size=0.9, random_state=42, shuffle=True
    )

    # Split that 10% into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_sample, Y_sample, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    N = 500
    sr = 0.8
    lr = 1

    X_train, X_test, Y_train, Y_test = create_training_data_paramsearch()

    dataset = (X_train, X_test, Y_train, Y_test)
    best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

    print(best)
