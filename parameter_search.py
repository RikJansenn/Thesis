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

data_len = 5000
train_len = 4000

hyperopt_config = {
    "exp": "hyperopt_ips",    # the experimentation name
    "hp_max_evals": 100,              # the number of different sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to choose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 925],
        "sr": ["choice", 0.73],
        "lr": ["choice", 0.95],
        "mu": ["choice", 0.0],
        "sigma": ["loguniform", 0.05, 1],
        "learning_rate": ["loguniform", 1e-6, 1e-1],
        "epochs": ["choice", 2],
        "input_scaling": ["choice", 1.0],
        "input_connectivity": ["choice", 0.73],
        "ridge": ["choice", 1e-6],
        "seed": ["choice", 1234]
    }
}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

def objective(dataset, config, *, input_scaling, input_connectivity, N, sr, lr, ridge, mu, sigma, learning_rate, epochs, seed):
    # This step may vary depending on what you put inside 'dataset'
    x_train, x_test, y_train, y_test = dataset

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []; r2s = []; kl_values = []; entropy_values = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = IPReservoir(
            units=int(N),
            sr=sr,
            lr=lr,
            mu=mu,
            sigma=sigma,
            learning_rate=learning_rate,
            epochs=epochs,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            activation="tanh",
            seed=variable_seed
        )
        readout = Ridge(ridge=ridge, input_dim=N)

        # Train your model and test your model.
        # Apply IP
        print("Applying IP...")
        reservoir.fit(x_train, warmup=100)

        # Train readout layer
        print("Training...")
        x_train = reservoir.run(x_train)
        readout.fit(x_train, y_train)

        x_test = reservoir.run(x_test)
        predictions = readout.run(x_test)

        # Compute KL divergence and entropy on activations
        print("Computing KL divergence and entropy...")
        flat_states = x_train.flatten()
        kl, entropy = kl_divergence_and_entropy_fast(flat_states, mu=reservoir.mu, sigma=reservoir.sigma)

        # Metrics to be tracked
        # print("KL Divergence:", kl, "Entropy:", entropy)
        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)
        kl_values.append(kl)
        entropy_values.append(entropy)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.

    return {
        'loss': np.mean(losses),
        'r2': np.mean(r2s),
        'kl_mean': np.mean(kl_values),
        'kl_std': np.std(kl_values),
        'entropy_mean': np.mean(entropy_values),
        'entropy_std': np.std(entropy_values)
    }

def kl_divergence_and_entropy_fast(observed_samples, mu, sigma, bins=200):
    x_min = min(-3, observed_samples.min())
    x_max = max(3, observed_samples.max())
    p_hist, bin_edges = np.histogram(observed_samples, bins=bins, range=(x_min, x_max), density=True)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # bin centers
    q = norm.pdf(x, loc=mu, scale=sigma)

    # Normalize PDFs
    dx = x[1] - x[0]
    p_hist /= np.trapezoid(p_hist, x)
    q /= np.trapezoid(q, x)

    epsilon = 1e-12
    entropy = -np.trapezoid(p_hist * np.log(p_hist + epsilon), x)
    kl = np.trapezoid(p_hist * np.log((p_hist + epsilon) / (q + epsilon)), x)

    return kl, entropy

def create_data():
    X, Y = narma(data_len)
    X = X[-len(Y):]

    X_train, Y_train = X[:train_len], Y[:train_len]
    X_test, Y_test = X[train_len:], Y[train_len:]

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = create_data()
    dataset = (X_train, X_test, Y_train, Y_test)

    best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

    print(best)
