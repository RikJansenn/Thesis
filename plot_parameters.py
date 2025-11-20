import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
matplotlib.use("TkAgg")

def plot_parameters_against_measure(measure):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.boxplot(data=data, x="N",  y=measure, ax=axes[0])
    sns.stripplot(data=data, x="N",  y=measure, ax=axes[0], color="black", alpha=0.5, jitter=True)
    axes[0].set_title(f"{measure}against number of neurons")

    sns.boxplot(data=data, x="sr", y=measure, ax=axes[1])
    sns.stripplot(data=data, x="sr", y=measure, ax=axes[1], color="black", alpha=0.5, jitter=True)
    axes[1].set_title(f"{measure} against spectral radius")

    sns.boxplot(data=data, x="lr", y=measure, ax=axes[2])
    sns.stripplot(data=data, x="lr", y=measure, ax=axes[2], color="black", alpha=0.5, jitter=True)
    axes[2].set_title(f"{measure} against leaky rate")

    plt.tight_layout()
    plt.savefig(f"plots/{measure}_vs_parameters")
    plt.show()

    # sns.pairplot(data, x_vars=["N", "sr", "lr"], y_vars=measure, kind="reg", height=4)
    # plt.savefig(f"plots/{measure}_vs_parameters_v2")
    # plt.show()

def plot_correlation(data, param1, param2, measure):
    x = data[param1] + np.random.normal(0, 0.04*data[param1].std(), size=len(data))  # jitter
    y = data[param2] + np.random.normal(0, 0.04*data[param2].std(), size=len(data))  # jitter

    plt.figure(figsize=(16, 12))
    sc = plt.scatter(x, y, c=data[measure], cmap="viridis", s=80)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"{param1} vs {param2}, colored by {measure}")
    plt.colorbar(sc, label=measure)
    plt.savefig(f"plots/{param1}_vs_{param2}")
    plt.show()


def get_n_best_parameters(amount):
    best = data.nsmallest(amount, "KL_mean")
    best_params = best[["N", "sr", "lr", "KL_mean"]]

    return best_params

def get_parameters_under_threshold(threshold):
    params = data[data["KL_mean"] < threshold]
    params = params[["N", "sr", "lr", "KL_mean"]].sort_values("KL_mean")

    return params

if __name__ == "__main__":
    data = pd.read_csv("results_parameters_ip_100_iter.csv")
    plot = False

    if plot:
        # Plot parameters vs means and stds
        measures = {"KL_mean", "entropy_mean", "KL_std", "entropy_std"}
        for measure in measures:
            plot_parameters_against_measure(measure)

        # Plot correlations between parameters
        params = {"N", "lr", "sr"}
        for param1 in params:
            for param2 in params:
                if param1 != param2:
                    plot_correlation(data, param1, param2, "KL_mean")

    # best_params = get_n_best_parameters(50)
    best_params = get_parameters_under_threshold(0.1)
    print(best_params)

    for col in ["N", "sr", "lr"]:
        coef, p = spearmanr(data[col], data["KL_mean"])
        print(f"Spearman correlation {col} vs KL_mean: {coef:.3f}, p-value: {p:.3e}")

    # plt.figure(figsize=(8,6))
    # plt.hist(data['KL_mean'], bins=60)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.scatter(data['entropy_mean'], data['entropy_std'])
    # plt.xlabel("entropy_std mean")
    # plt.ylabel("entropy_std std")
    # plt.title("entropy_std mean vs entropy_std")
    # plt.savefig("plots/entropy_mean_vs_entropy_std")
    # plt.show()
