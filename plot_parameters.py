import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Only here because otherwise matplotlib doesn't work for me
matplotlib.use("TkAgg")

def plot_parameters_against_measure(measure):
    """
    Plot the individual parameters against given measure (e.g. mean KL divergence)

    :param measure: which measure to plot against
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.boxplot(data=data, x="N",  y=measure, ax=axes[0])
    sns.stripplot(data=data, x="N",  y=measure, ax=axes[0], color="black", alpha=0.5, jitter=True)
    axes[0].set_title(f"{measure} against number of neurons")

    sns.boxplot(data=data, x="sr", y=measure, ax=axes[1])
    sns.stripplot(data=data, x="sr", y=measure, ax=axes[1], color="black", alpha=0.5, jitter=True)
    axes[1].set_title(f"{measure} against spectral radius")

    sns.boxplot(data=data, x="lr", y=measure, ax=axes[2])
    sns.stripplot(data=data, x="lr", y=measure, ax=axes[2], color="black", alpha=0.5, jitter=True)
    axes[2].set_title(f"{measure} against leaky rate")

    plt.tight_layout()
    plt.savefig(f"plots/{measure}_vs_parameters")
    plt.show()

def plot_correlation(data, param1, param2, measure):
    """
    Plot the correlation between parameters against given measure (e.g. mean KL divergence)

    :param data: dataframe with parameter and measure information
    :param param1: first parameter to plot
    :param param2: second parameter to plot
    :param measure: which measure to plot parameters against
    """
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

def get_parameters_under_threshold(threshold):
    """
    Get parameter sets resulting in KL mean under a given threshold

    :param threshold: threshold of KL mean
    :return: sorted parameter sets and their KL mean
    """
    params = data[data["KL_mean"] < threshold]
    params = params[["N", "sr", "lr", "KL_mean"]].sort_values("KL_mean")

    return params

if __name__ == "__main__":
    data = pd.read_csv("csvs/results_parameters_ip_100_iter_v2.csv")
    plot = True

    # Convert the dataframe into the dataframe version with just the mean and std measures per parameter combo
    # since that is what this code was written for
    param_cols = ["N", "sr", "lr"]
    data = (
        data
        .groupby(param_cols)
        .agg(
            KL_mean=("KL", "mean"),
            KL_std=("KL", "std"),
            entropy_mean=("entropy", "mean"),
            entropy_std=("entropy", "std")
        )
        .reset_index()
    )

    if plot:
        # Plot parameters vs means and stds
        measures = {"KL_mean", "entropy_mean", "KL_std", "entropy_std"}
        for measure in measures:
            plot_parameters_against_measure(measure)

        # Plot correlations between parameters
        combos = {("N", "sr"), ("N", "lr"), ("lr", "sr")}
        for combo in combos:
            plot_correlation(data, combo[0], combo[1], "KL_mean")

    best_params = get_parameters_under_threshold(0.1)
    print(best_params)
