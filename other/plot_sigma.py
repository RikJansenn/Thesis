import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from scipy.stats import spearmanr

# Only here because otherwise matplotlib doesn't work for me
matplotlib.use("TkAgg")

def plot_parameter_conditioned(data, param, measure):
    """
    Plot a single parameter against a measure, with separate subplots
    for each combination of the other parameters.

    :param data: dataframe
    :param param: parameter to plot on x-axis ('N', 'sr', or 'lr')
    :param measure: measure to plot on y-axis
    """

    other_params = [p for p in ["N", "sr", "lr", "sigma"] if p != param]

    groups = list(data.groupby(other_params))
    n_plots = len(groups)

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
    )
    axes = axes.flatten()

    for ax, ((val1, val2), df_group) in zip(axes, groups):
        sns.boxplot(
            data=df_group,
            x=param,
            y=measure,
            ax=ax
        )
        sns.stripplot(
            data=df_group,
            x=param,
            y=measure,
            ax=ax,
            color="black",
            alpha=0.5,
            jitter=True
        )

        ax.set_title(f"{other_params[0]}={val1}, {other_params[1]}={val2}")

    # Hide unused axes
    for ax in axes[len(groups):]:
        ax.axis("off")

    fig.suptitle(f"{measure} vs {param}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"moreplots/{measure}_vs_{param}_conditioned")
    plt.show()

def plot_sigma(data, param, measure):
    ax = sns.boxplot(data=data, x="sigma", y=measure)
    sns.stripplot(data=data, x="sigma", y=measure, ax=ax, color="black", alpha=0.5, jitter=True)
    ax.set_title(f"{measure} against sigma for sr=1, lr=1")

    plt.tight_layout()
    plt.savefig(f"moreplots/{measure}_vs_parameters_all_2")
    plt.show()

def plot_correlation(data, param1, param2, measure):
    """
    Plot the correlation between parameters against given measure (e.g. mean KL divergence)

    :param data: dataframe with parameter and measure information
    :param param1: first parameter to plot
    :param param2: second parameter to plot
    :param measure: which measure to plot parameters against
    """
    x = data[param1] + np.random.normal(0, 0.09*data[param1].std(), size=len(data))  # jitter
    y = data[param2] + np.random.normal(0, 0.09*data[param2].std(), size=len(data))  # jitter

    plt.figure(figsize=(16, 12))
    sc = plt.scatter(x, y, c=data[measure], cmap="viridis", alpha=0.5, s=80)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"{param1} vs {param2}, colored by {measure}")
    plt.colorbar(sc, label=measure)
    plt.savefig(f"moreplots/{param1}_vs_{param2}_all")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../downloads/results_parameter_sweep")
    data_sigma = pd.read_csv("../downloads/results_parameter_sweep_sigma2")

    data = data[(data["N"] == 1200) & (data["sr"] == 0.8) & (data["lr"] == 0.94)]
    data = pd.concat([data, data_sigma], ignore_index=True)

    plot = True
    convert = False


    # Convert the dataframe into the dataframe version with just the mean and std measures per parameter combo
    # since that is what this code was written for
    if convert:
        param_cols = ["N", "sr", "lr", "sigma"]
        data = (
            data
            .groupby(param_cols)
            .agg(
                KL_mean=("KL", "mean"),
                KL_std=("KL", "std"),
                Accuracy_mean=("Accuracy", "mean"),
                Accuracy_std=("Accuracy", "std"),
                # entropy_mean=("entropy", "mean"),
                # entropy_std=("entropy", "std")
            )
            .reset_index()
        )
    # threeD_plot(data, 0.1)
    # parallel_coordinates(data)

    if not convert:
        if plot:
            # Plot parameters vs means and stds
            for measure in ["KL", "Accuracy"]:
                for param in ["sigma"]:
                    plot_sigma (data, param, measure)
                    # plot_parameter_conditioned(data, param, measure)

            # Plot correlations between parameters
            combos = {("N", "sr"), ("N", "lr"), ("lr", "sr")}
            for combo in combos:
                plot_correlation(data, combo[0], combo[1], "KL")

    # best_params = get_parameters_under_threshold(0.1)
    # print(best_params)
