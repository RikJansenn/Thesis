import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("tkAgg")

df = pd.read_csv("results_parameter_sweep_ipreservoir")
grouped = (
    df
    .groupby(["sr", "sigma"])
    .agg({
        "Accuracy": "mean",
        "F1": "mean",
        "KL": "mean",
        "Effective_sr": "mean"
    })
    .reset_index()
)

import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data=grouped, x="sigma", y="Accuracy", hue="sr", marker="o")
plt.show()