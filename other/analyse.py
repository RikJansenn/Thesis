import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
matplotlib.use('tkagg')

df = pd.read_csv("../downloads/results_parameter_sweep")
df.head()

group_cols = ["set", "N", "sr", "lr", "sigma"]

summary = (
    df
    .groupby(group_cols)
    .agg(
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_std=("Accuracy", "std"),
        KL_mean=("KL", "mean"),
        KL_std=("KL", "std"),
    )
    .reset_index()
)

print(
    df.groupby("sr")["Effective_sr"]
      .agg(["mean", "std", "min", "max"])
      .sort_index()
)

# plt.scatter(df["sr"], df["Effective_sr"], alpha=0.6)
# plt.plot([df["sr"].min(), df["sr"].max()],
#          [df["sr"].min(), df["sr"].max()], linestyle='--')
# plt.xlabel("sr")
# plt.ylabel("Effective_sr")
# plt.title("Effective_sr vs sr")
# plt.show()
#
# plt.scatter(df["KL"], df["Accuracy"], c=df["lr"])
# plt.xlabel("KL")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs KL divergence")
# plt.show()

print(summary.sort_values("Accuracy_mean", ascending=False))

print(
    df.groupby("N")["Accuracy"]
      .agg(["mean", "std"])
      .sort_index()
)

print(
    df.groupby("sr")["Accuracy"]
      .agg(["mean", "std"])
      .sort_index()
)

print(
    df.groupby("lr")["Accuracy"]
      .agg(["mean", "std"])
      .sort_index()
)

print(
    df.groupby("N")["KL"]
      .agg(["mean", "std"])
      .sort_index()
)

print(
    df.groupby("sr")["KL"]
      .agg(["mean", "std"])
      .sort_index()
)

print(
    df.groupby("lr")["KL"]
      .agg(["mean", "std"])
      .sort_index()
)