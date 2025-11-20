import matplotlib as plot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')

data = pd.read_csv("results_epochs_ip_20_iter.csv")

plt.figure(figsize=(8, 6))
plt.plot(data['epochs'], data['KL_mean'], label="KL Divergence")
#plt.plot(entropy, label="Entropy")
plt.xlabel("Epochs")
plt.ylabel("KL Divergence")
plt.title("KL Divergence over Training Epochs for IP")
plt.savefig("kl_divergence_epochs.png", dpi=300)
