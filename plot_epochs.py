import matplotlib as plot
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

x = [1, 2, 3, 4, 5]
kl = [0.0194, 0.0141, 0.0142, 0.0138, 0.0138]
entropy = [0.536, 0.0988, 0.165, 0.161, 0.178, 0.178]

plt.figure(figsize=(8, 6))
plt.plot(x, kl, label="KL Divergence")
#plt.plot(entropy, label="Entropy")
plt.xlabel("Epochs")
plt.ylabel("KL Divergence")
plt.title("KL Divergence over Training Epochs for IP")
plt.savefig("kl_divergence_epochs.png", dpi=300)
