import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('tkagg')

total_kl = []
avg_kl = []

with open("csvs/kls.txt", "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("Total KL:"):
            total_kl.append(float(line.split(":")[1]))
        elif line.startswith("Average KL:"):
            avg_kl.append(float(line.split(":")[1]))

plt.figure(figsize=(6, 6))
plt.scatter(total_kl, avg_kl, alpha=0.7)
plt.xlabel("Total KL")
plt.ylabel("Average KL")
plt.title("Total KL vs Average KL")
plt.grid(True)
plt.show()

total_norm = (total_kl - np.min(total_kl)) / (np.max(total_kl) - np.min(total_kl))
avg_norm = (avg_kl - np.min(avg_kl)) / (np.max(avg_kl) - np.min(avg_kl))

plt.figure(figsize=(10, 4))
plt.plot(total_norm, label="Total KL (norm)")
plt.plot(avg_norm, label="Average KL (norm)")
plt.legend()
plt.title("Normalized KL comparison")
plt.show()
