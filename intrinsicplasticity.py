from reservoirpy.nodes import IPReservoir
from reservoirpy.datasets import narma
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('tkagg')

reservoir = IPReservoir(100, mu=0.0, sigma=0.1, sr=0.95, activation="tanh", epochs=10)

x = narma(1000)
states_before = reservoir.run(x)
_ = reservoir.fit(x, warmup=100)
states_after = reservoir.run(x)

# 3. Ensure arrays (not lists)
if isinstance(states_before, list):
    states_before = np.vstack(states_before)
if isinstance(states_after, list):
    states_after = np.vstack(states_after)

# 4. Flatten neuron activations
a_before = states_before.ravel()
a_after = states_after.ravel()

# 5. Compute KDE (smoothed probability density)
x_vals = np.linspace(-1, 1, 400)
pdf_before = gaussian_kde(a_before)(x_vals)
pdf_after = gaussian_kde(a_after)(x_vals)

# 6. Compute target Gaussian distribution (mu, sigma from reservoir)
mu = reservoir.mu
sigma = reservoir.sigma
target_pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x_vals - mu) ** 2) / (2 * sigma**2))
target_pdf /= np.trapezoid(target_pdf, x_vals)  # normalize

# 7. Plot in ReservoirPy style

# Plot after
plt.figure(figsize=(8, 5))

# scatter: each activation point (for texture)
plt.scatter(a_after, np.random.normal(0, 0.2, size=len(a_after)),
            s=1, alpha=0.1, color="black")

# smooth lines
plt.plot(x_vals, pdf_after, color="orangered", lw=2, label="Global activation")
plt.plot(x_vals, target_pdf, "--", color="royalblue", lw=2, label="Target distribution")

plt.xlabel("Reservoir activations")
plt.ylabel("Probability density")
plt.xlim(-1, 1)
plt.ylim(0, max(pdf_after)*1.2)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

# scatter: each activation point (for texture)
plt.scatter(a_before, np.random.normal(0, 0.2, size=len(a_before)),
            s=1, alpha=0.1, color="black")

# smooth lines
plt.plot(x_vals, pdf_before, color="orangered", lw=2, label="Global activation")
plt.plot(x_vals, target_pdf, "--", color="royalblue", lw=2, label="Target distribution")

plt.xlabel("Reservoir activations")
plt.ylabel("Probability density")
plt.xlim(-1, 1)
plt.ylim(0, max(pdf_before)*1.2)
plt.legend()
plt.tight_layout()
plt.show()