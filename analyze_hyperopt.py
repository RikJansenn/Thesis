from reservoirpy.hyper import plot_hyperopt_report
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

fig = plot_hyperopt_report("hyperopt/hyperopt_stage1_learning_rates", ("learning_rate", "ridge"),  metric="r2")
fig.set_size_inches(24, 16)

fig.savefig("hyperopt_plot_stage1.png", dpi=300)

plt.show()
