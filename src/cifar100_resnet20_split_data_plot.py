import matplotlib.pyplot as plt
import numpy as np
import wandb

import matplotlib_style as _

api = wandb.Api()
# width multiplier = 32
weight_matching_run = api.run("skainswo/playing-the-lottery/r9tyrenf")
ensembling_run = api.run("skainswo/playing-the-lottery/2nwx9yyu")
combined_training_run = api.run("skainswo/playing-the-lottery/f40w12z7")

### Loss plot
fig = plt.figure()
ax = fig.add_subplot(111)
lambdas = np.linspace(0, 1, 25)

# Naive
# ax.plot(lambdas,
#         np.array(weight_matching_run.summary["train_loss_interp_naive"]),
#         color="grey",
#         linewidth=2,
#         label=f"Naïve weight interp.")
ax.plot(lambdas,
        np.array(weight_matching_run.summary["test_loss_interp_naive"]),
        color="grey",
        linewidth=2,
        linestyle="dashed",
        label="Naïve weight interp.")

# Ensembling
ax.plot(lambdas,
        np.array(ensembling_run.summary["test_loss_interp"]),
        color="tab:purple",
        marker="2",
        linestyle="dashed",
        linewidth=2,
        label="Ensembling")

# Weight matching
# ax.plot(lambdas,
#         np.array(weight_matching_run.summary["train_loss_interp_clever"]),
#         color="tab:green",
#         marker="v",
#         linewidth=2,
#         label="Weight matching weight interp.")
ax.plot(lambdas,
        np.array(weight_matching_run.summary["test_loss_interp_clever"]),
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2,
        label="Weight matching")

ax.axhline(y=combined_training_run.summary["test_loss"],
           linewidth=2,
           linestyle="dashed",
           label="Combined data training")

ax.set_ylim(1, 5.1)
ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$\nDataset $A$", "Model $B$\nDataset $B$"])
ax.set_ylabel("Test loss")
ax.set_title("Split data training")
ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/cifar100_resnet20_split_data.png", dpi=300)
plt.savefig("figs/cifar100_resnet20_split_data.eps")
plt.savefig("figs/cifar100_resnet20_split_data.pdf")
