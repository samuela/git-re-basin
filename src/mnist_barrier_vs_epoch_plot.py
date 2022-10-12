import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

import matplotlib_style as _
from plot_utils import loss_barrier_is_nonnegative

max_epoch = 25

api = wandb.Api()
# run = api.run("skainswo/git-re-basin/begnvj15")
artifact = Path(
    api.artifact("skainswo/git-re-basin/mnist_permutation_eval_vs_epoch:v0").download())

with open(artifact / "permutation_eval_vs_epoch.pkl", "rb") as f:
  interp_eval_vs_epoch = pickle.load(f)

train_loss_interp = np.array([x["train_loss_interp"] for x in interp_eval_vs_epoch])
train_barrier_vs_epoch = np.max(train_loss_interp,
                                axis=1) - 0.5 * (train_loss_interp[:, 0] + train_loss_interp[:, -1])

test_loss_interp = np.array([x["test_loss_interp"] for x in interp_eval_vs_epoch])
test_barrier_vs_epoch = np.max(test_loss_interp,
                               axis=1) - 0.5 * (test_loss_interp[:, 0] + test_loss_interp[:, -1])

fig = plt.figure()
# fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)

ax.arrow(5, 0.51, -4, 0.085, alpha=0.25)
ins1 = ax.inset_axes((0.2, 0.7, 0.25, 0.25))
ins1.plot(train_loss_interp[0, :])
ins1.plot(test_loss_interp[0, :], linestyle="dashed")
ins1.set_xticks([])
ins1.set_yticks([])

ax.arrow(21, 0.2, 4, -0.2, alpha=0.25)
ins2 = ax.inset_axes((0.7, 0.3, 0.25, 0.25))
ins2.plot(train_loss_interp[25, :])
ins2.plot(test_loss_interp[25, :], linestyle="dashed")
ins2.set_xticks([])
ins2.set_yticks([])
ymin, ymax = ins2.get_ylim()
ins2.set_ylim((ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))

ax.plot(
    1 + np.arange(max_epoch),
    train_barrier_vs_epoch[:max_epoch],
    marker="o",
    linewidth=2,
    label="Train",
)
ax.plot(
    1 + np.arange(max_epoch),
    test_barrier_vs_epoch[:max_epoch],
    marker="^",
    linestyle="dashed",
    linewidth=2,
    label="Test",
)

loss_barrier_is_nonnegative(ax)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss barrier")
ax.set_title(f"MNIST")
# ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/mnist_mlp_barrier_vs_epoch.png", dpi=300)
plt.savefig("figs/mnist_mlp_barrier_vs_epoch.eps")
plt.savefig("figs/mnist_mlp_barrier_vs_epoch.pdf")
