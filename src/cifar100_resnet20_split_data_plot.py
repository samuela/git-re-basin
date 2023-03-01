import pickle

import matplotlib.pyplot as plt
import numpy as np
import wandb

import matplotlib_style as _
from utils import lerp

api = wandb.Api()
# width multiplier = 32
# weight_matching_run = api.run("skainswo/git-re-basin/r9tyrenf")  # lerp
weight_matching_run = api.run("skainswo/git-re-basin/3so345lw")  # lerp, with top-5
# weight_matching_run = api.run("skainswo/git-re-basin/bcu96ank")  # slerp
ensembling_run = api.run("skainswo/git-re-basin/2nwx9yyu")
combined_training_run = api.run("skainswo/git-re-basin/f40w12z7")

ensembling_data = pickle.load(open("../cifar100_interp_logits.pkl", "rb"))

### Loss plot
fig = plt.figure()
ax = fig.add_subplot(111)
lambdas = np.linspace(0, 1, 25)

# Naive
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_naive"],
#         color="grey",
#         linewidth=2,
#         label=f"Naïve weight interp.")
ax.plot(lambdas,
        weight_matching_run.summary["test_loss_interp_naive"],
        color="grey",
        linewidth=2,
        linestyle="dashed",
        label="Naïve weight interp.")

# Ensembling
ax.plot(lambdas,
        ensembling_run.summary["test_loss_interp"],
        color="tab:purple",
        marker="2",
        linestyle="dashed",
        linewidth=2,
        label="Ensembling")

# Weight matching
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_clever"],
#         color="tab:green",
#         marker="v",
#         linewidth=2,
#         label="Weight matching weight interp.")
ax.plot(lambdas,
        weight_matching_run.summary["test_loss_interp_clever"],
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

plt.savefig("figs/cifar100_resnet20_split_data_test_loss.png", dpi=300)
plt.savefig("figs/cifar100_resnet20_split_data_test_loss.pdf")

### Top-1 Accuracy plot
fig = plt.figure()
ax = fig.add_subplot(111)
lambdas = np.linspace(0, 1, 25)

# Naive
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_naive"],
#         color="grey",
#         linewidth=2,
#         label=f"Naïve weight interp.")
ax.plot(lambdas,
        100 * np.array(weight_matching_run.summary["test_acc1_interp_naive"]),
        color="grey",
        linewidth=2,
        linestyle="dashed",
        label="Naïve weight interp.")

# Ensembling
ax.plot(lambdas,
        100 * np.array(ensembling_run.summary["test_acc_interp"]),
        color="tab:purple",
        marker="2",
        linestyle="dashed",
        linewidth=2,
        label="Ensembling")

# Weight matching
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_clever"],
#         color="tab:green",
#         marker="v",
#         linewidth=2,
#         label="Weight matching weight interp.")
ax.plot(lambdas,
        100 * np.array(weight_matching_run.summary["test_acc1_interp_clever"]),
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2,
        label="Weight matching")

ax.axhline(y=100 * combined_training_run.summary["test_accuracy"],
           linewidth=2,
           linestyle="dashed",
           label="Combined data training")

# ax.set_ylim(1, 5.1)
ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$\nDataset $A$", "Model $B$\nDataset $B$"])
ax.set_ylabel("Top-1 accuracy")
ax.set_title("CIFAR-100, Split data training")
# ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/cifar100_resnet20_split_data_test_acc1.png", dpi=300)
plt.savefig("figs/cifar100_resnet20_split_data_test_acc1.pdf")

### Top-5 Accuracy plot
fig = plt.figure()
ax = fig.add_subplot(111)
lambdas = np.linspace(0, 1, 25)

# Naive
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_naive"],
#         color="grey",
#         linewidth=2,
#         label=f"Naïve weight interp.")
ax.plot(lambdas,
        100 * np.array(weight_matching_run.summary["test_acc5_interp_naive"]),
        color="grey",
        linewidth=2,
        linestyle="dashed",
        label="Naïve weight interp.")

# Weight matching
# ax.plot(lambdas,
#         weight_matching_run.summary["train_loss_interp_clever"],
#         color="tab:green",
#         marker="v",
#         linewidth=2,
#         label="Weight matching weight interp.")
ax.plot(
    lambdas,
    100 * np.array(weight_matching_run.summary["test_acc5_interp_clever"]),
    color="tab:green",
    marker="^",
    linestyle="dashed",
    linewidth=2,
    label="Weight matching",
)

# Ensembling
def lam_top5_acc(lam):
  logits = lerp(lam, ensembling_data["a_test_logits"], ensembling_data["b_test_logits"])
  labels = ensembling_data["test_dataset"]["labels"]
  top5_num_correct = np.sum(np.isin(labels[:, np.newaxis], np.argsort(logits, axis=-1)[:, -5:]))
  return top5_num_correct / len(labels)

ax.plot(
    lambdas,
    [100 * lam_top5_acc(lam) for lam in lambdas],
    color="tab:purple",
    marker="2",
    linestyle="dashed",
    linewidth=2,
    label="Ensembling",
)

# See https://wandb.ai/skainswo/git-re-basin/runs/10kebhlr?workspace=user-skainswo for the calculation of this value
ax.axhline(y=100.0, linewidth=2, linestyle="dashed", label="Combined data training")

# ax.set_ylim(1, 5.1)
ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$\nDataset $A$", "Model $B$\nDataset $B$"])
ax.set_ylabel("Top-5 accuracy")
ax.set_title("CIFAR-100, Split data training")
# ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/cifar100_resnet20_split_data_test_acc5.png", dpi=300)
plt.savefig("figs/cifar100_resnet20_split_data_test_acc5.pdf")
