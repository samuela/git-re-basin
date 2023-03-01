import matplotlib.pyplot as plt
import numpy as np
import wandb

import matplotlib_style as _

api = wandb.Api()
wm32_run = api.run("skainswo/git-re-basin/223t7txl")

# ins2.plot(all_runs[-1].summary["train_loss_interp_clever"])
# ins2.plot(all_runs[-1].summary["test_loss_interp_clever"], linestyle="dashed")
# ins2.set_xticks([])
# ins2.set_yticks([])
# ymin, ymax = ins2.get_ylim()
# ins2.set_ylim((ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))

### Loss plot
fig = plt.figure()
ax = fig.add_subplot(111)
lambdas = np.linspace(0, 1, 25)

# Naive
ax.plot(
    lambdas,
    wm32_run.summary["train_loss_interp_naive"],
    color="grey",
    linewidth=2,
    # label="Naïve",
)
ax.plot(
    lambdas,
    wm32_run.summary["test_loss_interp_naive"],
    color="grey",
    linewidth=2,
    linestyle="dashed",
)

# Activation matching
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_loss_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2,
#         label=f"Activation matching")
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["test_loss_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2,
#         linestyle="dashed")

# Weight matching
ax.plot(lambdas,
        wm32_run.summary["train_loss_interp_clever"],
        color="tab:green",
        marker="^",
        linewidth=2,
        label="Weight matching")
ax.plot(
    lambdas,
    wm32_run.summary["test_loss_interp_clever"],
    color="tab:green",
    marker="^",
    linestyle="dashed",
    linewidth=2,
)

# STE matching
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["train_loss_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linewidth=2,
#         label=f"STE matching")
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["test_loss_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linestyle="dashed",
#         linewidth=2)

# ax.plot([], [], color="grey", linewidth=2, label="Train")
# ax.plot([], [], color="grey", linewidth=2, linestyle="dashed", label="Test")

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Loss")
ax.set_title(f"CIFAR-10, ResNet20 (32× width)")
# ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/cifar10_resnet20_loss_interp.png", dpi=300)
plt.savefig("figs/cifar10_resnet20_loss_interp.pdf")

### Accuracy plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Naive
ax.plot(
    lambdas,
    100 * np.array(wm32_run.summary["train_acc_interp_naive"]),
    color="grey",
    linewidth=2,
    label="Train",
)
ax.plot(
    lambdas,
    100 * np.array(wm32_run.summary["test_acc_interp_naive"]),
    color="grey",
    linewidth=2,
    linestyle="dashed",
    label="Test",
)

# Activation matching
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2,
#         linestyle="dashed")

# Weight matching
ax.plot(
    lambdas,
    100 * np.array(wm32_run.summary["train_acc_interp_clever"]),
    color="tab:green",
    marker="^",
    linewidth=2,
)
ax.plot(
    lambdas,
    100 * np.array(wm32_run.summary["test_acc_interp_clever"]),
    color="tab:green",
    marker="^",
    linestyle="dashed",
    linewidth=2,
)

# STE matching
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linestyle="dashed",
#         linewidth=2)

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Accuracy")
ax.set_title("CIFAR-10, ResNet20 (32× width)")
# ax.legend(loc="lower right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/cifar10_resnet20_accuracy_interp.png", dpi=300)
plt.savefig("figs/cifar10_resnet20_accuracy_interp.pdf")
