import argparse
import pickle
from pathlib import Path
import jax
import torch
torch.set_default_tensor_type(torch.FloatTensor)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax.serialization import from_bytes
from jax import random
from tqdm import tqdm

from resnet import ResNet50
from cifar10_resnet20_train import BLOCKS_PER_GROUP, ResNet, make_stuff
from datasets import ImageNet
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import (
    apply_permutation,
    resnet50_permutation_spec,
    weight_matching,
)
import pickle

import numpy as np
import torchmetrics
import tqdm


def plot_interp_loss(
    epoch,
    lambdas,
    train_loss_interp_naive,
    test_loss_interp_naive,
    train_loss_interp_clever,
    test_loss_interp_clever,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lambdas,
        train_loss_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train, naïve interp.",
    )
    ax.plot(
        lambdas,
        test_loss_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test, naïve interp.",
    )
    ax.plot(
        lambdas,
        train_loss_interp_clever,
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train, permuted interp.",
    )
    ax.plot(
        lambdas,
        test_loss_interp_clever,
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test, permuted interp.",
    )
    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"Loss landscape between the two models (epoch {epoch})")
    ax.legend(loc="upper right", framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_interp_acc(
    epoch,
    lambdas,
    train_acc_interp_naive,
    test_acc_interp_naive,
    train_acc_interp_clever,
    test_acc_interp_clever,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lambdas,
        train_acc_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train, naïve interp.",
    )
    ax.plot(
        lambdas,
        test_acc_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test, naïve interp.",
    )
    ax.plot(
        lambdas,
        train_acc_interp_clever,
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train, permuted interp.",
    )
    ax.plot(
        lambdas,
        test_acc_interp_clever,
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test, permuted interp.",
    )
    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"Accuracy between the two models (epoch {epoch})")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # args = parser.parse_args()

    model = ResNet50(n_classes=1000)

    def load_model(filepath):
        with open(filepath, "rb") as fh:
            return from_bytes(
                model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"],
                fh.read(),
            )

    model_a = pickle.load(open("../a_variables.pkl", "rb"))
    model_b = pickle.load(open("../b_variables.pkl", "rb"))

    # train_ds, test_ds = load_cifar10()

    permutation_spec = resnet50_permutation_spec()

    @jax.jit
    def model_apply1(variables, images):
        return model.apply(variables, images, mutable=["batch_stats"])

    @jax.jit
    def model_apply2(variables, images):
        return model.apply(variables, images)

    accs = []
    for key in range(10):
        final_permutation = weight_matching(
            random.PRNGKey(key),
            permutation_spec,
            flatten_params(model_a["params"]),
            flatten_params(model_b["params"]),
        )

        model_b_clever = model_b["params"]
        model_b_clever = unflatten_params(
            apply_permutation(permutation_spec, final_permutation, flatten_params(model_b["params"])))

        dataloader = ImageNet()
        loss = torch.nn.CrossEntropyLoss()

        # for lam in np.linspace(0, 1, 25):
        lam = 0.5
        model_ab = {
            "params": lerp(lam, model_a["params"], model_b_clever),
            # "params": model_a["params"],
            "batch_stats": model_a["batch_stats"]
        }
        # from flax.core import unfreeze
        # model_ab = unfreeze(pickle.load(open("final-perm-interp.pkl", "rb")))

        for (images, labels), _ in zip(dataloader.train_loader, tqdm.trange(100)):
            images = jnp.moveaxis(jnp.asarray(images.numpy()), 1, 3)
            y, new_batch_stats = model_apply1(model_ab, images)
            model_ab["batch_stats"] = new_batch_stats['batch_stats']
            # logits = torch.tensor(np.asarray(y))
            # print((labels.unsqueeze(1) == logits.argmax(dim=1, keepdim=True)).sum(), loss(logits, labels))

        acc1 = torchmetrics.Accuracy()
        acc5 = torchmetrics.Accuracy(top_k=5)
        sum_loss = 0
        for images, labels in dataloader.val_loader:
            images = jnp.moveaxis(jnp.asarray(images.numpy()), 1, 3)
            y = model_apply2(model_ab, images)
            logits = torch.tensor(np.asarray(y))
            sum_loss += loss(logits, labels)
            acc1(logits, labels)
            acc5(logits, labels)

            # print({
            #     # "lambda": lam,
            #     # "loss": (sum_loss / len(dataloader.val_loader)).item(),
            #     "acc1": acc1.compute().item(),
            #     "acc5": acc5.compute().item()
            # })

        accs.append(acc1.compute().item())


# if __name__ == "__main__":
#   main()
