import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax.serialization import from_bytes
from jax import random
from tqdm import tqdm

from cifar10_resnet20_train import BLOCKS_PER_GROUP, ResNet, make_stuff
from datasets import load_cifar10
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import (apply_permutation, resnet20_permutation_spec, weight_matching)

def plot_interp_loss(epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                     train_loss_interp_clever, test_loss_interp_clever):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_loss_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_loss_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_loss_interp_clever,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permuted interp.")
  ax.plot(lambdas,
          test_loss_interp_clever,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permuted interp.")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Loss")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
  ax.set_title(f"Loss landscape between the two models (epoch {epoch})")
  ax.legend(loc="upper right", framealpha=0.5)
  fig.tight_layout()
  return fig

def plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_acc_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_acc_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_acc_interp_clever,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permuted interp.")
  ax.plot(lambdas,
          test_acc_interp_clever,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permuted interp.")
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
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--width-multiplier", type=int, required=True)
  parser.add_argument("--load-epoch", type=int, required=True)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar10", "resnet20", "weight-matching"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.width_multiplier = args.width_multiplier
    config.seed = args.seed
    config.load_epoch = args.load_epoch

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=10,
                   width_multiplier=config.width_multiplier)
    stuff = make_stuff(model)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())

    filename = f"checkpoint{config.load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"cifar10-resnet-weights:{config.model_a}").get_path(
                filename).download()))
    model_b = load_model(
        Path(
            wandb_run.use_artifact(f"cifar10-resnet-weights:{config.model_b}").get_path(
                filename).download()))

    train_ds, test_ds = load_cifar10()

    permutation_spec = resnet20_permutation_spec()
    final_permutation = weight_matching(random.PRNGKey(config.seed), permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))

    # Save final_permutation as an Artifact
    artifact = wandb.Artifact("model_b_permutation",
                              type="permutation",
                              metadata={
                                  "dataset": "cifar10",
                                  "model": "resnet20",
                                  "analysis": "weight-matching"
                              })
    with artifact.new_file("permutation.pkl", mode="wb") as f:
      pickle.dump(final_permutation, f)
    wandb_run.log_artifact(artifact)

    lambdas = jnp.linspace(0, 1, num=25)
    train_loss_interp_naive = []
    test_loss_interp_naive = []
    train_acc_interp_naive = []
    test_acc_interp_naive = []
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a, model_b)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 1000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 1000)
      train_loss_interp_naive.append(train_loss)
      test_loss_interp_naive.append(test_loss)
      train_acc_interp_naive.append(train_acc)
      test_acc_interp_naive.append(test_acc)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b)))

    train_loss_interp_clever = []
    test_loss_interp_clever = []
    train_acc_interp_clever = []
    test_acc_interp_clever = []
    for lam in tqdm(lambdas):
      clever_p = lerp(lam, model_a, model_b_clever)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](clever_p, train_ds, 1000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](clever_p, test_ds, 1000)
      train_loss_interp_clever.append(train_loss)
      test_loss_interp_clever.append(test_loss)
      train_acc_interp_clever.append(train_acc)
      test_acc_interp_clever.append(test_acc)

    assert len(lambdas) == len(train_loss_interp_naive)
    assert len(lambdas) == len(test_loss_interp_naive)
    assert len(lambdas) == len(train_acc_interp_naive)
    assert len(lambdas) == len(test_acc_interp_naive)
    assert len(lambdas) == len(train_loss_interp_clever)
    assert len(lambdas) == len(test_loss_interp_clever)
    assert len(lambdas) == len(train_acc_interp_clever)
    assert len(lambdas) == len(test_acc_interp_clever)

    print("Plotting...")
    fig = plot_interp_loss(config.load_epoch, lambdas, train_loss_interp_naive,
                           test_loss_interp_naive, train_loss_interp_clever,
                           test_loss_interp_clever)
    plt.savefig(f"cifar10_resnet20_weight_matching_interp_loss_epoch{config.load_epoch}.png",
                dpi=300)
    wandb.log({"interp_loss_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig = plot_interp_acc(config.load_epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"cifar10_resnet20_weight_matching_interp_accuracy_epoch{config.load_epoch}.png",
                dpi=300)
    wandb.log({"interp_acc_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    wandb.log({
        "train_loss_interp_naive": train_loss_interp_naive,
        "test_loss_interp_naive": test_loss_interp_naive,
        "train_acc_interp_naive": train_acc_interp_naive,
        "test_acc_interp_naive": test_acc_interp_naive,
        "train_loss_interp_clever": train_loss_interp_clever,
        "test_loss_interp_clever": test_loss_interp_clever,
        "train_acc_interp_clever": train_acc_interp_clever,
        "test_acc_interp_clever": test_acc_interp_clever,
    })

    print({
        "train_loss_interp_naive": train_loss_interp_naive,
        "test_loss_interp_naive": test_loss_interp_naive,
        "train_acc_interp_naive": train_acc_interp_naive,
        "test_acc_interp_naive": test_acc_interp_naive,
        "train_loss_interp_clever": train_loss_interp_clever,
        "test_loss_interp_clever": test_loss_interp_clever,
        "train_acc_interp_clever": train_acc_interp_clever,
        "test_acc_interp_clever": test_acc_interp_clever,
    })

# if __name__ == "__main__":
#   main()
