"""Associate units between the two models by comparing the correlations between
activations in intermediate layers."""
import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax import linen as nn
from flax.core import freeze
from flax.serialization import from_bytes
from jax import jit, random, vmap
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import matplotlib_style as _
from cifar10_mlp_train import MLPModel, make_stuff
from datasets import load_cifar10
from online_stats import OnlineCovariance, OnlineMean
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import apply_permutation, mlp_permutation_spec

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
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar10", "mlp", "activation-matching"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.load_epoch = 99

    model = MLPModel()
    stuff = make_stuff(model)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())

    filename = f"checkpoint{config.load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"cifar10-mlp-weights:{config.model_a}").get_path(
                filename).download()))
    model_b = load_model(
        Path(
            wandb_run.use_artifact(f"cifar10-mlp-weights:{config.model_b}").get_path(
                filename).download()))

    train_ds, test_ds = load_cifar10()

    num_train_examples = train_ds["images_u8"].shape[0]
    assert num_train_examples == 50_000

    batch_size = 500
    assert num_train_examples % batch_size == 0

    num_mlp_layers = 3
    all_layers = [f"Dense_{i}" for i in range(num_mlp_layers)]

    def get_intermediates(params, images_u8):
      """Calculate intermediate activations for all layers in flax's format."""
      images_f32 = vmap(stuff["normalize_transform"])(None, images_u8)
      _, state = model.apply({"params": params},
                             images_f32,
                             capture_intermediates=lambda mdl, _: isinstance(mdl, nn.Dense),
                             mutable=["intermediates"])
      return state["intermediates"]

    def normalize_activations(intermediates):
      """Simplify the activation dict format and flatten everything to be (batch_size, channels)."""

      def dense(i: int):
        k = f"Dense_{i}"
        # The activations are (batch_size, num_units) so we don't need to reshape.
        act = intermediates[k]["__call__"][0]
        act = nn.relu(act)
        return act

      return {f"Dense_{i}": dense(i) for i in range(num_mlp_layers)}

    get_activations = jit(
        lambda params, images_u8: normalize_activations(get_intermediates(params, images_u8)))

    # Permute the training data in case we want to use a subset.
    train_data_perm = random.permutation(random.PRNGKey(123), num_train_examples).reshape(
        (-1, batch_size))

    # Calculate mean activations
    def _calc_means():

      def one(params):
        means = {f"Dense_{i}": OnlineMean.init(512) for i in range(num_mlp_layers)}
        for i in range(num_train_examples // batch_size):
          images_u8 = train_ds["images_u8"][train_data_perm[i]]
          act = get_activations(params, images_u8)
          means = {layer: means[layer].update(act[layer]) for layer in all_layers}
        return means

      return one(model_a), one(model_b)

    a_means, b_means = _calc_means()

    # Calculate the covariance stats between the two
    def _calc_stats():
      stats = {
          layer: OnlineCovariance.init(a_means[layer].mean(), b_means[layer].mean())
          for layer in all_layers
      }
      for i in range(num_train_examples // batch_size):
        images_u8 = train_ds["images_u8"][train_data_perm[i]]
        a_act = get_activations(model_a, images_u8)
        b_act = get_activations(model_b, images_u8)
        stats = {layer: stats[layer].update(a_act[layer], b_act[layer]) for layer in all_layers}
      return stats

    layer_stats = _calc_stats()

    def _matching():

      def one(corr):
        ri, ci = linear_sum_assignment(corr, maximize=True)
        assert (ri == jnp.arange(len(ri))).all()
        return ci

      # match based on pearson correlations
      # return {layer: one(layer_stats[layer].pearson_correlation()) for layer in all_layers}

      # match based on E[A.T @ B]
      return {layer: one(layer_stats[layer].E_ab()) for layer in all_layers}

    model_b_perm = _matching()

    artifact = wandb.Artifact("cifar10_mlp_activation_matching",
                              type="permutation",
                              metadata={
                                  "dataset": "cifar10",
                                  "model": "mlp",
                                  "analysis": "activation-matching"
                              })
    with artifact.new_file("permutation.pkl", mode="wb") as f:
      pickle.dump({f"P_{i}": model_b_perm[f"Dense_{i}"] for i in range(num_mlp_layers)}, f)
    wandb_run.log_artifact(artifact)

    print("Plotting E_AB plots...")
    for layer in all_layers:
      fig = plt.figure(figsize=(8, 4))
      fig.suptitle(f"{layer} $\\mathbb{{E}}[A B]$")
      E_ab = layer_stats[layer].E_ab()

      plt.subplot(1, 2, 1)
      plt.title("Before")
      plt.imshow(E_ab, origin="upper")
      plt.xlabel("Model B channels")
      plt.ylabel("Model A channels")

      plt.subplot(1, 2, 2)
      plt.title("After")
      plt.imshow(E_ab[:, model_b_perm[layer]], origin="upper")
      plt.xlabel("Model B channels")
      plt.ylabel("Model A channels")
      # plt.colorbar()

      fig.tight_layout()
      plt.savefig(f"cifar10_mlp_activation_inner_products_{layer}.png", dpi=300)
      plt.savefig(f"cifar10_mlp_activation_inner_products_{layer}.pdf")
      plt.savefig(f"cifar10_mlp_activation_inner_products_{layer}.eps")
      wandb_run.log({f"activation_inner_products/{layer}": wandb.Image(fig)})
      plt.close(fig)

    print("Plotting Pearson correlation plots...")
    for layer in all_layers:
      fig = plt.figure(figsize=(8, 4))
      fig.suptitle(f"{layer} $Corr[A, B]$")
      corr = layer_stats[layer].pearson_correlation()

      plt.subplot(1, 2, 1)
      plt.title("Before")
      plt.imshow(corr, origin="upper")
      plt.xlabel("Model B features")
      plt.ylabel("Model A features")

      plt.subplot(1, 2, 2)
      plt.title("After")
      plt.imshow(corr[:, model_b_perm[layer]], origin="upper")
      plt.xlabel("Model B features")
      plt.ylabel("Model A features")
      # plt.colorbar()

      fig.tight_layout()
      plt.savefig(f"cifar10_mlp_activation_correlations_{layer}.png", dpi=300)
      plt.savefig(f"cifar10_mlp_activation_correlations_{layer}.pdf")
      plt.savefig(f"cifar10_mlp_activation_correlations_{layer}.eps")
      wandb_run.log({f"activation_correlations/{layer}": wandb.Image(fig)})
      plt.close(fig)

    print("Activation mean/stddev plots...")
    for layer in all_layers:
      # Activation mean/stddev plots
      fig = plt.figure()
      plt.title(f"MLP on CIFAR10 ({layer})")
      plt.scatter(a_means[layer].mean(),
                  jnp.sqrt(layer_stats[layer].a_variance()),
                  alpha=0.5,
                  label="Model A features")
      plt.scatter(b_means[layer].mean(),
                  jnp.sqrt(layer_stats[layer].b_variance()),
                  alpha=0.5,
                  label="Model B features")
      plt.xlabel("Feature activation mean")
      plt.ylabel("Feature activation std. dev.")
      plt.legend()
      plt.tight_layout()
      plt.savefig(f"cifar10_mlp_activation_scatter_{layer}.png", dpi=300)
      plt.savefig(f"cifar10_mlp_activation_scatter_{layer}.pdf")
      plt.savefig(f"cifar10_mlp_activation_scatter_{layer}.eps")
      wandb_run.log({f"activation_scatter/{layer}": wandb.Image(fig)})
      plt.close(fig)

    permutation_spec = mlp_permutation_spec(num_mlp_layers)
    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec,
                          {f"P_{i}": model_b_perm[f"Dense_{i}"]
                           for i in range(num_mlp_layers)}, flatten_params(model_b)))

    lambdas = jnp.linspace(0, 1, num=25)

    print("Naive interp...")
    train_loss_interp_naive = []
    test_loss_interp_naive = []
    train_acc_interp_naive = []
    test_acc_interp_naive = []
    for lam in tqdm(lambdas):
      naive_p = freeze(lerp(lam, model_a, model_b))
      naive_train_loss, naive_train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds,
                                                                             1000)
      naive_test_loss, naive_test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 1000)
      train_loss_interp_naive.append(naive_train_loss)
      test_loss_interp_naive.append(naive_test_loss)
      train_acc_interp_naive.append(naive_train_acc)
      test_acc_interp_naive.append(naive_test_acc)

    print("Activation matching interp...")
    train_loss_interp_clever = []
    test_loss_interp_clever = []
    train_acc_interp_clever = []
    test_acc_interp_clever = []
    for lam in tqdm(lambdas):
      clever_p = freeze(lerp(lam, model_a, model_b_clever))
      clever_train_loss, clever_train_acc = stuff["dataset_loss_and_accuracy"](clever_p, train_ds,
                                                                               1000)
      clever_test_loss, clever_test_acc = stuff["dataset_loss_and_accuracy"](clever_p, test_ds,
                                                                             1000)
      train_loss_interp_clever.append(clever_train_loss)
      test_loss_interp_clever.append(clever_test_loss)
      train_acc_interp_clever.append(clever_train_acc)
      test_acc_interp_clever.append(clever_test_acc)

    assert len(lambdas) == len(train_loss_interp_naive)
    assert len(lambdas) == len(test_loss_interp_naive)
    assert len(lambdas) == len(train_acc_interp_naive)
    assert len(lambdas) == len(test_acc_interp_naive)
    assert len(lambdas) == len(train_loss_interp_clever)
    assert len(lambdas) == len(test_loss_interp_clever)
    assert len(lambdas) == len(train_acc_interp_clever)
    assert len(lambdas) == len(test_acc_interp_clever)

    print("Plotting interpolation results...")
    fig = plot_interp_loss(config.load_epoch, lambdas, train_loss_interp_naive,
                           test_loss_interp_naive, train_loss_interp_clever,
                           test_loss_interp_clever)
    plt.savefig(f"cifar10_mlp_interp_loss_epoch{config.load_epoch}_activation_matching.png",
                dpi=300)
    plt.savefig(f"cifar10_mlp_interp_loss_epoch{config.load_epoch}_activation_matching.pdf")
    plt.savefig(f"cifar10_mlp_interp_loss_epoch{config.load_epoch}_activation_matching.eps")
    wandb_run.log({"loss_lerp": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_interp_acc(config.load_epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"cifar10_mlp_interp_accuracy_epoch{config.load_epoch}_activation_matching.png",
                dpi=300)
    plt.savefig(f"cifar10_mlp_interp_accuracy_epoch{config.load_epoch}_activation_matching.pdf")
    plt.savefig(f"cifar10_mlp_interp_accuracy_epoch{config.load_epoch}_activation_matching.eps")
    wandb_run.log({"accuracy_lerp": wandb.Image(fig)})
    plt.close(fig)

    wandb_run.log({
        "train_loss_interp_clever": train_loss_interp_clever,
        "test_loss_interp_clever": test_loss_interp_clever,
        "train_acc_interp_clever": train_acc_interp_clever,
        "test_acc_interp_clever": test_acc_interp_clever,
        "train_loss_interp_naive": train_loss_interp_naive,
        "test_loss_interp_naive": test_loss_interp_naive,
        "train_acc_interp_naive": train_acc_interp_naive,
        "test_acc_interp_naive": test_acc_interp_naive,
    })

    print({
        "train_loss_interp_clever": train_loss_interp_clever,
        "test_loss_interp_clever": test_loss_interp_clever,
        "train_acc_interp_clever": train_acc_interp_clever,
        "test_acc_interp_clever": test_acc_interp_clever,
        "train_loss_interp_naive": train_loss_interp_naive,
        "test_loss_interp_naive": test_loss_interp_naive,
        "train_acc_interp_naive": train_acc_interp_naive,
        "test_acc_interp_naive": test_acc_interp_naive,
    })
