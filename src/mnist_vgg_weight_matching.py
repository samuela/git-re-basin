"""
--model-a=v5 --model-b=v4
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.serialization import from_bytes
from jax import random, tree_map
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import wandb
from mnist_vgg16_run import (init_train_state, load_datasets, make_stuff, make_vgg_width_ablation)
from utils import (ec2_get_instance_type, flatten_params, rngmix, unflatten_params)

class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
  """We assume that one permutation cannot appear in two axes of the same weight array."""
  assert num_hidden_layers >= 1
  return permutation_spec_from_axes_to_perm({
      "Dense_0/kernel": (None, "P_0"),
      **{f"Dense_{i}/kernel": (f"P_{i-1}", f"P_{i}")
         for i in range(1, num_hidden_layers)},
      **{f"Dense_{i}/bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers-1}", None),
      f"Dense_{num_hidden_layers}/bias": (None, ),
  })

def vgg16_permutation_spec() -> PermutationSpec:
  return permutation_spec_from_axes_to_perm({
      "Conv_0/kernel": (None, None, None, "P_Conv_0"),
      **{f"Conv_{i}/kernel": (None, None, f"P_Conv_{i-1}", f"P_Conv_{i}")
         for i in range(1, 13)},
      **{f"Conv_{i}/bias": (f"P_Conv_{i}", )
         for i in range(13)},
      **{f"LayerNorm_{i}/scale": (f"P_Conv_{i}", )
         for i in range(13)},
      **{f"LayerNorm_{i}/bias": (f"P_Conv_{i}", )
         for i in range(13)},
      "Dense_0/kernel": ("P_Conv_12", "P_Dense_0"),
      "Dense_0/bias": ("P_Dense_0", ),
      "Dense_1/kernel": ("P_Dense_0", "P_Dense_1"),
      "Dense_1/bias": ("P_Dense_1", ),
      "Dense_2/kernel": ("P_Dense_1", None),
      "Dense_2/bias": (None, ),
  })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      w = jnp.take(w, perm[p], axis=axis)

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(rng, ps: PermutationSpec, params_a, params_b):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
  perm_names = list(perm.keys())

  for iteration in range(100):
    progress = False
    for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = jnp.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
        A += w_a @ w_b.T
        # A += w_a.size * (w_a @ w_b.T)

      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()

      oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
      newL = jnp.vdot(A, jnp.eye(n)[ci, :])
      print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = jnp.array(ci)

    if not progress:
      break

  return perm

def test_weight_matching():
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=1)
  rng = random.PRNGKey(123)
  num_hidden = 10
  shapes = {
      "Dense_0/kernel": (2, num_hidden),
      "Dense_0/bias": (num_hidden, ),
      "Dense_1/kernel": (num_hidden, 3),
      "Dense_1/bias": (3, )
  }
  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
  perm = weight_matching(rng, ps, params_a, params_b)
  print(perm)

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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["mnist", "vgg16", "weight-matching"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.seed = args.seed
    config.epoch = 24

    model = make_vgg_width_ablation(64)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            init_train_state(random.PRNGKey(0),
                             model,
                             learning_rate=-1,
                             num_epochs=100,
                             batch_size=100,
                             num_train_examples=50_000), fh.read())

    artifact_a = Path(wandb_run.use_artifact(f"mnist-vgg16-weights:{config.model_a}").download())
    artifact_b = Path(wandb_run.use_artifact(f"mnist-vgg16-weights:{config.model_b}").download())
    model_a = load_model(artifact_a / f"checkpoint{config.epoch}")
    model_b = load_model(artifact_b / f"checkpoint{config.epoch}")

    stuff = make_stuff(model)
    train_ds, test_ds = load_datasets()

    permutation_spec = vgg16_permutation_spec()
    final_permutation = weight_matching(random.PRNGKey(config.seed), permutation_spec,
                                        flatten_params(model_a.params),
                                        flatten_params(model_b.params))

    # Save final_permutation as an Artifact
    artifact = wandb.Artifact("model_b_permutation",
                              type="permutation",
                              metadata={
                                  "dataset": "mnist",
                                  "model": "vgg16"
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
      naive_p = tree_map(lambda a, b: (1 - lam) * a + lam * b, model_a.params, model_b.params)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 1000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 1000)
      train_loss_interp_naive.append(train_loss)
      test_loss_interp_naive.append(test_loss)
      train_acc_interp_naive.append(train_acc)
      test_acc_interp_naive.append(test_acc)

    # final_permutation = {k: jnp.argsort(v) for k, v in final_permutation.items()}
    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b.params)))

    train_loss_interp_clever = []
    test_loss_interp_clever = []
    train_acc_interp_clever = []
    test_acc_interp_clever = []
    for lam in tqdm(lambdas):
      clever_p = tree_map(lambda a, b: (1 - lam) * a + lam * b, model_a.params, model_b_clever)
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
    fig = plot_interp_loss(config.epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                           train_loss_interp_clever, test_loss_interp_clever)
    plt.savefig(f"mnist_vgg16_weight_matching_interp_loss_epoch{config.epoch}.png", dpi=300)
    wandb.log({"interp_loss_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig = plot_interp_acc(config.epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_vgg16_weight_matching_interp_accuracy_epoch{config.epoch}.png", dpi=300)
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

if __name__ == "__main__":
  main()
