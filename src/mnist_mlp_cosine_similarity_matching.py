import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
from flax import linen as nn
from flax.serialization import from_bytes
from jax import random, tree_map
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, init_train_state, load_datasets, make_stuff
from utils import (RngPooper, ec2_get_instance_type, flatten_params, kmatch, timeblock,
                   unflatten_params)

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def cosine_similarity(X, Y):
  # X: (m, d)
  # Y: (n, d)
  # return: (m, n)
  return (X @ Y.T) / jnp.linalg.norm(X, axis=-1).reshape((-1, 1)) / jnp.linalg.norm(Y, axis=-1)

def match_filters(paramsA, paramsB):
  """Permute the parameters of paramsB to match paramsA as closely as possible.
  Returns the permutation to apply to the weights of paramsB in order to align
  as best as possible with paramsA along with the permuted paramsB."""
  paf = flatten_params(paramsA)
  pbf = flatten_params(paramsB)

  perm = {}
  pbf_new = {**pbf}

  num_layers = max(int(kmatch("Dense_*/**", k).group(1)) for k in paf.keys())
  # range is [0, num_layers), so we're safe here since we don't want to be
  # reordering the output of the last layer.
  for layer in range(num_layers):
    # Maximize since we're dealing with similarities, not distances.
    # Note that it's critically important to use `pbf_new` here, not `pbf`!
    ri, ci = linear_sum_assignment(cosine_similarity(paf[f"Dense_{layer}/kernel"].T,
                                                     pbf_new[f"Dense_{layer}/kernel"].T),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()

    perm[f"Dense_{layer}"] = ci

    pbf_new[f"Dense_{layer}/kernel"] = pbf_new[f"Dense_{layer}/kernel"][:, ci]
    pbf_new[f"Dense_{layer}/bias"] = pbf_new[f"Dense_{layer}/bias"][ci]
    pbf_new[f"Dense_{layer+1}/kernel"] = pbf_new[f"Dense_{layer+1}/kernel"][ci, :]

  return perm, unflatten_params(pbf_new)

# def apply_permutation(perm, params):
#   pf = flatten_params(params)
#   num_layers = max(int(kmatch("Dense_*/**", k).group(1)) for k in pf.keys())
#   pf_new = {**pf}
#   for layer in range(num_layers):
#     p = perm[f"Dense_{layer} Dense_{layer+1}"]
#     pf_new[f"Dense_{layer}/kernel"] = pf_new[f"Dense_{layer}/kernel"][:, p]
#     pf_new[f"Dense_{layer}/bias"] = pf_new[f"Dense_{layer}/bias"][p]
#     pf_new[f"Dense_{layer+1}/kernel"] = pf_new[f"Dense_{layer+1}/kernel"][p, :]
#   return unflatten_params(pf_new)

def test_cosine_similarity():
  rp = RngPooper(random.PRNGKey(0))

  for _ in range(10):
    X = random.normal(rp.poop(), (3, 5))
    Y = random.normal(rp.poop(), (7, 5))
    assert jnp.allclose(1 - cosine_similarity(X, Y), cdist(X, Y, metric="cosine"))

def test_match_filters():
  rp = RngPooper(random.PRNGKey(0))

  class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

  model = Model()
  p1 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  p2 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  # print(tree_map(jnp.shape, flatten_params(p1)))

  _, new_p2 = match_filters(p1["params"], p2["params"])

  # Test that the model is the same after permutation.
  random_input = random.normal(rp.poop(), (128, 28 * 28))
  # print(jnp.max(jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))))
  assert ((jnp.abs(model.apply(p2, random_input) - model.apply({"params": new_p2}, random_input))) <
          1e-5).all()

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
      tags=["mnist", "mlp", "cosine-similarity-matching"],
      # See https://github.com/wandb/client/issues/3672.
      mode="online",
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.seed = args.seed
    config.epoch = 49

    model = MLPModel()

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(init_train_state(random.PRNGKey(0), -1, model), fh.read())

    artifact_a = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_a}").download())
    artifact_b = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_b}").download())
    model_a = load_model(artifact_a / f"checkpoint{config.epoch}")
    model_b = load_model(artifact_b / f"checkpoint{config.epoch}")

    stuff = make_stuff(model)
    train_ds, test_ds = load_datasets()

    final_permutation, model_b_clever = match_filters(model_a.params, model_b.params)

    # Save final_permutation as an Artifact
    artifact = wandb.Artifact("model_b_permutation",
                              type="permutation",
                              metadata={
                                  "dataset": "mnist",
                                  "model": "mlp"
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
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 10_000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 10_000)
      train_loss_interp_naive.append(train_loss)
      test_loss_interp_naive.append(test_loss)
      train_acc_interp_naive.append(train_acc)
      test_acc_interp_naive.append(test_acc)

    train_loss_interp_clever = []
    test_loss_interp_clever = []
    train_acc_interp_clever = []
    test_acc_interp_clever = []
    for lam in tqdm(lambdas):
      clever_p = tree_map(lambda a, b: (1 - lam) * a + lam * b, model_a.params, model_b_clever)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](clever_p, train_ds, 10_000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](clever_p, test_ds, 10_000)
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
    plt.savefig(f"mnist_mlp_interp_loss_epoch{config.epoch}.png", dpi=300)
    wandb.log({"interp_loss_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig = plot_interp_acc(config.epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_interp_accuracy_epoch{config.epoch}.png", dpi=300)
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
  with timeblock("Tests"):
    test_cosine_similarity()
    test_match_filters()

  main()
