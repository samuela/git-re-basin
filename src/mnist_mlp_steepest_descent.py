import argparse
import operator
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
from flax.serialization import from_bytes
from jax import grad, jit, random, tree_map
from jax.tree_util import tree_reduce
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, init_train_state, load_datasets, make_stuff
from utils import (ec2_get_instance_type, flatten_params, rngmix, timeblock, unflatten_params)

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

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

def apply_permutation(P, params):
  pf = flatten_params(params)
  return unflatten_params({
      "Dense_0/kernel": pf["Dense_0/kernel"][:, P["Dense_0"]],
      "Dense_0/bias": pf["Dense_0/bias"][P["Dense_0"]],
      "Dense_1/kernel": pf["Dense_1/kernel"][P["Dense_0"], :][:, P["Dense_1"]],
      "Dense_1/bias": pf["Dense_1/bias"][P["Dense_1"]],
      "Dense_2/kernel": pf["Dense_2/kernel"][P["Dense_1"], :][:, P["Dense_2"]],
      "Dense_2/bias": pf["Dense_2/bias"][P["Dense_2"]],
      "Dense_3/kernel": pf["Dense_3/kernel"][P["Dense_2"], :],
      "Dense_3/bias": pf["Dense_3/bias"],
  })

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["mnist", "mlp", "steepest-descent"],
      # See https://github.com/wandb/client/issues/3672.
      mode="online",
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.test = args.test
    config.seed = args.seed
    config.batch_size = 10_000
    # This is the epoch that we pull the model A/B params from.
    config.load_epoch = 49

    model = MLPModel()

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(init_train_state(random.PRNGKey(0), -1, model), fh.read())

    with timeblock("load models"):
      artifact_a = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_a}").download())
      artifact_b = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_b}").download())
      model_a = load_model(artifact_a / f"checkpoint{config.load_epoch}")
      model_b = load_model(artifact_b / f"checkpoint{config.load_epoch}")

      # For some reason flax rehydrates model paramenters as numpy arrays
      # instead of jax.numpy arrays.
      model_a = model_a.replace(params=tree_map(jnp.asarray, model_a.params))
      model_b = model_b.replace(params=tree_map(jnp.asarray, model_b.params))

    with timeblock("make_stuff"):
      stuff = make_stuff(model)
    with timeblock("load datasets"):
      train_ds, test_ds = load_datasets(smoke_test_mode=config.test)
    num_train_examples = train_ds["images_u8"].shape[0]
    num_test_examples = test_ds["images_u8"].shape[0]
    assert num_train_examples % config.batch_size == 0
    assert num_test_examples % config.batch_size == 0

    ### Calculate the gradient of the loss wrt to model_a.params.
    num_batches = num_train_examples // config.batch_size
    batch_ix = jnp.arange(num_train_examples).reshape((-1, config.batch_size))
    grady = jit(
        grad(lambda params, images_u8, labels: stuff["batch_eval"](params, images_u8, labels)[0]))
    grad_L_a = tree_map(jnp.zeros_like, model_a.params)
    with timeblock("get that gradient son"):
      for i in range(num_batches):
        p = batch_ix[i, :]
        images_u8 = train_ds["images_u8"][p, :, :, :]
        labels = train_ds["labels"][p]
        g = grady(model_a.params, images_u8, labels)
        grad_L_a = tree_map(operator.add, grad_L_a, g)

    # Don't forget to normalize so that we get the mean gradient!
    grad_L_a = tree_map(lambda x: config.batch_size * x / num_train_examples, grad_L_a)

    # <dL/dA, A>
    constant_term = tree_reduce(operator.add, tree_map(jnp.vdot, grad_L_a, model_a.params))

    def lsa(M1, N1, M2, N2, m1, n1, P0, P1, P2):
      M1 = M1[P0, :]
      M2 = M2[:, P2]
      A = (M1.T @ N1 + M2 @ N2.T + jnp.outer(m1, n1)).T
      garbage, newP1 = linear_sum_assignment(A)
      newP1 = jnp.array(newP1)
      assert (garbage == jnp.arange(len(garbage))).all()
      return (
          newP1,
          jnp.vdot(jnp.eye(len(P1))[P1, :], A),
          jnp.vdot(jnp.eye(len(newP1))[newP1, :], A),
      )

    # TODO: hardcoding the number of layers for now
    num_layers = 4
    Ms = [model_b.params[f"Dense_{i}"]["kernel"] for i in range(num_layers)]
    Ns = [grad_L_a[f"Dense_{i}"]["kernel"] for i in range(num_layers)]
    ms = [model_b.params[f"Dense_{i}"]["bias"] for i in range(num_layers)]
    ns = [grad_L_a[f"Dense_{i}"]["bias"] for i in range(num_layers)]

    Ps = [jnp.arange(Ms[i].shape[1]) for i in range(num_layers - 1)]
    Ps = [jnp.arange(Ms[0].shape[0])] + Ps + [jnp.arange(Ms[-1].shape[1])]

    def calc_slope():
      return -constant_term + tree_reduce(
          operator.add,
          tree_map(
              jnp.vdot,
              ([M1[P0, :][:, P1]
                for M1, P0, P1 in zip(Ms, Ps, Ps[1:])], [m1[P1] for m1, P1 in zip(ms, Ps[1:])]),
              (Ns, ns),
          )).item()

    slope = calc_slope()

    rng = random.PRNGKey(config.seed)
    for iteration in range(100):
      progress = False
      for l in random.permutation(rngmix(rng, iteration), num_layers - 1):
        M1, N1, M2, N2, m1, n1 = Ms[l], Ns[l], Ms[l + 1], Ns[l + 1], ms[l], ns[l]
        P0, P1, P2 = Ps[l], Ps[l + 1], Ps[l + 2]
        Ps[l + 1], oldL, newL = lsa(M1, N1, M2, N2, m1, n1, P0, P1, P2)
        progress = progress or newL < oldL - 1e-12
        slope += newL - oldL
        print("slope", slope)
      if not progress:
        break

    print(calc_slope(), " -- double check our running loss calculation")

    final_permutation = {"Dense_0": Ps[1], "Dense_1": Ps[2], "Dense_2": Ps[3]}

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

    ### plotting
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

    model_b_clever = apply_permutation(final_permutation, model_b.params)

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
    fig = plot_interp_loss(config.load_epoch, lambdas, train_loss_interp_naive,
                           test_loss_interp_naive, train_loss_interp_clever,
                           test_loss_interp_clever)
    plt.savefig(f"mnist_mlp_steepest_descent_interp_loss_epoch{config.load_epoch}.png", dpi=300)
    wandb_run.log({"interp_loss_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig = plot_interp_acc(config.load_epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_steepest_descent_interp_accuracy_epoch{config.load_epoch}.png", dpi=300)
    wandb_run.log({"interp_acc_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    wandb_run.log({}, commit=True)

if __name__ == "__main__":
  main()
