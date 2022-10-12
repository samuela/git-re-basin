import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
from einops import reduce
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad
from jax.lax import stop_gradient
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, init_train_state, load_datasets, make_stuff
from utils import (RngPooper, ec2_get_instance_type, flatten_params, rngmix, unflatten_params)

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

def sinkhorn_knopp_projection(A, num_iter=10):
  # We clip to be positive before calling this function.
  A = jnp.maximum(A, 0)
  for _ in range(num_iter):
    # normalize rows
    A = A / reduce(A, "i j -> i 1", "sum")
    # normalize columns
    A = A / reduce(A, "i j -> 1 j", "sum")
  return A

def permute_params_init(rng):
  # Dense_0 Dense_1 Dense_2 Dense_3
  rp = RngPooper(rng)
  return {
      "Dense_0": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
      "Dense_1": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
      "Dense_2": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
  }

def permute_params_apply(permute_params, hardened_permute_params, model_params):

  # See https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient
  def _P(name):
    zero = permute_params[name] - stop_gradient(permute_params[name])
    return stop_gradient(hardened_permute_params[name]) + zero

  invmul = lambda A, B: A.T @ B
  # invmul = jnp.linalg.solve

  P = {"Dense_0": _P("Dense_0"), "Dense_1": _P("Dense_1"), "Dense_2": _P("Dense_2")}

  m = flatten_params(model_params)
  return unflatten_params({
      # Dense_0 has a fixed input.
      "Dense_0/kernel": m["Dense_0/kernel"] @ P["Dense_0"],
      "Dense_0/bias": m["Dense_0/bias"].T @ P["Dense_0"],
      "Dense_1/kernel": invmul(P["Dense_0"], m["Dense_1/kernel"] @ P["Dense_1"]),
      "Dense_1/bias": m["Dense_1/bias"].T @ P["Dense_1"],
      "Dense_2/kernel": invmul(P["Dense_1"], m["Dense_2/kernel"] @ P["Dense_2"]),
      "Dense_2/bias": m["Dense_2/bias"].T @ P["Dense_2"],
      # The output of Dense_3 has a fixed order so we don't need to the bias.
      "Dense_3/kernel": invmul(P["Dense_2"], m["Dense_3/kernel"]),
      "Dense_3/bias": m["Dense_3/bias"],
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
      tags=["mnist", "mlp", "straight-through-estimator"],
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
    config.num_epochs = 10
    config.batch_size = 1000
    config.learning_rate = 1e-2
    # This is the epoch that we pull the model A/B params from.
    config.load_epoch = 49

    model = MLPModel()

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(init_train_state(random.PRNGKey(0), -1, model), fh.read())

    artifact_a = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_a}").download())
    artifact_b = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_b}").download())
    model_a = load_model(artifact_a / f"checkpoint{config.load_epoch}")
    model_b = load_model(artifact_b / f"checkpoint{config.load_epoch}")

    stuff = make_stuff(model)
    train_ds, test_ds = load_datasets(smoke_test_mode=config.test)
    num_train_examples = train_ds["images_u8"].shape[0]
    num_test_examples = test_ds["images_u8"].shape[0]
    assert num_train_examples % config.batch_size == 0
    assert num_test_examples % config.batch_size == 0

    train_loss_a, train_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a.params, train_ds,
                                                                        10_000)
    train_loss_b, train_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b.params, train_ds,
                                                                        10_000)
    test_loss_a, test_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a.params, test_ds,
                                                                      10_000)
    test_loss_b, test_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b.params, test_ds,
                                                                      10_000)

    print({
        "train_loss_a": train_loss_a,
        "train_accuracy_a": train_accuracy_a,
        "train_loss_b": train_loss_b,
        "train_accuracy_b": train_accuracy_b,
        "test_loss_a": test_loss_a,
        "test_accuracy_a": test_accuracy_a,
        "test_loss_b": test_loss_b,
        "test_accuracy_b": test_accuracy_b,
    })

    baseline_train_loss = 0.5 * (train_loss_a + train_loss_b)

    def lsa(A):
      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()
      return ci

    def permutation_matrix(ixs):
      """Convert a permutation array, eg. [2, 3, 0, 1], to a permutation matrix."""
      # This is confusing, but indexing the columns onto the rows is actually the correct thing to do
      return jnp.eye(len(ixs), dtype=jnp.bool_)[ixs, :]

    @jit
    def batch_eval(permute_params, hardened_permute_params, images_u8, labels):
      model_b_permuted_params = permute_params_apply(permute_params, hardened_permute_params,
                                                     model_b.params)
      interp_params = tree_map(lambda a, b: 0.5 * (a + b), model_a.params, model_b_permuted_params)
      l, num_correct = stuff["batch_eval"](interp_params, images_u8, labels)

      # Makes life easier to know when we're winning. stop_gradient shouldn't be
      # necessary but I'm paranoid.
      l -= stop_gradient(baseline_train_loss)

      return l, {"num_correct": num_correct, "accuracy": num_correct / config.batch_size}

    @jit
    def step(train_state, hardened_permute_params, images_u8, labels):
      (l, metrics), g = value_and_grad(batch_eval,
                                       has_aux=True)(train_state.params, hardened_permute_params,
                                                     images_u8, labels)
      train_state = train_state.apply_gradients(grads=g)

      # Project onto Birkhoff polytope.
      train_state = train_state.replace(
          params=tree_map(sinkhorn_knopp_projection, train_state.params))

      return train_state, {**metrics, "loss": l}

    rng = random.PRNGKey(config.seed)

    tx = optax.sgd(learning_rate=config.learning_rate, momentum=0.9)
    train_state = TrainState.create(apply_fn=None,
                                    params=permute_params_init(rngmix(rng, "init")),
                                    tx=tx)

    for epoch in tqdm(range(config.num_epochs)):
      train_data_perm = random.permutation(rngmix(rng, f"epoch-{epoch}"),
                                           num_train_examples).reshape((-1, config.batch_size))
      for i in range(num_train_examples // config.batch_size):
        hardened_pp = {k: permutation_matrix(lsa(v)) for k, v in train_state.params.items()}
        train_state, metrics = step(train_state, hardened_pp,
                                    train_ds["images_u8"][train_data_perm[i]],
                                    train_ds["labels"][train_data_perm[i]])
        wandb_run.log(metrics)

        if not jnp.isfinite(metrics["loss"]):
          raise ValueError(f"Loss is not finite: {metrics['loss']}")

    final_permutation = {k: jnp.argsort(lsa(v)) for k, v in train_state.params.items()}

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
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 10_000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 10_000)
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
    fig = plot_interp_loss(config.load_epoch, lambdas, train_loss_interp_naive,
                           test_loss_interp_naive, train_loss_interp_clever,
                           test_loss_interp_clever)
    plt.savefig(f"mnist_mlp_ste_interp_loss_epoch{config.load_epoch}.png", dpi=300)
    wandb_run.log({"interp_loss_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig = plot_interp_acc(config.load_epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_ste_interp_accuracy_epoch{config.load_epoch}.png", dpi=300)
    wandb_run.log({"interp_acc_fig": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    wandb_run.log({}, commit=True)

if __name__ == "__main__":
  main()
