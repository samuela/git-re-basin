"""STEv2 on MLP, MNIST. Params represent the B endpoint and projection is inner
product matching."""
import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import wandb
from flax.core import freeze
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad
from jax.lax import stop_gradient
from tqdm import tqdm

from mnist_mlp_train import MLPModel, load_datasets, make_stuff
from utils import (ec2_get_instance_type, flatten_params, lerp, rngmix, unflatten_params)
from weight_matching import (apply_permutation, mlp_permutation_spec, weight_matching)

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

# def main():
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["mnist", "mlp", "ste2"],
      # See https://github.com/wandb/client/issues/3672.
      mode="online",
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.seed = args.seed
    config.num_epochs = 250
    config.batch_size = 1000
    config.learning_rate = 1e-2
    # This is the epoch that we pull the model A/B params from.
    config.load_epoch = 99

    model = MLPModel()
    stuff = make_stuff(model)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 28, 28, 1)))["params"], fh.read())

    filename = f"checkpoint{config.load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_a}").get_path(
                filename).download()))
    model_b = load_model(
        Path(
            wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_b}").get_path(
                filename).download()))

    train_ds, test_ds = load_datasets()
    num_train_examples = train_ds["images_u8"].shape[0]
    num_test_examples = test_ds["images_u8"].shape[0]
    assert num_train_examples % config.batch_size == 0
    assert num_test_examples % config.batch_size == 0

    train_loss_a, train_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a, train_ds, 10_000)
    train_loss_b, train_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b, train_ds, 10_000)
    test_loss_a, test_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a, test_ds, 10_000)
    test_loss_b, test_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b, test_ds, 10_000)

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

    @jit
    def batch_eval(params, projected_params, model_a_params, images_u8, labels):
      # See https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient
      ste_params = tree_map(lambda x, px: stop_gradient(px) + (x - stop_gradient(x)), params,
                            projected_params)
      midpoint_params = lerp(0.5, ste_params, freeze({"params": model_a_params}))
      l, info = stuff["batch_eval"](midpoint_params["params"], images_u8, labels)

      # Makes life easier to know when we're winning. stop_gradient shouldn't be
      # necessary but I'm paranoid.
      l -= stop_gradient(baseline_train_loss)

      return l, {**info, "accuracy": info["num_correct"] / config.batch_size}

    @jit
    def step(train_state, projected_params, images_u8, labels):
      (l, metrics), g = value_and_grad(batch_eval, has_aux=True)(
          train_state.params,
          projected_params,
          model_a,
          images_u8,
          labels,
      )
      train_state = train_state.apply_gradients(grads=g)
      return train_state, {**metrics, "loss": l}

    rng = random.PRNGKey(config.seed)

    # TODO: test Adam here, but better to try on CIFAR-10 VGG first...
    tx = optax.sgd(learning_rate=config.learning_rate, momentum=0.9)
    # tx = optax.adam(learning_rate=config.learning_rate, momentum=0.9)

    # Init at the model A params
    init_params = freeze({"params": model_a})

    train_state = TrainState.create(apply_fn=None, params=init_params, tx=tx)

    permutation_spec = mlp_permutation_spec(3)

    # The last permutation, useful for warm-starting the next weight matching.
    perm = None

    # Best permutation found so far...
    best_perm = None
    best_perm_loss = 999

    for epoch in tqdm(range(config.num_epochs)):
      train_data_perm = random.permutation(rngmix(rng, f"epoch-{epoch}"),
                                           num_train_examples).reshape((-1, config.batch_size))
      for i in range(num_train_examples // config.batch_size):
        # This is maximizing inner product
        perm = weight_matching(rngmix(rng, f"epoch-{epoch}-{i}"),
                               permutation_spec,
                               flatten_params(train_state.params["params"]),
                               flatten_params(model_b),
                               max_iter=100,
                               init_perm=perm)

        projected_params = unflatten_params(
            apply_permutation(permutation_spec, perm, flatten_params(model_b)))

        train_state, metrics = step(train_state, freeze({"params": projected_params}),
                                    train_ds["images_u8"][train_data_perm[i]],
                                    train_ds["labels"][train_data_perm[i]])

        if metrics["loss"] < best_perm_loss:
          best_perm_loss = metrics["loss"]
          best_perm = perm

        wandb_run.log(metrics)

        if not jnp.isfinite(metrics["loss"]):
          raise ValueError(f"Loss is not finite: {metrics['loss']}")

    # Save final_permutation as an Artifact
    artifact = wandb.Artifact("model_b_permutation",
                              type="permutation",
                              metadata={
                                  "dataset": "mnist",
                                  "model": "mlp",
                                  "analysis": "ste2"
                              })
    with artifact.new_file("permutation.pkl", mode="wb") as f:
      pickle.dump(best_perm, f)
    wandb_run.log_artifact(artifact)

    ### plotting
    lambdas = jnp.linspace(0, 1, num=25)

    train_loss_interp_naive = []
    test_loss_interp_naive = []
    train_acc_interp_naive = []
    test_acc_interp_naive = []
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a, model_b)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 10_000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 10_000)
      train_loss_interp_naive.append(train_loss)
      test_loss_interp_naive.append(test_loss)
      train_acc_interp_naive.append(train_acc)
      test_acc_interp_naive.append(test_acc)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, best_perm, flatten_params(model_b)))

    train_loss_interp_clever = []
    test_loss_interp_clever = []
    train_acc_interp_clever = []
    test_acc_interp_clever = []
    for lam in tqdm(lambdas):
      clever_p = lerp(lam, model_a, model_b_clever)
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
    plt.savefig(f"mnist_mlp_ste2_interp_loss_epoch{config.load_epoch}.png", dpi=300)
    plt.savefig(f"mnist_mlp_ste2_interp_loss_epoch{config.load_epoch}.eps")
    wandb_run.log({"interp_loss_fig": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_interp_acc(config.load_epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_ste2_interp_accuracy_epoch{config.load_epoch}.png", dpi=300)
    plt.savefig(f"mnist_mlp_ste2_interp_accuracy_epoch{config.load_epoch}.eps")
    wandb_run.log({"interp_accuracy_fig": wandb.Image(fig)})
    plt.close(fig)

    wandb_run.log({
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
