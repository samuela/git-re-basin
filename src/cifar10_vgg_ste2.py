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
from jax.tree_util import tree_reduce
from tqdm import tqdm

from cifar10_vgg_run import (init_train_state, make_stuff, make_vgg_width_ablation)
from datasets import load_cifar10
from utils import (ec2_get_instance_type, flatten_params, rngmix, unflatten_params)
from weight_matching import (apply_permutation, vgg16_permutation_spec, weight_matching)

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
  parser.add_argument("--width-multiplier", type=int, required=True)
  parser.add_argument("--learning-rate", type=float, required=True)
  parser.add_argument("--params-represent",
                      choices=["midpoint", "b_endpoint"],
                      default="b_endpoint")
  parser.add_argument("--projection-type", choices=["l2", "inner_product"], default="inner_product")
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar10", "vgg16", "straight-through-estimator-2"],
      # See https://github.com/wandb/client/issues/3672.
      mode="online",
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.seed = args.seed
    config.width_multiplier = args.width_multiplier
    config.num_epochs = 250
    config.batch_size = 500
    config.learning_rate = args.learning_rate
    # This is the epoch that we pull the model A/B params from.
    config.load_epoch = 99

    config.params_represent = args.params_represent
    config.projection_type = args.projection_type

    model = make_vgg_width_ablation(config.width_multiplier)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            init_train_state(random.PRNGKey(0),
                             model,
                             learning_rate=-1,
                             num_epochs=100,
                             batch_size=100,
                             num_train_examples=50_000), fh.read())

    artifact_a = Path(wandb_run.use_artifact(f"cifar10-vgg-weights:{config.model_a}").download())
    artifact_b = Path(wandb_run.use_artifact(f"cifar10-vgg-weights:{config.model_b}").download())
    model_a = load_model(artifact_a / f"checkpoint{config.load_epoch}")
    model_b = load_model(artifact_b / f"checkpoint{config.load_epoch}")

    stuff = make_stuff(model)
    train_ds, test_ds = load_cifar10()
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

    @jit
    def batch_eval(params, projected_params, model_a_params, images_u8, labels):
      # See https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient
      ste_params = tree_map(lambda x, px: stop_gradient(px) + (x - stop_gradient(x)), params,
                            projected_params)

      # "params <~> midpoint" else, "params <~> B endpoint"
      eval_params = ste_params if config.params_represent == "midpoint" else tree_map(
          lambda x, y: 0.5 * (x + y), ste_params, freeze({"params": model_a_params}))

      l, info = stuff["batch_eval"](eval_params["params"], images_u8, labels)

      # Makes life easier to know when we're winning. stop_gradient shouldn't be
      # necessary but I'm paranoid.
      l -= stop_gradient(baseline_train_loss)

      return l, {**info, "accuracy": info["num_correct"] / config.batch_size}

    @jit
    def step(train_state, projected_params, images_u8, labels):
      (l, metrics), g = value_and_grad(batch_eval, has_aux=True)(
          train_state.params,
          projected_params,
          model_a.params,
          images_u8,
          labels,
      )
      train_state = train_state.apply_gradients(grads=g)
      return train_state, {**metrics, "loss": l}

    rng = random.PRNGKey(config.seed)

    tx = optax.sgd(learning_rate=config.learning_rate, momentum=0.9)
    # tx = optax.radam(learning_rate=config.learning_rate)

    permutation_spec = vgg16_permutation_spec()

    # init_params = model.init(rngmix(rng, "init"), jnp.zeros((1, 32, 32, 3)))["params"]

    # Better init when projecting with L2?
    # init_perm = weight_matching(rngmix(rng, "weight_matching"), permutation_spec,
    #                             flatten_params(model_a.params), flatten_params(model_b.params))
    # init_params = unflatten_params(
    #     apply_permutation(permutation_spec, init_perm, flatten_params(model_b.params)))

    # init_params = tree_map(lambda x, y: 0.5 * (x + y), init_params, model_a.params)

    init_params = model_a.params

    # init_params = tree_map(lambda x, y: 0.5 * (x + y), model_a.params, model_b.params)

    train_state = TrainState.create(apply_fn=None, params=freeze({"params": init_params}), tx=tx)

    perm = None

    # Best permutation found so far...
    best_perm = None
    best_perm_loss = 999

    artifact = wandb.Artifact("model_b_permutation",
                              type="permutation",
                              metadata={
                                  "dataset": "cifar10",
                                  "model": "vgg16",
                                  "method": "ste2"
                              })

    for epoch in tqdm(range(config.num_epochs)):
      train_data_perm = random.permutation(rngmix(rng, f"epoch-{epoch}"),
                                           num_train_examples).reshape((-1, config.batch_size))
      for i in range(num_train_examples // config.batch_size):
        if config.projection_type == "inner_product":
          perm = weight_matching(rngmix(rng, f"epoch-{epoch}-{i}"),
                                 permutation_spec,
                                 flatten_params(train_state.params["params"]),
                                 flatten_params(model_b.params),
                                 max_iter=100,
                                 init_perm=perm)
        if config.projection_type == "l2":
          perm = weight_matching(
              rngmix(rng, f"epoch-{epoch}-{i}"),
              permutation_spec,
              flatten_params(
                  tree_map(lambda w, w_a: 2 * w - w_a, train_state.params["params"],
                           model_a.params)),
              flatten_params(model_b.params),
              max_iter=100,
              init_perm=perm,
          )

        # "params <~> midpoint", else "params <~> B endpoint"
        if config.params_represent == "midpoint":
          projected_params = tree_map(
              lambda x, y: 0.5 * (x + y), model_a.params,
              unflatten_params(
                  apply_permutation(permutation_spec, perm, flatten_params(model_b.params))))
        if config.params_represent == "b_endpoint":
          projected_params = unflatten_params(
              apply_permutation(permutation_spec, perm, flatten_params(model_b.params)))

        # Good ole' SGD, no STE
        # projected_params = train_state.params["params"]

        projection_l2 = jnp.sqrt(
            tree_reduce(
                lambda x, y: x + y,
                tree_map(lambda x, y: jnp.sum(jnp.square(x - y)),
                         flatten_params(train_state.params["params"]),
                         flatten_params(projected_params))))

        train_state, metrics = step(train_state, freeze({"params": projected_params}),
                                    train_ds["images_u8"][train_data_perm[i]],
                                    train_ds["labels"][train_data_perm[i]])
        wandb_run.log({**metrics, "projection_l2": projection_l2})

        if metrics["loss"] < best_perm_loss:
          best_perm_loss = metrics["loss"]
          best_perm = perm

        if not jnp.isfinite(metrics["loss"]):
          raise ValueError(f"Loss is not finite: {metrics['loss']}")

        # Save permutation to artifact
        with artifact.new_file(f"permutation_e{epoch}i{i}.pkl", mode="wb") as f:
          pickle.dump(perm, f)

    wandb_run.log_artifact(artifact)

# if __name__ == "__main__":
#   main()
