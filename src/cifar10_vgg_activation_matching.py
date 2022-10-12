"""Associate units between the two models by comparing the correlations between
activations in intermediate layers.

TODO:
* use wandb.run
* get model a/b from argparse
* pull weights from artifacts
* log the plots
* save permutation to artifact
"""
import argparse
from glob import glob

import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange
from flax import linen as nn
from flax.core import freeze
from flax.serialization import from_bytes
from jax import jit, random, tree_map, vmap
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from cifar10_vgg_run import VGG16Wide, init_train_state, make_stuff
from datasets import load_cifar10
from online_stats import OnlineCovariance, OnlineMean
from utils import flatten_params, unflatten_params

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

def vgg16_permutify(permutation, params):
  """Permute the parameters of `params` based on `permutation`."""
  params_flat = flatten_params(params)

  # print(tree_map(jnp.shape, params_flat))

  # conv  kernel shape: (width, height, in_channel, out_channel)
  # dense kernel shape: (in, out)

  # VGG16: Conv0-Conv12 flatten Dense0-Dense2

  def conv_norm_conv(pf, l1, l2, perm):
    """Permute the output channels of Conv_{l1} and the input channels of
    Conv_{l2}."""
    return {
        **pf, f"params/Conv_{l1}/kernel": pf[f"params/Conv_{l1}/kernel"][:, :, :, perm],
        f"params/Conv_{l1}/bias": pf[f"params/Conv_{l1}/bias"][perm],
        f"params/LayerNorm_{l1}/scale": pf[f"params/LayerNorm_{l1}/scale"][perm],
        f"params/LayerNorm_{l1}/bias": pf[f"params/LayerNorm_{l1}/bias"][perm],
        f"params/Conv_{l2}/kernel": pf[f"params/Conv_{l2}/kernel"][:, :, perm, :]
    }

  def dense_dense(pf, l1, l2, perm):
    """Permute the output channels of Dense_{l1} and the input channels of Dense_{l2}."""
    return {
        **pf, f"params/Dense_{l1}/kernel": pf[f"params/Dense_{l1}/kernel"][:, perm],
        f"params/Dense_{l1}/bias": pf[f"params/Dense_{l1}/bias"][perm],
        f"params/Dense_{l2}/kernel": pf[f"params/Dense_{l2}/kernel"][perm, :]
    }

  def conv_norm_flatten_dense(pf, l1, l2, perm):
    """Permute the output channels of Conv_{l1} and the input channels of Dense_{l2}."""
    # Note that the flatten is kind of a no-op since the flatten is (batch, 1, 1, 512) -> (batch, 512)
    return {
        **pf, f"params/Conv_{l1}/kernel": pf[f"params/Conv_{l1}/kernel"][:, :, :, perm],
        f"params/Conv_{l1}/bias": pf[f"params/Conv_{l1}/bias"][perm],
        f"params/LayerNorm_{l1}/scale": pf[f"params/LayerNorm_{l1}/scale"][perm],
        f"params/LayerNorm_{l1}/bias": pf[f"params/LayerNorm_{l1}/bias"][perm],
        f"params/Dense_{l2}/kernel": pf[f"params/Dense_{l2}/kernel"][perm, :]
    }

  params_flat_new = {**params_flat}

  # Backbone conv layers
  for layer in range(12):
    params_flat_new = conv_norm_conv(params_flat_new, layer, layer + 1,
                                     permutation[f"Conv_{layer}"])

  # Conv_12 flatten Dense_0
  params_flat_new = conv_norm_flatten_dense(params_flat_new, 12, 0, permutation["Conv_12"])

  # (Dense_0, Dense_1) and (Dense_1, Dense_2)
  params_flat_new = dense_dense(params_flat_new, 0, 1, permutation["Dense_0"])
  params_flat_new = dense_dense(params_flat_new, 1, 2, permutation["Dense_1"])

  return unflatten_params(params_flat_new)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument("--model-a", type=str, required=True)
  # parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  epoch = 99
  model = VGG16Wide()

  def load_checkpoint(run, epoch):
    # See https://github.com/wandb/client/issues/3247.
    # f = wandb.restore(f"checkpoint_{epoch}", run)
    _, _, run_id = run.split("/")
    files = glob(f"./wandb/run-*-{run_id}/files/checkpoint_{epoch}")
    assert len(files) == 1
    with open(files[0], "rb") as f:
      contents = f.read()
      # Note that a lot of the hyperparams here are not relevant, we just need
      # to get the right PyTree shape for flax to use when deserializing
      # checkpoints.
      _, ret = from_bytes((0,
                           init_train_state(random.PRNGKey(0),
                                            model,
                                            learning_rate=0.1,
                                            num_epochs=100,
                                            batch_size=128,
                                            num_train_examples=50_000)), contents)
      return ret

  model_a = load_checkpoint("skainswo/git-re-basin/3467b9pk", epoch)
  model_b = load_checkpoint("skainswo/git-re-basin/s622wdyh", epoch)

  train_ds, test_ds = load_cifar10()

  # TODO use config.test once we switch to wandb
  if args.test:
    train_ds["images_u8"] = train_ds["images_u8"][:10_000]
    train_ds["labels"] = train_ds["labels"][:10_000]

  # `lax.scan` requires that all the batches have identical shape so we have to
  # skip the final batch if it is incomplete.
  num_train_examples = train_ds["images_u8"].shape[0]
  # assert num_train_examples == 50_000

  batch_size = 500
  assert num_train_examples % batch_size == 0
  stuff = make_stuff(model)
  # Permute the training data in case we want to use a subset.
  train_data_perm = random.permutation(random.PRNGKey(123), num_train_examples).reshape(
      (-1, batch_size))

  all_layers = [f"Conv_{i}" for i in range(13)] + ["Dense_0", "Dense_1"]

  def get_intermediates(params, images_u8):
    """Calculate intermediate activations for all layers in flax's format."""
    images_f32 = vmap(stuff["normalize_transform"])(None, images_u8)
    _, state = model.apply({"params": params},
                           images_f32,
                           capture_intermediates=lambda mdl, _: isinstance(mdl, nn.LayerNorm) or
                           isinstance(mdl, nn.Dense),
                           mutable=["intermediates"])
    return state["intermediates"]

  def normalize_activations(intermediates):
    """Simplify the activation dict format and flatten everything to be (batch_size, channels)."""

    def layernorm(i: int):
      k = f"LayerNorm_{i}"
      act = intermediates[k]["__call__"][0]
      act = rearrange(act, "batch w h c -> (batch w h) c")
      act = nn.relu(act)
      return act

    def dense(i: int):
      k = f"Dense_{i}"
      # The activations are (batch_size, num_units) so we don't need to reshape.
      act = intermediates[k]["__call__"][0]
      act = nn.relu(act)
      return act

    return {
        "Dense_0": dense(0),
        "Dense_1": dense(1),
        **{f"Conv_{i}": layernorm(i)
           for i in range(13)},
    }

  get_activations = jit(
      lambda params, images_u8: normalize_activations(get_intermediates(params, images_u8)))

  # Calculate mean activations
  def _calc_means():

    def one(params):
      means = {
          "Dense_0": OnlineMean.init(4096),
          "Dense_1": OnlineMean.init(4096),
          **{f"Conv_{i}": OnlineMean.init(512)
             for i in range(13)}
      }
      for i in tqdm(range(num_train_examples // batch_size)):
        images_u8 = train_ds["images_u8"][train_data_perm[i]]
        act = get_activations(params, images_u8)
        means = {layer: means[layer].update(act[layer]) for layer in all_layers}
      return means

    return one(model_a.params), one(model_b.params)

  a_means, b_means = _calc_means()

  # Calculate the Pearson correlation between activations of the two models on
  # each layer.
  def _calc_corr():
    stats = {
        layer: OnlineCovariance.init(a_means[layer].mean(), b_means[layer].mean())
        for layer in all_layers
    }
    for i in tqdm(range(num_train_examples // batch_size)):
      images_u8 = train_ds["images_u8"][train_data_perm[i]]
      a_act = get_activations(model_a.params, images_u8)
      b_act = get_activations(model_b.params, images_u8)
      stats = {layer: stats[layer].update(a_act[layer], b_act[layer]) for layer in all_layers}
    return stats

  cov_stats = _calc_corr()

  def _matching():

    def one(corr):
      ri, ci = linear_sum_assignment(corr, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()
      return ci

    return {layer: one(cov_stats[layer].pearson_correlation()) for layer in all_layers}

  model_b_perm = _matching()

  print("Plotting correlations and activation statistics...")
  for layer in all_layers:
    # Pearson correlation plots
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(layer)
    corr = cov_stats[layer].pearson_correlation()

    plt.subplot(1, 2, 1)
    plt.title("Before matching")
    plt.imshow(corr, origin="upper")
    plt.xlabel("Model B channels")
    plt.ylabel("Model A channels")

    plt.subplot(1, 2, 2)
    plt.title("After matching and permuting units")
    plt.imshow(corr[:, model_b_perm[layer]], origin="upper")
    plt.xlabel("Model B channels")
    plt.ylabel("Model A channels")
    # plt.colorbar()

    fig.tight_layout()
    plt.savefig(f"cifar10_vgg16_activation_correlations_{layer}.png", dpi=300)
    plt.close(fig)

    # Activation mean/stddev plots
    fig = plt.figure()
    plt.title(f"VGG16 on CIFAR-10 ({layer})")
    plt.scatter(a_means[layer].mean(),
                jnp.sqrt(cov_stats[layer].a_variance()),
                alpha=0.5,
                label="Model A channels")
    plt.scatter(b_means[layer].mean(),
                jnp.sqrt(cov_stats[layer].b_variance()),
                alpha=0.5,
                label="Model B channels")
    plt.xlabel("Channel activation mean")
    plt.ylabel("Channel activation std. dev.")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cifar10_vgg16_activation_scatter_{layer}.png", dpi=300)
    plt.close(fig)

  model_b_params_permuted = vgg16_permutify(model_b_perm, {"params": model_b.params})

  lambdas = jnp.linspace(0, 1, num=10)
  train_loss_interp_clever = []
  test_loss_interp_clever = []
  train_acc_interp_clever = []
  test_acc_interp_clever = []

  train_loss_interp_naive = []
  test_loss_interp_naive = []
  train_acc_interp_naive = []
  test_acc_interp_naive = []

  print("Evaluating model interpolations...")
  for lam in tqdm(lambdas):
    naive_p = freeze(tree_map(lambda a, b: (1 - lam) * a + lam * b, model_a.params, model_b.params))
    naive_train_loss, naive_train_acc = stuff["dataset_loss_and_accuracy"](naive_p, train_ds, 1000)
    naive_test_loss, naive_test_acc = stuff["dataset_loss_and_accuracy"](naive_p, test_ds, 1000)
    train_loss_interp_naive.append(naive_train_loss)
    test_loss_interp_naive.append(naive_test_loss)
    train_acc_interp_naive.append(naive_train_acc)
    test_acc_interp_naive.append(naive_test_acc)

    clever_p = freeze(
        tree_map(lambda a, b: (1 - lam) * a + lam * b, model_a.params,
                 model_b_params_permuted["params"]))
    clever_train_loss, clever_train_acc = stuff["dataset_loss_and_accuracy"](clever_p, train_ds,
                                                                             1000)
    clever_test_loss, clever_test_acc = stuff["dataset_loss_and_accuracy"](clever_p, test_ds, 1000)
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
  fig = plot_interp_loss(epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                         train_loss_interp_clever, test_loss_interp_clever)
  plt.savefig(f"cifar10_vgg16_interp_loss_epoch{epoch}_activation_matching.png", dpi=300)
  plt.savefig(f"cifar10_vgg16_interp_loss_epoch{epoch}_activation_matching.pdf")
  plt.close(fig)

  fig = plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                        train_acc_interp_clever, test_acc_interp_clever)
  plt.savefig(f"cifar10_vgg16_interp_accuracy_epoch{epoch}_activation_matching.png", dpi=300)
  plt.savefig(f"cifar10_vgg16_interp_accuracy_epoch{epoch}_activation_matching.pdf")
  plt.close(fig)
