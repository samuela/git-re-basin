from glob import glob

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
from flax.core import freeze
from flax.serialization import from_bytes
from jax import random, tree_map
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from cifar10_vgg_run import TestVGG, VGG16Wide, init_train_state, make_stuff
from datasets import load_cifar10
from utils import RngPooper, flatten_params, timeblock, unflatten_params

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def cosine_similarity(X, Y):
  # X: (m, d)
  # Y: (n, d)
  # return: (m, n)
  return (X @ Y.T) / jnp.linalg.norm(X, axis=-1).reshape((-1, 1)) / jnp.linalg.norm(Y, axis=-1)

def permutify(paramsA, paramsB):
  """Permute the parameters of paramsB to match paramsA as closely as possible.
  Returns the permuted version of paramsB. Only works on sequences of Dense
  layers for now."""
  paf = flatten_params(paramsA)
  pbf = flatten_params(paramsB)

  # conv  kernel shape: (width, height, in_channel, out_channel)
  # dense kernel shape: (in, out)

  # VGG16: Conv0-Conv12 flatten Dense0-Dense2

  def conv_gn_conv(paf, pbf, l1, l2):
    """Permute the output channels of Conv_{l1} and the input channels of Conv_{l2}."""
    ka = paf[f"params/Conv_{l1}/kernel"]
    kb = pbf[f"params/Conv_{l1}/kernel"]
    assert ka.shape == kb.shape
    ri, ci = linear_sum_assignment(cosine_similarity(
        jnp.reshape(jnp.moveaxis(ka, -1, 0), (ka.shape[-1], -1)),
        jnp.reshape(jnp.moveaxis(kb, -1, 0), (kb.shape[-1], -1))),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return {
        **pbf, f"params/Conv_{l1}/kernel": pbf[f"params/Conv_{l1}/kernel"][:, :, :, ci],
        f"params/Conv_{l1}/bias": pbf[f"params/Conv_{l1}/bias"][ci],
        f"params/LayerNorm_{l1}/scale": pbf[f"params/LayerNorm_{l1}/scale"][ci],
        f"params/LayerNorm_{l1}/bias": pbf[f"params/LayerNorm_{l1}/bias"][ci],
        f"params/Conv_{l2}/kernel": pbf[f"params/Conv_{l2}/kernel"][:, :, ci, :]
    }

  def dense_dense(paf, pbf, l1, l2):
    """Permute the output channels of Dense_{l1} and the input channels of Dense_{l2}."""
    ka = paf[f"params/Dense_{l1}/kernel"]
    kb = pbf[f"params/Dense_{l1}/kernel"]
    assert ka.shape == kb.shape
    ri, ci = linear_sum_assignment(cosine_similarity(ka.T, kb.T), maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return {
        **pbf, f"params/Dense_{l1}/kernel": pbf[f"params/Dense_{l1}/kernel"][:, ci],
        f"params/Dense_{l1}/bias": pbf[f"params/Dense_{l1}/bias"][ci],
        f"params/Dense_{l2}/kernel": pbf[f"params/Dense_{l2}/kernel"][ci, :]
    }

  def conv_gn_flatten_dense(paf, pbf, l1, l2):
    # Note that this is much simpler than the general case since we also know
    # that the output of Conv_{l1} has shape (_, 1, 1, 512) when inputs are
    # (_, 32, 32, 3) as is the case with CIFAR-10. And Dense_{l2} has shape
    # (512, _).
    ka = paf[f"params/Conv_{l1}/kernel"]
    kb = pbf[f"params/Conv_{l1}/kernel"]
    assert ka.shape == kb.shape
    ri, ci = linear_sum_assignment(cosine_similarity(
        jnp.reshape(jnp.moveaxis(ka, -1, 0), (ka.shape[-1], -1)),
        jnp.reshape(jnp.moveaxis(kb, -1, 0), (kb.shape[-1], -1))),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return {
        **pbf, f"params/Conv_{l1}/kernel": pbf[f"params/Conv_{l1}/kernel"][:, :, :, ci],
        f"params/Conv_{l1}/bias": pbf[f"params/Conv_{l1}/bias"][ci],
        f"params/LayerNorm_{l1}/scale": pbf[f"params/LayerNorm_{l1}/scale"][ci],
        f"params/LayerNorm_{l1}/bias": pbf[f"params/LayerNorm_{l1}/bias"][ci],
        f"params/Dense_{l2}/kernel": pbf[f"params/Dense_{l2}/kernel"][ci, :]
    }

  pbf_new = {**pbf}

  # Backbone conv layers
  for layer in range(12):
    pbf_new = conv_gn_conv(paf, pbf_new, layer, layer + 1)

  # Conv_12 flatten Dense_0
  pbf_new = conv_gn_flatten_dense(paf, pbf_new, 12, 0)

  # (Dense_0, Dense_1) and (Dense_1, Dense_2)
  pbf_new = dense_dense(paf, pbf_new, 0, 1)
  pbf_new = dense_dense(paf, pbf_new, 1, 2)

  return unflatten_params(pbf_new)

def test_cosine_similarity():
  rp = RngPooper(random.PRNGKey(0))

  for _ in range(10):
    X = random.normal(rp.poop(), (3, 5))
    Y = random.normal(rp.poop(), (7, 5))
    assert jnp.allclose(1 - cosine_similarity(X, Y), cdist(X, Y, metric="cosine"))

def test_permutify():
  rp = RngPooper(random.PRNGKey(0))

  model = TestVGG()
  # model = VGG16()
  p1 = model.init(rp.poop(), jnp.zeros((1, 32, 32, 3)))
  p2 = model.init(rp.poop(), jnp.zeros((1, 32, 32, 3)))
  # print(tree_map(jnp.shape, flatten_params(p1)))

  new_p2 = permutify(p1, p2)

  # Test that the model is the same after permutation.
  random_input = random.normal(rp.poop(), (128, 32, 32, 3))
  # print(jnp.max(jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))))
  assert ((jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))) < 5e-5).all()

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
  with timeblock("Tests"):
    test_cosine_similarity()
    test_permutify()

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
                                            num_train_examples=50000)), contents)
      return ret

  model_a = load_checkpoint("skainswo/git-re-basin/3467b9pk", epoch)
  model_b = load_checkpoint("skainswo/git-re-basin/s622wdyh", epoch)

  lambdas = jnp.linspace(0, 1, num=10)
  train_loss_interp_clever = []
  test_loss_interp_clever = []
  train_acc_interp_clever = []
  test_acc_interp_clever = []

  train_loss_interp_naive = []
  test_loss_interp_naive = []
  train_acc_interp_naive = []
  test_acc_interp_naive = []

  train_ds, test_ds = load_cifar10()
  stuff = make_stuff(model, train_ds, train_batch_size=128)
  b2 = permutify({"params": model_a.params}, {"params": model_b.params})
  for lam in tqdm(lambdas):
    naive_p = freeze(tree_map(lambda a, b: lam * a + (1 - lam) * b, model_a.params, model_b.params))
    naive_train_loss, naive_train_acc = stuff.dataset_loss_and_accuracy(naive_p, train_ds, 1000)
    naive_test_loss, naive_test_acc = stuff.dataset_loss_and_accuracy(naive_p, test_ds, 1000)
    train_loss_interp_naive.append(naive_train_loss)
    test_loss_interp_naive.append(naive_test_loss)
    train_acc_interp_naive.append(naive_train_acc)
    test_acc_interp_naive.append(naive_test_acc)

    clever_p = freeze(tree_map(lambda a, b: lam * a + (1 - lam) * b, model_a.params, b2["params"]))
    clever_train_loss, clever_train_acc = stuff.dataset_loss_and_accuracy(clever_p, train_ds, 1000)
    clever_test_loss, clever_test_acc = stuff.dataset_loss_and_accuracy(clever_p, test_ds, 1000)
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

  print("Plotting...")
  fig = plot_interp_loss(epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                         train_loss_interp_clever, test_loss_interp_clever)
  plt.savefig(f"cifar10_vgg16_interp_loss_epoch{epoch}_filter_matching.png", dpi=300)
  plt.savefig(f"cifar10_vgg16_interp_loss_epoch{epoch}_filter_matching.pdf")
  plt.close(fig)

  fig = plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                        train_acc_interp_clever, test_acc_interp_clever)
  plt.savefig(f"cifar10_vgg16_interp_accuracy_epoch{epoch}_filter_matching.png", dpi=300)
  plt.savefig(f"cifar10_vgg16_interp_accuracy_epoch{epoch}_filter_matching.pdf")
  plt.close(fig)
