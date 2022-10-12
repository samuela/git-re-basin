"""No longer relevant..."""
from glob import glob

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.core import freeze
from flax.serialization import from_bytes
from jax import random, tree_map
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from mnist_convnet_run import (ConvNetModel, TestModel, get_datasets, init_train_state, make_stuff)
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

  # {in:fixed Conv_0 Conv_1 any:mean:fixed Dense_0 Dense_1 fixed:out}

  def conv_conv(paf, pbf, l1, l2):
    """Permute the output channels of Conv_{l1} and the input channels of Conv_{l2}."""
    k1 = paf[f"params/Conv_{l1}/kernel"]
    k2 = pbf[f"params/Conv_{l1}/kernel"]
    assert k1.shape == k2.shape
    ri, ci = linear_sum_assignment(cosine_similarity(
        jnp.reshape(jnp.moveaxis(k1, -1, 0), (k1.shape[-1], -1)),
        jnp.reshape(jnp.moveaxis(k2, -1, 0), (k2.shape[-1], -1))),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return {
        **pbf, f"params/Conv_{l1}/kernel": pbf[f"params/Conv_{l1}/kernel"][:, :, :, ci],
        f"params/Conv_{l1}/bias": pbf[f"params/Conv_{l1}/bias"][ci],
        f"params/Conv_{l2}/kernel": pbf[f"params/Conv_{l2}/kernel"][:, :, ci, :]
    }

  def dense_dense(paf, pbf, l1, l2):
    """Permute the output channels of Dense_{l1} and the input channels of Dense_{l2}."""
    k1 = paf[f"params/Dense_{l1}/kernel"]
    k2 = pbf[f"params/Dense_{l1}/kernel"]
    assert k1.shape == k2.shape
    ri, ci = linear_sum_assignment(cosine_similarity(k1.T, k2.T), maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return {
        **pbf, f"params/Dense_{l1}/kernel": pbf[f"params/Dense_{l1}/kernel"][:, ci],
        f"params/Dense_{l1}/bias": pbf[f"params/Dense_{l1}/bias"][ci],
        f"params/Dense_{l2}/kernel": pbf[f"params/Dense_{l2}/kernel"][ci, :]
    }

  pbf_new = {**pbf}

  # (Conv_0, Conv_1)
  pbf_new = conv_conv(paf, pbf_new, 0, 1)

  # (Conv_1, Conv_2)
  pbf_new = conv_conv(paf, pbf_new, 1, 2)

  # (Dense_0, Dense_1)
  pbf_new = dense_dense(paf, pbf_new, 0, 1)

  return unflatten_params(pbf_new)

def test_cosine_similarity():
  rp = RngPooper(random.PRNGKey(0))

  for _ in range(10):
    X = random.normal(rp.poop(), (3, 5))
    Y = random.normal(rp.poop(), (7, 5))
    assert jnp.allclose(1 - cosine_similarity(X, Y), cdist(X, Y, metric="cosine"))

def test_permutify():
  rp = RngPooper(random.PRNGKey(0))

  model = TestModel()
  p1 = model.init(rp.poop(), jnp.zeros((1, 28, 28, 1)))
  p2 = model.init(rp.poop(), jnp.zeros((1, 28, 28, 1)))
  # print(tree_map(jnp.shape, flatten_params(p1)))

  new_p2 = permutify(p1, p2)

  # Test that the model is the same after permutation.
  random_input = random.normal(rp.poop(), (128, 28, 28, 1))
  # print(jnp.max(jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))))
  assert ((jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))) < 1e-5).all()

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

  epoch = 49
  model = ConvNetModel()

  def load_checkpoint(run, epoch):
    # See https://github.com/wandb/client/issues/3247.
    # f = wandb.restore(f"checkpoint_{epoch}", run)
    _, _, run_id = run.split("/")
    files = glob(f"./wandb/run-*-{run_id}/files/checkpoint_{epoch}")
    assert len(files) == 1
    with open(files[0], "rb") as f:
      contents = f.read()
      _, ret = from_bytes((0, init_train_state(random.PRNGKey(0), 0.0, model)), contents)
      return ret

  model_a = load_checkpoint("skainswo/git-re-basin/n83xc9z7", epoch)
  model_b = load_checkpoint("skainswo/git-re-basin/33fxg7w5", epoch)

  lambdas = jnp.linspace(0, 1, num=10)
  train_loss_interp_clever = []
  test_loss_interp_clever = []
  train_acc_interp_clever = []
  test_acc_interp_clever = []

  train_loss_interp_naive = []
  test_loss_interp_naive = []
  train_acc_interp_naive = []
  test_acc_interp_naive = []

  stuff = make_stuff(model)
  train_ds, test_ds = get_datasets(test_mode=False)
  num_train_examples = train_ds.cardinality().numpy()
  num_test_examples = test_ds.cardinality().numpy()
  # Might as well use the larget batch size that we can fit into memory here.
  train_ds_batched = tfds.as_numpy(train_ds.batch(2048))
  test_ds_batched = tfds.as_numpy(test_ds.batch(2048))
  for lam in tqdm(lambdas):
    # TODO make this look like the permuted version below
    naive_p = tree_map(lambda a, b: lam * a + (1 - lam) * b, model_a.params, model_b.params)
    train_loss_interp_naive.append(stuff.dataset_loss(naive_p, train_ds_batched))
    test_loss_interp_naive.append(stuff.dataset_loss(naive_p, test_ds_batched))
    train_acc_interp_naive.append(
        stuff.dataset_total_correct(naive_p, train_ds_batched) / num_train_examples)
    test_acc_interp_naive.append(
        stuff.dataset_total_correct(naive_p, test_ds_batched) / num_test_examples)

    b2 = permutify({"params": model_a.params}, {"params": model_b.params})
    clever_p = tree_map(lambda a, b: lam * a + (1 - lam) * b, freeze({"params": model_a.params}),
                        b2)
    train_loss_interp_clever.append(stuff.dataset_loss(clever_p["params"], train_ds_batched))
    test_loss_interp_clever.append(stuff.dataset_loss(clever_p["params"], test_ds_batched))
    train_acc_interp_clever.append(
        stuff.dataset_total_correct(clever_p["params"], train_ds_batched) / num_train_examples)
    test_acc_interp_clever.append(
        stuff.dataset_total_correct(clever_p["params"], test_ds_batched) / num_test_examples)

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
  plt.savefig(f"mnist_convnet_interp_loss_epoch{epoch}.png", dpi=300)
  plt.close(fig)

  fig = plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                        train_acc_interp_clever, test_acc_interp_clever)
  plt.savefig(f"mnist_convnet_interp_accuracy_epoch{epoch}.png", dpi=300)
  plt.close(fig)
