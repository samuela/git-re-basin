"""Train many MNIST MLPs, each seeing a random subset of the dataset. Then merge
the models with MergeMany and evaluate calibration, etc."""
import augmax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap
from tqdm import tqdm

import matplotlib_style as _
from utils import (flatten_params, lerp, rngmix, timeblock, tree_l2, unflatten_params)
from weight_matching import (PermutationSpec, apply_permutation, mlp_permutation_spec,
                             weight_matching)

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def load_mnist():
  """Return the training and test datasets, unbatched."""
  # See https://www.tensorflow.org/datasets/overview#as_batched_tftensor_batch_size-1.
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("mnist", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("mnist", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

# def load_fashion_mnist():
#   # See https://www.tensorflow.org/datasets/overview#as_batched_tftensor_batch_size-1.
#   train_ds_images_u8, train_ds_labels = tfds.as_numpy(
#       tfds.load("mnist", split="train", batch_size=-1, as_supervised=True))
#   test_ds_images_u8, test_ds_labels = tfds.as_numpy(
#       tfds.load("mnist", split="test", batch_size=-1, as_supervised=True))
#   train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
#   test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
#   return train_ds, test_ds

activation = nn.relu

class MLPModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = jnp.reshape(x, (-1, 28 * 28))
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

def make_stuff(model):
  normalize_transform = augmax.ByteToFloat()

  @jit
  def batch_eval(params, images_u8, labels):
    images_f32 = vmap(normalize_transform)(None, images_u8)
    logits = model.apply({"params": params}, images_f32)
    y_onehot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_onehot, axis=-1))
    return loss, {"logits": logits, "num_correct": num_correct}

  @jit
  def step(train_state, images_f32, labels):
    (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_f32, labels)
    return train_state.apply_gradients(grads=g), {"batch_loss": l, **info}

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    num_examples = dataset["images_u8"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, infos = zip(*[
        batch_eval(
            params,
            dataset["images_u8"][batch_ix[i, :], :, :, :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    return (
        jnp.sum(batch_size * jnp.array(losses)) / num_examples,
        sum(x["num_correct"] for x in infos) / num_examples,
    )

  def dataset_logits(params, dataset, batch_size: int):
    num_examples = dataset["images_u8"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    _, infos = zip(*[
        batch_eval(
            params,
            dataset["images_u8"][batch_ix[i, :], :, :, :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    return jnp.concatenate([x["logits"] for x in infos])

  return {
      "normalize_transform": normalize_transform,
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy,
      "dataset_logits": dataset_logits
  }

if __name__ == "__main__":
  num_models = 32

  batch_size = 500
  learning_rate = 1e-3
  num_epochs = 100

  model = MLPModel()
  stuff = make_stuff(model)

  with timeblock("load datasets"):
    train_ds, test_ds = load_mnist()
    print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
    print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

    num_train_examples = train_ds["images_u8"].shape[0]
    num_test_examples = test_ds["images_u8"].shape[0]
    assert num_train_examples % batch_size == 0
    print("num_train_examples", num_train_examples)
    print("num_test_examples", num_test_examples)

  def train_one(seed: int):
    rng = random.PRNGKey(seed)
    tx = optax.adam(learning_rate)

    num_subset_examples = num_train_examples // 2
    subset_ix = random.permutation(rngmix(rng, "subset_ix"),
                                   jnp.arange(num_train_examples))[0:num_subset_examples]
    subset_ds = {
        "images_u8": train_ds["images_u8"][subset_ix, :, :, :],
        "labels": train_ds["labels"][subset_ix],
    }

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rngmix(rng, "init"), jnp.zeros((1, 28, 28, 1)))["params"],
        tx=tx,
    )

    for epoch in tqdm(range(num_epochs)):
      infos = []
      batch_ix = random.permutation(rngmix(rng, f"epoch-{epoch}"), num_subset_examples).reshape(
          (-1, batch_size))
      for i in range(batch_ix.shape[0]):
        p = batch_ix[i, :]
        images_u8 = subset_ds["images_u8"][p, :, :, :]
        labels = subset_ds["labels"][p]
        train_state, info = stuff["step"](train_state, images_u8, labels)
        infos.append(info)

      # train_loss = sum(batch_size * x["batch_loss"] for x in infos) / num_train_examples
      # train_accuracy = sum(x["num_correct"] for x in infos) / num_train_examples

    return train_state.params

  all_params = [train_one(i) for i in range(num_models)]

  for i, params in enumerate(all_params):
    train_loss, train_acc = stuff['dataset_loss_and_accuracy'](params, train_ds, 10_000)
    test_loss, test_acc = stuff['dataset_loss_and_accuracy'](params, test_ds, 10_000)
    print(f"{i}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

  tree_mean = lambda args: tree_map(lambda *x: sum(x) / len(x), *args)

  ### Naive
  params_naive = tree_mean(all_params)
  train_loss_naive, train_acc_naive = stuff["dataset_loss_and_accuracy"](params_naive, train_ds,
                                                                         10_000)
  test_loss_naive, test_acc_naive = stuff["dataset_loss_and_accuracy"](params_naive, test_ds,
                                                                       10_000)
  print(f"[naive] train loss: {train_loss_naive:.4f}, train accuracy: {train_acc_naive:.4f} "
        f"test loss: {test_loss_naive:.4f}, test accuracy: {test_acc_naive:.4f}")

  permutation_spec = mlp_permutation_spec(3)

  def match2(p1, p2):
    perm = weight_matching(random.PRNGKey(123),
                           permutation_spec,
                           flatten_params(p1),
                           flatten_params(p2),
                           silent=True)
    p2_clever = unflatten_params(apply_permutation(permutation_spec, perm, flatten_params(p2)))
    return lerp(0.5, p1, p2_clever)

  params01 = match2(all_params[0], all_params[1])
  test_loss_01, test_acc_01 = stuff['dataset_loss_and_accuracy'](params01, test_ds, 10_000)
  print(f"test loss 0->1: {test_loss_01:.4f}, test accuracy 0->1: {test_acc_01:.4f}")

  def match_many(rng, permutation_spec: PermutationSpec, ps, max_iter=100):
    for iteration in range(max_iter):
      progress = False
      for p_ix in random.permutation(rngmix(rng, iteration), len(ps)):
        other_models_mean = tree_mean(ps[:p_ix] + ps[p_ix + 1:])
        l2_before = tree_l2(other_models_mean, ps[p_ix])
        perm = weight_matching(rngmix(rng, f"{iteration}-{p_ix}"),
                               permutation_spec,
                               flatten_params(other_models_mean),
                               flatten_params(ps[p_ix]),
                               silent=True)
        ps[p_ix] = unflatten_params(
            apply_permutation(permutation_spec, perm, flatten_params(ps[p_ix])))
        l2_after = tree_l2(other_models_mean, ps[p_ix])
        progress = progress or l2_after < l2_before - 1e-12
        print(f"iteration {iteration}/model {p_ix}: l2 diff {l2_after - l2_before:.4f}")

      if not progress:
        break

    return ps

  params_barycenter = tree_mean(match_many(random.PRNGKey(123), permutation_spec, all_params))
  train_loss_barycenter, train_acc_barycenter = stuff["dataset_loss_and_accuracy"](
      params_barycenter, train_ds, 10_000)
  test_loss_barycenter, test_acc_barycenter = stuff["dataset_loss_and_accuracy"](params_barycenter,
                                                                                 test_ds, 10_000)
  print(
      f"[barycenter] train loss: {train_loss_barycenter:.4f}, train accuracy: {train_acc_barycenter:.4f} "
      f"test loss: {test_loss_barycenter:.4f}, test accuracy: {test_acc_barycenter:.4f}")

  ### Plotting
  plt.figure(figsize=(12, 6))

  num_bins = 10
  bins = np.linspace(0, 1, num_bins + 1)
  bin_locations = 0.5 * (bins[:-1] + bins[1:])

  def one(bin_ix, probs, labels):
    lo, hi = bins[bin_ix], bins[bin_ix + 1]
    mask = (lo <= probs) & (probs <= hi)
    y_onehot = jax.nn.one_hot(labels, 10)
    return np.mean(y_onehot[mask])

  # Train
  plt.subplot(1, 2, 1)
  plotting_ds = train_ds

  individual_model_logits = [
      stuff["dataset_logits"](p, plotting_ds, 10_000) for p in tqdm(all_params)
  ]
  barycenter_logits = stuff["dataset_logits"](params_barycenter, plotting_ds, 10_000)
  naive_logits = stuff["dataset_logits"](params_naive, plotting_ds, 10_000)
  ensemble_logits = sum(individual_model_logits) / num_models

  individual_model_probs = [jax.nn.softmax(x) for x in individual_model_logits]
  barycenter_probs = jax.nn.softmax(barycenter_logits)
  naive_probs = jax.nn.softmax(naive_logits)
  ensemble_probs = jax.nn.softmax(ensemble_logits)

  individual_model_ys = [[one(ix, probs, plotting_ds["labels"]) for ix in range(num_bins)]
                         for probs in tqdm(individual_model_probs)]
  wm_ys = [one(ix, barycenter_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  naive_ys = [one(ix, naive_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  ensemble_ys = [one(ix, ensemble_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

  plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")

  plt.plot([], [], color="tab:orange", alpha=0.5, label="Individual models")
  for ys in individual_model_ys:
    plt.plot(bin_locations, ys, color="tab:orange", alpha=0.25)

  plt.plot(bin_locations,
           np.nan_to_num(naive_ys),
           color="tab:grey",
           marker=".",
           label="Naïve merging")
  plt.plot(bin_locations,
           np.nan_to_num(ensemble_ys),
           color="tab:purple",
           marker="2",
           label="Model ensemble")
  plt.plot(bin_locations, wm_ys, color="tab:green", marker="^", linewidth=2, label="MergeMany")
  plt.xlabel("Predicted probability")
  plt.ylabel("True probability")
  plt.axis("equal")
  plt.legend()
  plt.title("Train")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xticks(np.linspace(0, 1, 5))
  plt.yticks(np.linspace(0, 1, 5))

  # Test
  plt.subplot(1, 2, 2)
  plotting_ds = test_ds

  individual_model_logits = [
      stuff["dataset_logits"](p, plotting_ds, 10_000) for p in tqdm(all_params)
  ]
  barycenter_logits = stuff["dataset_logits"](params_barycenter, plotting_ds, 10_000)
  naive_logits = stuff["dataset_logits"](params_naive, plotting_ds, 10_000)
  ensemble_logits = sum(individual_model_logits) / num_models

  individual_model_probs = [jax.nn.softmax(x) for x in individual_model_logits]
  barycenter_probs = jax.nn.softmax(barycenter_logits)
  naive_probs = jax.nn.softmax(naive_logits)
  ensemble_probs = jax.nn.softmax(ensemble_logits)

  individual_model_ys = [[one(ix, probs, plotting_ds["labels"]) for ix in range(num_bins)]
                         for probs in tqdm(individual_model_probs)]
  wm_ys = [one(ix, barycenter_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  naive_ys = [one(ix, naive_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  ensemble_ys = [one(ix, ensemble_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

  plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")

  plt.plot([], [], color="tab:orange", alpha=0.5, linestyle="dashed", label="Individual models")
  for ys in individual_model_ys:
    plt.plot(bin_locations, ys, color="tab:orange", linestyle="dashed", alpha=0.25)

  plt.plot(bin_locations,
           np.nan_to_num(naive_ys),
           color="tab:grey",
           marker=".",
           linestyle="dashed",
           label="Naïve merging")
  plt.plot(bin_locations,
           np.nan_to_num(ensemble_ys),
           color="tab:purple",
           marker="2",
           linestyle="dashed",
           label="Model ensemble")
  plt.plot(bin_locations,
           wm_ys,
           color="tab:green",
           marker="^",
           linewidth=2,
           linestyle="dashed",
           label="MergeMany")
  plt.xlabel("Predicted probability")
  plt.ylabel("True probability")
  plt.axis("equal")
  # plt.legend()
  plt.title("Test")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xticks(np.linspace(0, 1, 5))
  plt.yticks(np.linspace(0, 1, 5))

  plt.suptitle("MergeMany Calibration")
  plt.tight_layout()
  plt.savefig("figs/mnist_wm_many_calibration_plot.png", dpi=300)
  plt.savefig("figs/mnist_wm_many_calibration_plot.pdf")
