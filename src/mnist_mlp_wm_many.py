"""TODO"""
import augmax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap
from tqdm import tqdm

from mnist_mlp_train import load_datasets
from utils import (flatten_params, lerp, rngmix, timeblock, tree_l2,
                   unflatten_params)
from weight_matching import (PermutationSpec, apply_permutation,
                             mlp_permutation_spec, weight_matching)

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

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
    return loss, {"num_correct": num_correct}

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

  return {
      "normalize_transform": normalize_transform,
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy
  }

if __name__ == "__main__":
  # rng = random.PRNGKey(123)

  num_models = 5

  batch_size = 500
  learning_rate = 1e-3
  num_epochs = 100

  model = MLPModel()
  stuff = make_stuff(model)

  with timeblock("load_datasets"):
    train_ds, test_ds = load_datasets()
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

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rngmix(rng, "init"), jnp.zeros((1, 28, 28, 1)))["params"],
        tx=tx,
    )

    for epoch in tqdm(range(num_epochs)):
      infos = []
      batch_ix = random.permutation(rngmix(rng, f"epoch-{epoch}"), num_train_examples).reshape(
          (-1, batch_size))
      for i in range(batch_ix.shape[0]):
        p = batch_ix[i, :]
        images_u8 = train_ds["images_u8"][p, :, :, :]
        labels = train_ds["labels"][p]
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

  tree_mean = lambda args: tree_map(lambda *x: sum(x) / len(x), *args)

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

  params_barycenter = tree_mean(
      match_many(random.PRNGKey(123), permutation_spec, all_params))
  train_loss_barycenter, train_acc_barycenter = stuff['dataset_loss_and_accuracy'](
      params_barycenter, train_ds, 10_000)
  test_loss_barycenter, test_acc_barycenter = stuff['dataset_loss_and_accuracy'](params_barycenter,
                                                                                 test_ds, 10_000)
  print(
      f"[barycenter] train loss: {train_loss_barycenter:.4f}, train accuracy: {train_acc_barycenter:.4f} "
      f"test loss: {test_loss_barycenter:.4f}, test accuracy: {test_acc_barycenter:.4f}")
