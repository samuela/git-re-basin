import sys

import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from jax import jit, random, tree_map, value_and_grad
from tqdm import tqdm

import wandb
from parallel_mnist_plots import plot_interp_acc, plot_interp_loss
from permutations import permutify
from utils import RngPooper, ec2_get_instance_type, timeblock

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

config = wandb.config
config.ec2_instance_type = ec2_get_instance_type()
config.smoke_test = "--test" in sys.argv
config.learning_rate = 0.001
config.num_epochs = 10 if config.smoke_test else 50
config.batch_size = 7 if config.smoke_test else 256

wandb.init(entity="skainswo",
           project="git-re-basin",
           tags=["cifar10"],
           mode="disabled" if config.smoke_test else "online")

rp = RngPooper(random.PRNGKey(0))

activation = nn.relu

if config.smoke_test:

  class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=8, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = jnp.reshape(x, (x.shape[0], -1))
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

else:

  class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
      x = activation(x)
      x = jnp.reshape(x, (x.shape[0], -1))
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

model = Model()

@jit
def batch_loss(params, x, y):
  logits = model.apply(params, x)
  return -jnp.mean(jnp.sum(y * logits, axis=-1))

@jit
def batch_num_correct(params, x, y):
  logits = model.apply(params, x)
  return jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))

@jit
def step(opt_state, params, x, y):
  l, g = value_and_grad(batch_loss)(params, x, y)
  updates, opt_state = tx.update(g, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state, l

# See https://github.com/tensorflow/tensorflow/issues/53831.
train_ds = tfds.load("cifar10", split="train", as_supervised=True)
test_ds = tfds.load("cifar10", split="test", as_supervised=True)

# Note: The take/cache warning:
#     2022-01-25 07:32:58.144059: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
# is not because we're actually doing this in the wrong order, but rather that
# the dataset is loaded in and called .cache() on before we receive it.
if config.smoke_test:
  train_ds = train_ds.take(13)
  test_ds = test_ds.take(17)

# Normalize 0-255 pixel values to 0.0-1.0
normalize = lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.one_hot(label, depth=10))
train_ds = train_ds.map(normalize).cache()
test_ds = test_ds.map(normalize).cache()

num_train_examples = train_ds.cardinality().numpy()
num_test_examples = test_ds.cardinality().numpy()

def dataset_loss(params, ds):
  # Note that we multiply by the batch size here, in order to get the sum of the
  # losses, then average over the whole dataset.
  return jnp.mean(jnp.array([x.shape[0] * batch_loss(params, x, y) for x, y in ds]))

def dataset_total_correct(params, ds):
  return jnp.sum(jnp.array([batch_num_correct(params, x, y) for x, y in ds]))

tx = optax.adam(config.learning_rate)

params1 = model.init(rp.poop(), jnp.zeros((1, 32, 32, 3)))
params2 = model.init(rp.poop(), jnp.zeros((1, 32, 32, 3)))
opt_state1 = tx.init(params1)
opt_state2 = tx.init(params2)
for epoch in tqdm(range(config.num_epochs)):
  with timeblock(f"Epoch"):
    for images, labels in tfds.as_numpy(
        train_ds.shuffle(num_train_examples, seed=hash(f"{epoch}-1")).batch(config.batch_size)):
      params1, opt_state1, loss1 = step(opt_state1, params1, images, labels)
    for images, labels in tfds.as_numpy(
        train_ds.shuffle(num_train_examples, seed=hash(f"{epoch}-2")).batch(config.batch_size)):
      params2, opt_state2, loss2 = step(opt_state2, params2, images, labels)

  train_ds_batched = tfds.as_numpy(train_ds.batch(config.batch_size))
  test_ds_batched = tfds.as_numpy(test_ds.batch(config.batch_size))

  # This is inclusive on both ends.
  lambdas = jnp.linspace(0, 1, num=10)

  # TODO implement permutify for convnets!
  # params2_permuted = permutify(params1, params2)
  params2_permuted = params2

  def interp_naive(lam):
    return tree_map(lambda a, b: b * lam + a * (1 - lam), params1, params2)

  def interp_clever(lam):
    return tree_map(lambda a, b: b * lam + a * (1 - lam), params1, params2_permuted)

  with timeblock("Interpolation plot"):
    train_loss_interp_naive = jnp.array(
        [dataset_loss(interp_naive(l), train_ds_batched) for l in lambdas])
    test_loss_interp_naive = jnp.array(
        [dataset_loss(interp_naive(l), test_ds_batched) for l in lambdas])
    train_acc_interp_naive = jnp.array(
        [dataset_total_correct(interp_naive(l), train_ds_batched)
         for l in lambdas]) / num_train_examples
    test_acc_interp_naive = jnp.array(
        [dataset_total_correct(interp_naive(l), test_ds_batched)
         for l in lambdas]) / num_test_examples

    train_loss_interp_clever = jnp.array(
        [dataset_loss(interp_clever(l), train_ds_batched) for l in lambdas])
    test_loss_interp_clever = jnp.array(
        [dataset_loss(interp_clever(l), test_ds_batched) for l in lambdas])
    train_acc_interp_clever = jnp.array(
        [dataset_total_correct(interp_clever(l), train_ds_batched)
         for l in lambdas]) / num_train_examples
    test_acc_interp_clever = jnp.array(
        [dataset_total_correct(interp_clever(l), test_ds_batched)
         for l in lambdas]) / num_test_examples

  # These are redundant with the full arrays above, but we want pretty plots in
  # wandb.
  train_loss1 = train_loss_interp_naive[0]
  train_loss2 = train_loss_interp_naive[-1]
  test_loss1 = test_loss_interp_naive[0]
  test_loss2 = test_loss_interp_naive[-1]
  train_acc1 = train_acc_interp_naive[0]
  train_acc2 = train_acc_interp_naive[-1]
  test_acc1 = test_acc_interp_naive[0]
  test_acc2 = test_acc_interp_naive[-1]

  interp_loss_plot = plot_interp_loss(epoch, lambdas, train_loss_interp_naive,
                                      test_loss_interp_naive, train_loss_interp_clever,
                                      test_loss_interp_clever)
  interp_acc_plot = plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                                    train_acc_interp_clever, test_acc_interp_clever)
  wandb.log({
      "epoch": epoch,
      "train_loss1": train_loss1,
      "train_loss2": train_loss2,
      "train_acc1": train_acc1,
      "train_acc2": train_acc2,
      "test_loss1": test_loss1,
      "test_loss2": test_loss2,
      "test_acc1": test_acc1,
      "test_acc2": test_acc2,
      # This doesn't really change, but it's more convenient to store it here
      # when we go to make videos/plots later.
      "lambdas": lambdas,
      "train_loss_interp_naive": train_loss_interp_naive,
      "test_loss_interp_naive": test_loss_interp_naive,
      "train_acc_interp_naive": train_acc_interp_naive,
      "test_acc_interp_naive": test_acc_interp_naive,
      "train_loss_interp_clever": train_loss_interp_clever,
      "test_loss_interp_clever": test_loss_interp_clever,
      "train_acc_interp_clever": train_acc_interp_clever,
      "test_acc_interp_clever": test_acc_interp_clever,
      "interp_loss_plot": wandb.Image(interp_loss_plot),
      "interp_acc_plot": wandb.Image(interp_acc_plot),
  })
