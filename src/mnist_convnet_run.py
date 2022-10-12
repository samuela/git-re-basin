"""Train a convnet on MNIST on one random seed. Serialize the model for
interpolation downstream."""
import argparse

import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad
from tqdm import tqdm

import wandb
from utils import RngPooper, ec2_get_instance_type, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

activation = nn.relu

class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=8, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = activation(x)

    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(32)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

class ConvNetModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    # Take the mean along the channel dimension. Otherwise the following dense
    # layer is massive.
    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(1024)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

def make_stuff(model):
  ret = lambda: None

  @jit
  def batch_loss(params, images, y_onehot):
    logits = model.apply({"params": params}, images)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))

  @jit
  def batch_num_correct(params, images, y_onehot):
    logits = model.apply({"params": params}, images)
    return jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_onehot, axis=-1))

  @jit
  def step(train_state, images, y_onehot):
    l, g = value_and_grad(batch_loss)(train_state.params, images, y_onehot)
    return train_state.apply_gradients(grads=g), l

  def dataset_loss(params, ds):
    # Note that we multiply by the batch size here, in order to get the sum of the
    # losses, then average over the whole dataset.
    return jnp.mean(jnp.array([x.shape[0] * batch_loss(params, x, y) for x, y in ds]))

  def dataset_total_correct(params, ds):
    return jnp.sum(jnp.array([batch_num_correct(params, x, y) for x, y in ds]))

  ret.batch_loss = batch_loss
  ret.batch_num_correct = batch_num_correct
  ret.step = step
  ret.dataset_loss = dataset_loss
  ret.dataset_total_correct = dataset_total_correct
  return ret

def get_datasets(test_mode):
  """Return the training and test datasets, unbatched.

  test_mode: Whether or not we're running in "smoke test" mode.
  """
  train_ds = tfds.load("mnist", split="train", as_supervised=True)
  test_ds = tfds.load("mnist", split="test", as_supervised=True)
  # Note: The take/cache warning:
  #     2022-01-25 07:32:58.144059: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
  # is not because we're actually doing this in the wrong order, but rather that
  # the dataset is loaded in and called .cache() on before we receive it.
  if test_mode:
    train_ds = train_ds.take(13)
    test_ds = test_ds.take(17)

  # Normalize 0-255 pixel values to 0.0-1.0
  normalize = lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.one_hot(label, depth=10))
  train_ds = train_ds.map(normalize).cache()
  test_ds = test_ds.map(normalize).cache()
  return train_ds, test_ds

def init_train_state(rng, learning_rate, model):
  tx = optax.adam(learning_rate)
  vars = model.init(rng, jnp.zeros((1, 28, 28, 1)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--resume", type=str, help="wandb run to resume from (eg. 1kqqa9js)")
  parser.add_argument("--resume-epoch",
                      type=int,
                      help="The epoch to resume from. Required if --resume is set.")
  args = parser.parse_args()

  wandb.init(project="git-re-basin",
             entity="skainswo",
             tags=["mnist", "convnet"],
             resume="must" if args.resume is not None else None,
             id=args.resume,
             mode="disabled" if args.test else "online")

  # Note: hopefully it's ok that we repeat this even when resuming a run?
  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.test = args.test
  config.seed = args.seed
  config.learning_rate = 0.001
  config.num_epochs = 10 if config.test else 50
  config.batch_size = 7 if config.test else 512

  rp = RngPooper(random.PRNGKey(config.seed))

  model = TestModel() if config.test else ConvNetModel()
  stuff = make_stuff(model)

  train_ds, test_ds = get_datasets(test_mode=config.test)
  num_train_examples = train_ds.cardinality().numpy()
  num_test_examples = test_ds.cardinality().numpy()

  train_state = init_train_state(rp.poop(), config.learning_rate, model)
  start_epoch = 0

  if args.resume is not None:
    # Bring the the desired resume epoch into the wandb run directory so that it
    # can then be picked up by `restore_checkpoint` below.
    wandb.restore(f"checkpoint_{args.resume_epoch}")
    last_epoch, train_state = restore_checkpoint(wandb.run.dir, (0, train_state))
    # We need to increment last_epoch, because we store `(i, train_state)`
    # where `train_state` is the state _after_ i'th epoch. So we're actually
    # starting from the next epoch.
    start_epoch = last_epoch + 1

  for epoch in tqdm(range(start_epoch, config.num_epochs),
                    initial=start_epoch,
                    total=config.num_epochs):
    with timeblock(f"Epoch"):
      # Set the seed as a hash of the epoch and the overall random seed, so that
      # we ensure different seeds see different data orders, since tfds's random
      # seed is independent of our `RngPooper`.
      for images, labels in tfds.as_numpy(
          train_ds.shuffle(num_train_examples,
                           seed=hash(f"{config.seed}-{epoch}")).batch(config.batch_size)):
        train_state, batch_loss = stuff.step(train_state, images, labels)

    train_ds_batched = tfds.as_numpy(train_ds.batch(config.batch_size))
    test_ds_batched = tfds.as_numpy(test_ds.batch(config.batch_size))

    # Evaluate train/test loss/accuracy
    with timeblock("Model eval"):
      train_loss = stuff.dataset_loss(train_state.params, train_ds_batched)
      test_loss = stuff.dataset_loss(train_state.params, test_ds_batched)
      train_accuracy = stuff.dataset_total_correct(train_state.params,
                                                   train_ds_batched) / num_train_examples
      test_accuracy = stuff.dataset_total_correct(train_state.params,
                                                  test_ds_batched) / num_test_examples

    if not config.test:
      # See https://docs.wandb.ai/guides/track/advanced/save-restore
      save_checkpoint(wandb.run.dir, (epoch, train_state), epoch, keep_every_n_steps=10)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    })
