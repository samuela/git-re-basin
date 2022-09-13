"""See https://github.com/google/jax/discussions/10674."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad

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

  @jit
  def batch_eval(params, images_u8, labels):
    images_f32 = jnp.array(images_u8, dtype=jnp.float32) / 256.0
    logits = model.apply({"params": params}, images_f32)
    y_onehot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_onehot, axis=-1))
    return loss, num_correct

  @jit
  def step(train_state, images_f32, labels):
    (l, num_correct), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_f32,
                                                                   labels)
    return train_state.apply_gradients(grads=g), l

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    num_examples = dataset["images_u8"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, num_corrects = zip(*[
        batch_eval(
            params,
            dataset["images_u8"][batch_ix[i, :], :, :, :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    losses = jnp.array(losses)
    num_corrects = jnp.array(num_corrects)
    return jnp.sum(batch_size * losses) / num_examples, jnp.sum(num_corrects) / num_examples

  return {
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy
  }

def get_datasets():
  num_train = 1000
  num_test = 1000
  return {
      "images_u8":
      random.choice(random.PRNGKey(0), jnp.arange(256, dtype=jnp.uint8), (num_train, 28, 28, 1)),
      "labels":
      random.choice(random.PRNGKey(2), jnp.arange(10, dtype=jnp.uint8), (num_train, ))
  }, {
      "images_u8":
      random.choice(random.PRNGKey(3), jnp.arange(256, dtype=jnp.uint8), (num_test, 28, 28, 1)),
      "labels":
      random.choice(random.PRNGKey(4), jnp.arange(10, dtype=jnp.uint8), (num_test, ))
  }

def init_train_state(rng, learning_rate, model):
  tx = optax.adam(learning_rate)
  vars = model.init(rng, jnp.zeros((1, 28, 28, 1)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

def main():
  learning_rate = 0.001
  num_epochs = 10
  batch_size = 100

  train_ds, test_ds = get_datasets()
  print("train_ds images_u8 hash", hash(np.array(train_ds["images_u8"]).tobytes()))
  print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
  print("test_ds images_u8 hash", hash(np.array(test_ds["images_u8"]).tobytes()))
  print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

  num_train_examples = train_ds["images_u8"].shape[0]
  assert num_train_examples % batch_size == 0

  model = MLPModel()
  stuff = make_stuff(model)
  train_state = init_train_state(random.PRNGKey(123), learning_rate, model)

  for epoch in range(num_epochs):
    batch_ix = random.permutation(random.PRNGKey(epoch), num_train_examples).reshape(
        (-1, batch_size))
    for i in range(batch_ix.shape[0]):
      p = batch_ix[i, :]
      images_u8 = train_ds["images_u8"][p, :, :, :]
      labels = train_ds["labels"][p]
      train_state, batch_loss = stuff["step"](train_state, images_u8, labels)

    # Evaluate train/test loss/accuracy
    train_loss, train_accuracy = stuff["dataset_loss_and_accuracy"](train_state.params, train_ds,
                                                                    1000)
    test_loss, test_accuracy = stuff["dataset_loss_and_accuracy"](train_state.params, test_ds, 1000)

    print({
        "epoch": epoch,
        "batch_loss": float(batch_loss),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
    })

if __name__ == "__main__":
  main()
