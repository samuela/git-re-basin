"""Train a MLP on CIFAR-10 on one random seed."""
import argparse

import flax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random, tree_map
from tqdm import tqdm

from cifar10_vgg_run import make_stuff
from datasets import load_cifar10
from utils import ec2_get_instance_type, flatten_params, rngmix, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

activation = nn.relu

class MLPModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = jnp.reshape(x, (-1, 32 * 32 * 3))
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], required=True)
  parser.add_argument("--learning-rate", type=float, required=True)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar10", "mlp", "training"],
      mode="disabled" if args.test else "online",
      job_type="train",
  ) as wandb_run:
    artifact = wandb.Artifact("cifar10-mlp-weights", type="model-weights")

    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.test = args.test
    config.seed = args.seed
    config.optimizer = args.optimizer
    config.learning_rate = args.learning_rate
    config.num_epochs = 100
    config.batch_size = 100

    rng = random.PRNGKey(config.seed)

    model = MLPModel()
    stuff = make_stuff(model)

    with timeblock("load datasets"):
      train_ds, test_ds = load_cifar10()
      print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
      print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

      num_train_examples = train_ds["images_u8"].shape[0]
      num_test_examples = test_ds["images_u8"].shape[0]
      assert num_train_examples % config.batch_size == 0
      print("num_train_examples", num_train_examples)
      print("num_test_examples", num_test_examples)

    if config.optimizer == "sgd":
      lr_schedule = optax.warmup_cosine_decay_schedule(
          init_value=1e-6,
          peak_value=config.learning_rate,
          warmup_steps=num_train_examples // config.batch_size,
          # Confusingly, `decay_steps` is actually the total number of steps,
          # including the warmup.
          decay_steps=config.num_epochs * (num_train_examples // config.batch_size),
      )
      # tx = optax.sgd(lr_schedule, momentum=0.9)
      tx = optax.chain(optax.add_decayed_weights(5e-4), optax.sgd(lr_schedule, momentum=0.9))
    elif config.optimizer == "adam":
      tx = optax.adam(config.learning_rate)
    else:
      tx = optax.adamw(config.learning_rate, weight_decay=5e-4)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rngmix(rng, "init"), jnp.zeros((1, 32, 32, 3)))["params"],
        tx=tx,
    )

    for epoch in tqdm(range(config.num_epochs)):
      infos = []
      with timeblock(f"Epoch"):
        batch_ix = random.permutation(rngmix(rng, f"epoch-{epoch}"), num_train_examples).reshape(
            (-1, config.batch_size))
        batch_rngs = random.split(rngmix(rng, f"batch_rngs-{epoch}"), batch_ix.shape[0])
        for i in range(batch_ix.shape[0]):
          p = batch_ix[i, :]
          images_u8 = train_ds["images_u8"][p, :, :, :]
          labels = train_ds["labels"][p]
          train_state, info = stuff["step"](batch_rngs[i], train_state, images_u8, labels)
          infos.append(info)

      train_loss = sum(config.batch_size * x["batch_loss"] for x in infos) / num_train_examples
      train_accuracy = sum(x["num_correct"] for x in infos) / num_train_examples

      # Evaluate test loss/accuracy
      with timeblock("Test set eval"):
        test_loss, test_accuracy = stuff["dataset_loss_and_accuracy"](train_state.params, test_ds,
                                                                      1000)

      params_l2 = tree_map(lambda x: jnp.sqrt(jnp.sum(x**2)),
                           flatten_params({"params_l2": train_state.params}))

      # See https://github.com/wandb/client/issues/3690.
      wandb_run.log({
          "epoch": epoch,
          "train_loss": train_loss,
          "test_loss": test_loss,
          "train_accuracy": train_accuracy,
          "test_accuracy": test_accuracy,
          **params_l2
      })

      # No point saving the model at all if we're running in test mode.
      # With layer width 512, the MLP is 3.7MB per checkpoint.
      if not config.test:
        with timeblock("model serialization"):
          with artifact.new_file(f"checkpoint{epoch}", mode="wb") as f:
            f.write(flax.serialization.to_bytes(train_state.params))

    # This will be a no-op when config.test is enabled anyhow, since wandb will
    # be initialized with mode="disabled".
    wandb_run.log_artifact(artifact)
