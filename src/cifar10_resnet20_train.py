"""

See
* https://github.com/hushon/JAX-ResNet-CIFAR10/blob/main/resnet_cifar.py
* https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
import argparse

import augmax
import flax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap
from tqdm import tqdm

from cifar100_resnet20_train import NUM_CLASSES
from datasets import load_cifar10, load_cifar10_split
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import ec2_get_instance_type, rngmix, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

NUM_CLASSES = 10

def make_stuff(model):
  train_transform = augmax.Chain(
      # augmax does not seem to support random crops with padding. See https://github.com/khdlr/augmax/issues/6.
      augmax.RandomSizedCrop(32, 32, zoom_range=(0.8, 1.2)),
      augmax.HorizontalFlip(),
      augmax.Rotate(),
  )
  # Applied to all input images, test and train.
  normalize_transform = augmax.Chain(augmax.ByteToFloat(), augmax.Normalize())

  @jit
  def batch_eval(params, images_u8, labels):
    images_f32 = vmap(normalize_transform)(None, images_u8)
    y_onehot = jax.nn.one_hot(labels, NUM_CLASSES)
    logits = model.apply({"params": params}, images_f32)
    l = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    return l, {"num_correct": num_correct}

  @jit
  def step(rng, train_state, images, labels):
    images_transformed = vmap(train_transform)(random.split(rng, images.shape[0]), images)
    (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_transformed,
                                                            labels)
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
      "train_transform": train_transform,
      "normalize_transform": normalize_transform,
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy,
  }

def init_train_state(rng, model, learning_rate, num_epochs, batch_size, num_train_examples,
                     weight_decay: float):
  # See https://github.com/kuangliu/pytorch-cifar.
  warmup_epochs = 5
  steps_per_epoch = num_train_examples // batch_size
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=1e-6,
      peak_value=learning_rate,
      warmup_steps=warmup_epochs * steps_per_epoch,
      # Confusingly, `decay_steps` is actually the total number of steps,
      # including the warmup.
      decay_steps=num_epochs * steps_per_epoch,
  )
  tx = optax.chain(optax.add_decayed_weights(weight_decay), optax.sgd(lr_schedule, momentum=0.9))
  # tx = optax.adamw(learning_rate=lr_schedule, weight_decay=5e-4)
  vars = model.init(rng, jnp.zeros((1, 32, 32, 3)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--data-split", choices=["split1", "split2", "both"], required=True)
  parser.add_argument("--width-multiplier", type=int, default=1)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar10", "resnet", "training"],
      mode="disabled" if args.test else "online",
      job_type="train",
  ) as wandb_run:
    artifact = wandb.Artifact("cifar10-resnet-weights", type="model-weights")

    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.test = args.test
    config.seed = args.seed
    config.data_split = args.data_split
    config.learning_rate = 0.1
    config.num_epochs = 10 if args.test else 250
    config.batch_size = 100
    config.width_multiplier = args.width_multiplier
    config.weight_decay = args.weight_decay

    rng = random.PRNGKey(config.seed)

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=NUM_CLASSES,
                   width_multiplier=config.width_multiplier)

    with timeblock("load datasets"):
      if config.data_split == "both":
        train_ds, test_ds = load_cifar10()
      else:
        split1, split2, test_ds = load_cifar10_split()
        train_ds = split1 if config.data_split == "split1" else split2

      print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
      print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

      num_train_examples = train_ds["images_u8"].shape[0]
      num_test_examples = test_ds["images_u8"].shape[0]
      assert num_train_examples % config.batch_size == 0
      print("num_train_examples", num_train_examples)
      print("num_test_examples", num_test_examples)

    stuff = make_stuff(model)
    train_state = init_train_state(rngmix(rng, "init"),
                                   model=model,
                                   learning_rate=config.learning_rate,
                                   num_epochs=config.num_epochs,
                                   batch_size=config.batch_size,
                                   num_train_examples=train_ds["images_u8"].shape[0],
                                   weight_decay=config.weight_decay)

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

      # See https://github.com/wandb/client/issues/3690.
      wandb_run.log({
          "epoch": epoch,
          "train_loss": train_loss,
          "test_loss": test_loss,
          "train_accuracy": train_accuracy,
          "test_accuracy": test_accuracy,
      })

      # No point saving the model at all if we're running in test mode.
      if (not config.test) and (epoch % 10 == 0 or epoch == config.num_epochs - 1):
        with timeblock("model serialization"):
          # See https://github.com/wandb/client/issues/3823
          filename = f"/tmp/checkpoint{epoch}"
          with open(filename, mode="wb") as f:
            f.write(flax.serialization.to_bytes(train_state.params))
          artifact.add_file(filename)

    # This will be a no-op when config.test is enabled anyhow, since wandb will
    # be initialized with mode="disabled".
    wandb_run.log_artifact(artifact)
