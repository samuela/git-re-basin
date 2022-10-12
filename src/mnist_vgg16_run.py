"""Train VGG16 on MNIST."""
import argparse

import augmax
import flax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap
from tqdm import tqdm

import wandb
from mnist_mlp_run import load_datasets
from utils import ec2_get_instance_type, rngmix, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def make_vgg(backbone_layers, classifier_width: int, norm):

  class VGG(nn.Module):

    @nn.compact
    def __call__(self, x):
      for l in backbone_layers:
        if isinstance(l, int):
          x = nn.Conv(features=l, kernel_size=(3, 3))(x)
          x = norm()(x)
          x = nn.relu(x)
        elif l == "m":
          x = nn.max_pool(x, (2, 2), strides=(2, 2))
        else:
          raise

      # Classifier
      # Note: everyone seems to do a different thing here.
      # * https://github.com/davisyoshida/vgg16-haiku/blob/4ef0bd001bf9daa4cfb2fa83ea3956ec01add3a8/vgg/vgg.py#L56
      #     does average pooling with a kernel size of (7, 7)
      # * https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/models/vgg.py#L37
      #     does average pooling with a kernel size of (1, 1) which doesn't seem
      #     to accomplish anything. See https://github.com/kuangliu/pytorch-cifar/issues/110.
      #     But this paper also doesn't really do the dense layers the same as in
      #     the paper either...
      # * The paper itself doesn't mention any kind of pooling...
      #
      # I'll stick to replicating the paper as closely as possible for now.
      (_b, w, h, _c) = x.shape
      assert w == h == 1
      x = jnp.reshape(x, (x.shape[0], -1))
      x = nn.Dense(classifier_width)(x)
      x = nn.relu(x)
      x = nn.Dense(classifier_width)(x)
      x = nn.relu(x)
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

  return VGG

TestVGG = make_vgg(
    [64, 64, "m", 64, 64, "m", 64, 64, 64, "m", 64, 64, 64, "m", 64, 64, 64, "m"],
    classifier_width=8,
    #  norm=lambda: lambda x: x,
    norm=nn.LayerNorm)

VGG16 = make_vgg(
    [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
    classifier_width=4096,
    norm=nn.LayerNorm)

VGG16ThinClassifier = make_vgg(
    [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
    classifier_width=512,
    norm=nn.LayerNorm)

def make_vgg_width_ablation(width_multiplier: int):
  m = width_multiplier
  return make_vgg([
      m * 1, m * 1, "m", m * 2, m * 2, "m", m * 4, m * 4, m * 4, "m", m * 8, m * 8, m * 8, "m",
      m * 8, m * 8, m * 8, "m"
  ],
                  classifier_width=m * 8,
                  norm=nn.LayerNorm)()

# 378.2MB
VGG16Wide = make_vgg(
    [512, 512, "m", 512, 512, "m", 512, 512, 512, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
    classifier_width=4096,
    norm=nn.LayerNorm)

VGG19 = make_vgg([
    64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512,
    512, "m"
],
                 classifier_width=4096,
                 norm=nn.LayerNorm)

def make_stuff(model):
  # Applied to all input images, test and train.
  normalize_transform = augmax.Chain(augmax.Resize(32, 32), augmax.ByteToFloat())

  @jit
  def batch_eval(params, images_u8, labels):
    images_f32 = vmap(normalize_transform)(None, images_u8)
    y_onehot = jax.nn.one_hot(labels, 10)
    logits = model.apply({"params": params}, images_f32)
    l = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    return l, {"num_correct": num_correct}

  @jit
  def step(rng, train_state, images_u8, labels):
    # images_transformed = vmap(train_transform)(random.split(rng, images_u8.shape[0]), images_u8)
    (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_u8, labels)
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
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy,
  }

def init_train_state(rng, model, learning_rate, num_epochs, batch_size, num_train_examples):
  # See https://github.com/kuangliu/pytorch-cifar.
  warmup_epochs = 1
  steps_per_epoch = num_train_examples // batch_size
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=1e-6,
      peak_value=learning_rate,
      warmup_steps=warmup_epochs * steps_per_epoch,
      # Confusingly, `decay_steps` is actually the total number of steps,
      # including the warmup.
      decay_steps=num_epochs * steps_per_epoch,
  )
  tx = optax.chain(optax.add_decayed_weights(5e-4), optax.sgd(lr_schedule, momentum=0.9))
  # tx = optax.adamw(learning_rate=lr_schedule, weight_decay=5e-4)
  vars = model.init(rng, jnp.zeros((1, 32, 32, 1)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--width-multiplier", type=int, default=64)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["mnist", "vgg16"],
      mode="disabled" if args.test else "online",
      job_type="train",
  ) as wandb_run:
    artifact = wandb.Artifact("mnist-vgg16-weights", type="model-weights")

    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.test = args.test
    config.seed = args.seed
    config.learning_rate = 0.001
    config.num_epochs = 25
    config.width_multiplier = args.width_multiplier
    config.batch_size = 100

    rng = random.PRNGKey(config.seed)

    # model = TestVGG() if config.test else VGG16ThinClassifier()
    model = make_vgg_width_ablation(config.width_multiplier)
    with timeblock("load datasets"):
      train_ds, test_ds = load_datasets()
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
                                   num_train_examples=train_ds["images_u8"].shape[0])

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
          with artifact.new_file(f"checkpoint{epoch}", mode="wb") as f:
            f.write(flax.serialization.to_bytes(train_state))

    # This will be a no-op when config.test is enabled anyhow, since wandb will
    # be initialized with mode="disabled".
    wandb_run.log_artifact(artifact)
