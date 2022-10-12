import argparse
from pathlib import Path

import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
from flax.serialization import from_bytes
from jax import jit, random, vmap
from tqdm import tqdm

from cifar100_resnet20_train import NUM_CLASSES, make_stuff
from datasets import load_cifar100
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import ec2_get_instance_type, lerp

NUM_CLASSES = 100

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--width-multiplier", type=int, required=True)
  parser.add_argument("--load-epoch", type=int, required=True)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar100", "resnet20", "ensembling"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.width_multiplier = args.width_multiplier
    config.seed = args.seed
    config.load_epoch = args.load_epoch

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=NUM_CLASSES,
                   width_multiplier=config.width_multiplier)
    stuff = make_stuff(model)

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())

    filename = f"checkpoint{config.load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"cifar100-resnet-weights:{config.model_a}").get_path(
                filename).download()))
    model_b = load_model(
        Path(
            wandb_run.use_artifact(f"cifar100-resnet-weights:{config.model_b}").get_path(
                filename).download()))

    train_ds, test_ds = load_cifar100()

    @jit
    def batch_logits(params, images_u8):
      images_f32 = vmap(stuff["normalize_transform"])(None, images_u8)
      return model.apply({"params": params}, images_f32)

    batch_size = 500

    def dataset_logits(params, dataset):
      num_examples = dataset["images_u8"].shape[0]
      assert num_examples % batch_size == 0
      num_batches = num_examples // batch_size
      batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
      # Can't use vmap or run in a single batch since that overloads GPU memory.
      return jnp.concatenate([
          batch_logits(params, dataset["images_u8"][batch_ix[i, :], ...])
          for i in tqdm(range(num_batches))
      ],
                             axis=0)

    train_logits_a = dataset_logits(model_a, train_ds)
    train_logits_b = dataset_logits(model_b, train_ds)
    test_logits_a = dataset_logits(model_a, test_ds)
    test_logits_b = dataset_logits(model_b, test_ds)

    lambdas = jnp.linspace(0, 1, num=25)
    train_loss_interp = jnp.array([
        jnp.mean(
            optax.softmax_cross_entropy(logits=lerp(lam, train_logits_a, train_logits_b),
                                        labels=jax.nn.one_hot(train_ds["labels"], NUM_CLASSES)))
        for lam in lambdas
    ])
    test_loss_interp = jnp.array([
        jnp.mean(
            optax.softmax_cross_entropy(logits=lerp(lam, test_logits_a, test_logits_b),
                                        labels=jax.nn.one_hot(test_ds["labels"], NUM_CLASSES)))
        for lam in lambdas
    ])

    train_acc_interp = jnp.array([
        jnp.sum(
            jnp.argmax(lerp(lam, train_logits_a, train_logits_b), axis=-1) == train_ds["labels"])
        for lam in lambdas
    ]) / train_ds["labels"].shape[0]
    test_acc_interp = jnp.array([
        jnp.sum(jnp.argmax(lerp(lam, test_logits_a, test_logits_b), axis=-1) == test_ds["labels"])
        for lam in lambdas
    ]) / test_ds["labels"].shape[0]

    wandb_run.log({
        "train_loss_interp": train_loss_interp,
        "test_loss_interp": test_loss_interp,
        "train_acc_interp": train_acc_interp,
        "test_acc_interp": test_acc_interp
    })

    fig = plt.figure()
    plt.plot(lambdas, train_loss_interp, label="train")
    plt.plot(lambdas, test_loss_interp, label="test")
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylabel("Loss")
    plt.title("Ensembling Loss")
    wandb_run.log({"loss_interp": wandb.Image(fig)})

    fig = plt.figure()
    plt.plot(lambdas, train_acc_interp, label="train")
    plt.plot(lambdas, test_acc_interp, label="test")
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Ensembling Accuracy")
    wandb_run.log({"acc_interp": wandb.Image(fig)})
