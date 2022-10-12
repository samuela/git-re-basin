import argparse
from pathlib import Path

import jax.numpy as jnp
import wandb
from flax.serialization import from_bytes
from jax import random

from cifar100_resnet20_train import NUM_CLASSES, make_stuff
from datasets import load_cifar100
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import ec2_get_instance_type

NUM_CLASSES = 100

# https://wandb.ai/skainswo/git-re-basin/runs/f40w12z7/overview?workspace=user-skainswo
# use model=v11, width multiplier=32, load-epoch=249

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--width-multiplier", type=int, required=True)
parser.add_argument("--load-epoch", type=int, required=True)
args = parser.parse_args()

with wandb.init(
    project="git-re-basin",
    entity="skainswo",
    tags=["cifar100", "resnet20", "top5"],
    job_type="analysis",
) as wandb_run:
  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.model = args.model
  config.width_multiplier = args.width_multiplier
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
  model = load_model(
      Path(
          wandb_run.use_artifact(f"cifar100-resnet-weights:{config.model}").get_path(
              filename).download()))

  train_ds, test_ds = load_cifar100()

  test_loss, test_acc1, test_acc5 = stuff["dataset_loss_and_accuracies"](model, test_ds, 1000)

  print({
      "test_loss": test_loss,
      "test_acc1": test_acc1,
      "test_acc5": test_acc5,
  })
  wandb_run.log({
      "test_loss": test_loss,
      "test_acc1": test_acc1,
      "test_acc5": test_acc5,
  })
