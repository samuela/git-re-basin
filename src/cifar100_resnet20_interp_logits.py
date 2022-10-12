import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import wandb
from flax.serialization import from_bytes
from jax import random

from cifar100_resnet20_train import make_stuff
from datasets import load_cifar100
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import apply_permutation, resnet20_permutation_spec

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--permutation", type=str, required=True)
  parser.add_argument("--width-multiplier", type=int, required=True)
  parser.add_argument("--load-epoch", type=int, required=True)
  args = parser.parse_args()

  with wandb.init(
      project="git-re-basin",
      entity="skainswo",
      tags=["cifar100", "resnet20", "logits"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.permutation = args.permutation
    config.width_multiplier = args.width_multiplier
    config.load_epoch = args.load_epoch

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=100,
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

    permutation_spec = resnet20_permutation_spec()
    final_permutation = pickle.load(
        Path(
            wandb_run.use_artifact(f"model_b_permutation:{config.permutation}").get_path(
                "permutation.pkl").download()).open("rb"))

    print("model A")
    a_train_logits = stuff["dataset_logits"](model_a, train_ds, 1000)
    a_test_logits = stuff["dataset_logits"](model_a, test_ds, 1000)

    print("model B")
    b_train_logits = stuff["dataset_logits"](model_b, train_ds, 1000)
    b_test_logits = stuff["dataset_logits"](model_b, test_ds, 1000)

    print("naive interpolation")
    naive_interp_p = lerp(0.5, model_a, model_b)
    naive_train_logits = stuff["dataset_logits"](naive_interp_p, train_ds, 1000)
    naive_test_logits = stuff["dataset_logits"](naive_interp_p, test_ds, 1000)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b)))

    print("clever interpolation")
    clever_interp_p = lerp(0.5, model_a, model_b_clever)
    clever_train_logits = stuff["dataset_logits"](clever_interp_p, train_ds, 1000)
    clever_test_logits = stuff["dataset_logits"](clever_interp_p, test_ds, 1000)

    with Path("cifar100_interp_logits.pkl").open("wb") as fh:
      pickle.dump(
          {
              "train_dataset": train_ds,
              "test_dataset": test_ds,
              "a_train_logits": a_train_logits,
              "a_test_logits": a_test_logits,
              "b_train_logits": b_train_logits,
              "b_test_logits": b_test_logits,
              "naive_train_logits": naive_train_logits,
              "naive_test_logits": naive_test_logits,
              "clever_train_logits": clever_train_logits,
              "clever_test_logits": clever_test_logits,
          }, fh)
