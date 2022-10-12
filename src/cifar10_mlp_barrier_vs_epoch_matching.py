import pickle
from pathlib import Path

import jax.numpy as jnp
import wandb
from flax.serialization import from_bytes
from jax import random
from tqdm import tqdm

from cifar10_mlp_train import MLPModel, make_stuff
from datasets import load_cifar10
from utils import flatten_params, lerp, unflatten_params
from weight_matching import (apply_permutation, mlp_permutation_spec, weight_matching)

with wandb.init(
    project="git-re-basin",
    entity="skainswo",
    tags=["cifar10", "mlp", "weight-matching", "barrier-vs-epoch"],
    job_type="analysis",
) as wandb_run:
  # api = wandb.Api()
  # seed0_run = api.run("skainswo/git-re-basin/1b1gztfx")
  # seed1_run = api.run("skainswo/git-re-basin/1hrmw7wr")

  config = wandb.config
  config.total_epochs = 100
  config.seed = 123

  model = MLPModel()
  stuff = make_stuff(model)

  def load_model(filepath):
    with open(filepath, "rb") as fh:
      return from_bytes(
          model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())

  seed0_artifact = Path(wandb_run.use_artifact("cifar10-mlp-weights:v13").download())
  seed1_artifact = Path(wandb_run.use_artifact("cifar10-mlp-weights:v14").download())

  permutation_spec = mlp_permutation_spec(3)

  def match_one_epoch(epoch: int):
    model_a = load_model(seed0_artifact / f"checkpoint{epoch}")
    model_b = load_model(seed1_artifact / f"checkpoint{epoch}")
    return weight_matching(
        random.PRNGKey(config.seed),
        permutation_spec,
        flatten_params(model_a),
        flatten_params(model_b),
    )

  permutation_vs_epoch = [match_one_epoch(i) for i in tqdm(range(config.total_epochs))]

  artifact = wandb.Artifact("cifar10_permutation_vs_epoch",
                            type="permutation_vs_epoch",
                            metadata={
                                "dataset": "cifar10",
                                "model": "mlp",
                                "analysis": "weight-matching"
                            })
  with artifact.new_file("permutation_vs_epoch.pkl", mode="wb") as f:
    pickle.dump(permutation_vs_epoch, f)
  wandb_run.log_artifact(artifact)

  # Eval
  train_ds, test_ds = load_cifar10()

  def eval_one(epoch, permutation):
    model_a = load_model(seed0_artifact / f"checkpoint{epoch}")
    model_b = load_model(seed1_artifact / f"checkpoint{epoch}")

    lambdas = jnp.linspace(0, 1, num=25)

    model_b_perm = unflatten_params(
        apply_permutation(permutation_spec, permutation, flatten_params(model_b)))

    train_loss_interp = []
    test_loss_interp = []
    train_acc_interp = []
    test_acc_interp = []
    for lam in lambdas:
      clever_p = lerp(lam, model_a, model_b_perm)
      train_loss, train_acc = stuff["dataset_loss_and_accuracy"](clever_p, train_ds, 10_000)
      test_loss, test_acc = stuff["dataset_loss_and_accuracy"](clever_p, test_ds, 10_000)
      train_loss_interp.append(train_loss)
      test_loss_interp.append(test_loss)
      train_acc_interp.append(train_acc)
      test_acc_interp.append(test_acc)

    return {
        "train_loss_interp": train_loss_interp,
        "test_loss_interp": test_loss_interp,
        "train_acc_interp": train_acc_interp,
        "test_acc_interp": test_acc_interp,
    }

  interp_eval_vs_epoch = [eval_one(i, p) for i, p in tqdm(enumerate(permutation_vs_epoch))]

  artifact = wandb.Artifact("cifar10_permutation_eval_vs_epoch",
                            type="permutation_eval_vs_epoch",
                            metadata={
                                "dataset": "cifar10",
                                "model": "mlp",
                                "analysis": "weight-matching",
                                "interpolation": "lerp"
                            })
  with artifact.new_file("permutation_eval_vs_epoch.pkl", mode="wb") as f:
    pickle.dump(interp_eval_vs_epoch, f)
  wandb_run.log_artifact(artifact)
