from pathlib import Path

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.serialization import from_bytes
from jax import random
from jax.flatten_util import ravel_pytree
from matplotlib import tri
from scipy.stats import qmc
from tqdm import tqdm

import matplotlib_style as _
from mnist_mlp_train import MLPModel, load_datasets, make_stuff
from utils import (ec2_get_instance_type, flatten_params, lerp, timeblock, unflatten_params)
from weight_matching import (apply_permutation, mlp_permutation_spec, weight_matching)

matplotlib.rcParams["text.usetex"] = True

with wandb.init(
    project="playing-the-lottery",
    entity="skainswo",
    tags=["mnist", "mlp", "weight-matching", "barrier-vs-epoch"],
    job_type="analysis",
) as wandb_run:
  # api = wandb.Api()
  # seed0_run = api.run("skainswo/playing-the-lottery/1b1gztfx")
  # seed1_run = api.run("skainswo/playing-the-lottery/1hrmw7wr")

  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.epoch = 99
  config.seed = 123
  config.num_eval_points = 1024

  model = MLPModel()
  stuff = make_stuff(model)

  def load_model(filepath):
    with open(filepath, "rb") as fh:
      return from_bytes(
          model.init(random.PRNGKey(0), jnp.zeros((1, 28, 28, 1)))["params"], fh.read())

  model_a = load_model(
      Path(
          wandb_run.use_artifact("mnist-mlp-weights:v15").get_path(
              f"checkpoint{config.epoch}").download()))
  model_b = load_model(
      Path(
          wandb_run.use_artifact("mnist-mlp-weights:v16").get_path(
              f"checkpoint{config.epoch}").download()))

  permutation_spec = mlp_permutation_spec(3)

  with timeblock("weight_matching"):
    permutation = weight_matching(
        random.PRNGKey(config.seed),
        permutation_spec,
        flatten_params(model_a),
        flatten_params(model_b),
    )

  # Eval
  train_ds, test_ds = load_datasets()

  model_b_rebasin = unflatten_params(
      apply_permutation(permutation_spec, permutation, flatten_params(model_b)))

  # We use model_a as the origin

  model_a_flat, unflatten = ravel_pytree(model_a)
  model_b_flat, _ = ravel_pytree(model_b)
  model_b_rebasin_flat, _ = ravel_pytree(model_b_rebasin)

  # project the vector a onto the vector b
  proj = lambda a, b: jnp.dot(a, b) / jnp.dot(b, b) * b
  norm = lambda a: jnp.sqrt(jnp.dot(a, a))
  normalize = lambda a: a / norm(a)

  basis1 = model_b_flat - model_a_flat
  scale = norm(basis1)
  basis1_normed = normalize(basis1)

  a_to_pi_b = model_b_rebasin_flat - model_a_flat
  basis2 = a_to_pi_b - proj(a_to_pi_b, basis1)
  basis2_normed = normalize(basis2)

  project2d = lambda theta: jnp.array(
      [jnp.dot(theta - model_a_flat, basis1_normed),
       jnp.dot(theta - model_a_flat, basis2_normed)]) / scale

  eval_points = qmc.scale(
      qmc.Sobol(d=2, scramble=True, seed=config.seed).random(config.num_eval_points), [-0.5, -0.5],
      [1.5, 1.5])

  def eval_one(xy):
    params = unflatten(model_a_flat + scale * (basis1_normed * xy[0] + basis2_normed * xy[1]))
    return stuff["dataset_loss_and_accuracy"](params, test_ds, 10_000)

  eval_results = jnp.array(list(map(eval_one, tqdm(eval_points))))

  # Create grid values first.
  xi = np.linspace(-0.5, 1.5)
  yi = np.linspace(-0.5, 1.5)

  # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
  triang = tri.Triangulation(eval_points[:, 0], eval_points[:, 1])
  # We need to cap the maximum loss value so that the contouring is not completely saturated by wildly large losses
  interpolator = tri.LinearTriInterpolator(triang, jnp.minimum(4, eval_results[:, 0]))
  zi = interpolator(*np.meshgrid(xi, yi))

  plt.figure()
  plt.contour(xi, yi, zi, levels=20, linewidths=0.25, colors="k")
  # cmap = "RdGy"
  cmap = "RdYlBu_r"
  # cmap ="Spectral_r"
  # cmap = "coolwarm"
  # cmap = "RdBu_r"
  plt.contourf(xi, yi, zi, levels=20, cmap=cmap, extend="both")

  x, y = project2d(model_a_flat)
  plt.scatter([x], [y], marker="x", color="lightgrey", label="model_a")

  x, y = project2d(model_b_flat)
  plt.scatter([x], [y], marker="x", color="lightgrey", label="model_b")

  x, y = project2d(model_b_rebasin_flat)
  plt.scatter([x], [y], marker="x", color="lightgrey", label="model_b_rebasin")

  plt.text(-0.15, -0.15, r"${\bf \Theta_A}$", color="white", fontsize=24)
  plt.text(1.055, -0.15, r"${\bf \Theta_B}$", color="white", fontsize=24)
  x, y = project2d(model_b_rebasin_flat)
  plt.text(x - 0.325, y + 0.075, r"${\bf \pi(\Theta_B)}$", color="white", fontsize=24)

  arrow_start = np.array([1.0, 0.0])
  arrow_stop = np.array([x, y])
  # https://github.com/matplotlib/matplotlib/issues/17284#issuecomment-772820638
  # Draw line only
  plt.annotate("",
               xy=arrow_start,
               xytext=arrow_stop,
               arrowprops=dict(arrowstyle="-",
                               edgecolor="white",
                               facecolor="none",
                               linewidth=3,
                               linestyle="dashed",
                               shrinkA=17.5,
                               shrinkB=15))
  # Draw arrow head only
  plt.annotate("",
               xy=arrow_start,
               xytext=arrow_stop,
               arrowprops=dict(arrowstyle="<|-",
                               edgecolor="none",
                               facecolor="white",
                               mutation_scale=30,
                               linewidth=0,
                               shrinkA=12.5,
                               shrinkB=15))

  # "Git Re-Basin" box
  # box_x = 0.5 * (arrow_start[0] + arrow_stop[0]) + 0.325
  # box_y = 0.5 * (arrow_start[1] + arrow_stop[1]) + 0.2

  box_x = 0.5
  box_y = 1.3
  git_rebasin_text = r"\textbf{Git Re-Basin}"

  # Draw box only
  plt.text(box_x,
           box_y,
           git_rebasin_text,
           color=(0.0, 0.0, 0.0, 0.0),
           fontsize=24,
           horizontalalignment="center",
           verticalalignment="center",
           bbox=dict(boxstyle="round", fc=(1, 1, 1, 1), ec="black", pad=0.4))
  # Draw text only
  plt.text(box_x,
           box_y - 0.0115,
           git_rebasin_text,
           color=(0.0, 0.0, 0.0, 1.0),
           fontsize=24,
           horizontalalignment="center",
           verticalalignment="center")

  # plt.colorbar()
  plt.xlim(-0.4, 1.4)
  plt.ylim(-0.45, 1.3)
  plt.xticks([])
  plt.yticks([])
  plt.tight_layout()
  plt.savefig("figs/mnist_mlp_loss_contour.png", dpi=300)
