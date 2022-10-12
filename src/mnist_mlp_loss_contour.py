from pathlib import Path

import jax.numpy as jnp
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.serialization import from_bytes
from jax import random
from jax.flatten_util import ravel_pytree
from matplotlib import tri
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.stats import qmc
from tqdm import tqdm

import matplotlib_style as _
from mnist_mlp_train import MLPModel, load_datasets, make_stuff
from utils import (ec2_get_instance_type, flatten_params, lerp, timeblock,
                   unflatten_params)
from weight_matching import (apply_permutation, mlp_permutation_spec,
                             weight_matching)

matplotlib.rcParams["text.usetex"] = True

with wandb.init(
    project="git-re-basin",
    entity="skainswo",
    tags=["mnist", "mlp", "weight-matching", "loss-contour"],
    job_type="analysis",
) as wandb_run:
  # api = wandb.Api()
  # seed0_run = api.run("skainswo/git-re-basin/1b1gztfx")
  # seed1_run = api.run("skainswo/git-re-basin/1hrmw7wr")

  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.epoch = 99
  config.seed = 123
  config.num_eval_points = 2048

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
  interpolator = tri.LinearTriInterpolator(triang, jnp.minimum(0.55, eval_results[:, 0]))
  # interpolator = tri.LinearTriInterpolator(triang, jnp.log(jnp.minimum(1.5, eval_results[:, 0])))
  zi = interpolator(*np.meshgrid(xi, yi))

  plt.figure()
  num_levels = 13
  plt.contour(xi, yi, zi, levels=num_levels, linewidths=0.25, colors="grey", alpha=0.5)
  # cmap_name = "RdGy"
  # cmap_name = "RdYlBu"
  # cmap_name = "Spectral"
  cmap_name = "coolwarm_r"

  # cmap_name = "YlOrBr_r"
  # cmap_name = "RdBu"

  # See https://stackoverflow.com/a/18926541/3880977
  def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

  cmap = truncate_colormap(plt.get_cmap(cmap_name), 0.0, 0.9)
  plt.contourf(xi, yi, zi, levels=num_levels, cmap=cmap, extend="both")

  x, y = project2d(model_a_flat)
  plt.scatter([x], [y], marker="x", color="white", zorder=10)

  x, y = project2d(model_b_flat)
  plt.scatter([x], [y], marker="x", color="white", zorder=10)

  x, y = project2d(model_b_rebasin_flat)
  plt.scatter([x], [y], marker="x", color="white", zorder=10)

  label_bboxes = dict(facecolor="tab:grey", boxstyle="round", edgecolor="none", alpha=0.5)
  plt.text(-0.075,
           -0.1,
           r"${\bf \Theta_A}$",
           color="white",
           fontsize=24,
           bbox=label_bboxes,
           horizontalalignment="right",
           verticalalignment="top")
  plt.text(1.075,
           -0.1,
           r"${\bf \Theta_B}$",
           color="white",
           fontsize=24,
           bbox=label_bboxes,
           horizontalalignment="left",
           verticalalignment="top")
  x, y = project2d(model_b_rebasin_flat)
  plt.text(x - 0.075,
           y + 0.1,
           r"${\bf \pi(\Theta_B)}$",
           color="white",
           fontsize=24,
           bbox=label_bboxes,
           horizontalalignment="right",
           verticalalignment="bottom")

  # https://github.com/matplotlib/matplotlib/issues/17284#issuecomment-772820638
  # Draw line only
  connectionstyle = "arc3,rad=-0.3"
  plt.annotate("",
               xy=(1, 0),
               xytext=(x, y),
               arrowprops=dict(arrowstyle="-",
                               edgecolor="white",
                               facecolor="none",
                               linewidth=5,
                               linestyle=(0, (5, 3)),
                               shrinkA=20,
                               shrinkB=15,
                               connectionstyle=connectionstyle))
  # Draw arrow head only
  plt.annotate("",
               xy=(1, 0),
               xytext=(x, y),
               arrowprops=dict(arrowstyle="<|-",
                               edgecolor="none",
                               facecolor="white",
                               mutation_scale=40,
                               linewidth=0,
                               shrinkA=12.5,
                               shrinkB=15,
                               connectionstyle=connectionstyle))

  plt.annotate("",
               xy=(0, 0),
               xytext=(x, y),
               arrowprops=dict(arrowstyle="-",
                               edgecolor="white",
                               alpha=0.5,
                               facecolor="none",
                               linewidth=2,
                               linestyle="-",
                               shrinkA=10,
                               shrinkB=10))
  plt.annotate("",
               xy=(0, 0),
               xytext=(1, 0),
               arrowprops=dict(arrowstyle="-",
                               edgecolor="white",
                               alpha=0.5,
                               facecolor="none",
                               linewidth=2,
                               linestyle="-",
                               shrinkA=10,
                               shrinkB=10))

  plt.gca().add_artist(
      AnnotationBbox(OffsetImage(plt.imread(
          "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/check-mark-button_2705.png"
      ),
                                 zoom=0.1), (x / 2, y / 2),
                     frameon=False))
  plt.gca().add_artist(
      AnnotationBbox(OffsetImage(plt.imread(
          "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/cross-mark_274c.png"
      ),
                                 zoom=0.1), (0.5, 0),
                     frameon=False))

  # "Git Re-Basin" box
  #   box_x = 0.5 * (arrow_start[0] + arrow_stop[0])
  #   box_y = 0.5 * (arrow_start[1] + arrow_stop[1])
  # box_x = 0.5 * (arrow_start[0] + arrow_stop[0]) + 0.325
  # box_y = 0.5 * (arrow_start[1] + arrow_stop[1]) + 0.2

  box_x = 0.5
  box_y = 1.3
  # git_rebasin_text = r"\textsc{Git Re-Basin}"
  git_rebasin_text = r"\textbf{Git Re-Basin}"
  # git_rebasin_text = r"\texttt{\textdollar{} git re-basin}"

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
  #   plt.xlim(-0.9, 1.9)
  #   plt.ylim(-0.9, 1.9)
  plt.xticks([])
  plt.yticks([])
  plt.tight_layout()
  plt.savefig("figs/mnist_mlp_loss_contour.png", dpi=300)
  plt.savefig("figs/mnist_mlp_loss_contour.pdf")
