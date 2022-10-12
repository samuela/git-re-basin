# Example usage:
#     python parallel_mnist_videos.py skainswo/git-re-basin/2vzg9n1u

import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from parallel_mnist_plots import plot_interp_acc, plot_interp_loss

api = wandb.Api()
run = api.run(sys.argv[1])
history = run.history()

# TODO: this should no longer be necessary...
lambdas = jnp.linspace(0, 1, num=10)

with tempfile.TemporaryDirectory() as tempdir:
  for step in history:
    fig = plot_interp_loss(step["epoch"], lambdas, step["train_loss_interp_naive"],
                           step["test_loss_interp_naive"], step["train_loss_interp_clever"],
                           step["test_loss_interp_clever"])
    plt.savefig(Path(tempdir) / f"{step['epoch']:05d}.png")
    plt.close(fig)

  subprocess.run([
      "ffmpeg", "-r", "10", "-i",
      Path(tempdir) / "%05d.png", "-vcodec", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-y",
      f"parallel_mnist_interp_loss.mp4"
  ],
                 check=True)

with tempfile.TemporaryDirectory() as tempdir:
  for step in history:
    fig = plot_interp_acc(step["epoch"], lambdas, step["train_acc_interp_naive"],
                          step["test_acc_interp_naive"], step["train_acc_interp_clever"],
                          step["test_acc_interp_clever"])
    plt.savefig(Path(tempdir) / f"{step['epoch']:05d}.png")
    plt.close(fig)

  subprocess.run([
      "ffmpeg", "-r", "10", "-i",
      Path(tempdir) / "%05d.png", "-vcodec", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-y",
      f"parallel_mnist_interp_acc.mp4"
  ],
                 check=True)
