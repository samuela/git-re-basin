import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

import matplotlib_style as _

api = wandb.Api()
# run = api.run("skainswo/git-re-basin/begnvj15")
artifact = Path(
    api.artifact("skainswo/git-re-basin/mnist_permutation_eval_vs_epoch:v1").download())

with open(artifact / "permutation_eval_vs_epoch.pkl", "rb") as f:
  interp_eval_vs_epoch = pickle.load(f)

lambdas = np.linspace(0, 1, 25)

grey_color = "black"
highlight_color = "tab:orange"

# for epoch in tqdm(range(10)):
for epoch in tqdm(range(len(interp_eval_vs_epoch))):
  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1, 2, 1)

  # Naive
  ax1.plot(
      lambdas,
      np.array(interp_eval_vs_epoch[epoch]["naive_train_loss_interp"]),
      color=grey_color,
      linewidth=2,
      #   label="Naïve",
      label="Before",
  )
  ax1.plot(lambdas,
           np.array(interp_eval_vs_epoch[epoch]["naive_test_loss_interp"]),
           color=grey_color,
           linewidth=2,
           linestyle="dashed")

  ax1.plot(
      lambdas,
      np.array(interp_eval_vs_epoch[epoch]["clever_train_loss_interp"]),
      color=highlight_color,
      #    marker="^",
      linewidth=5,
      label="After (ours)")
  ax1.plot(
      lambdas,
      np.array(interp_eval_vs_epoch[epoch]["clever_test_loss_interp"]),
      color=highlight_color,
      #    marker="^",
      linestyle="dashed",
      linewidth=5)

  # ax1.set_ylim(-0.05, 1.6)

  #   ax1.set_xlabel("$\lambda$")
  ax1.set_xticks([0, 1])
  ax1.set_xticklabels(["Model $A$", "Model $B$"])
  ax1.set_ylabel("Loss", labelpad=7.5, fontsize=20)
  # ax1.set_title("Loss")
  ax1.legend(loc="upper right", framealpha=0.5)

  # Accuracy
  ax2 = fig.add_subplot(1, 2, 2)

  ax2.plot(lambdas,
           np.array(interp_eval_vs_epoch[epoch]["naive_train_acc_interp"]),
           color=grey_color,
           linewidth=2,
           label="Train")
  ax2.plot(lambdas,
           np.array(interp_eval_vs_epoch[epoch]["naive_test_acc_interp"]),
           color=grey_color,
           linewidth=2,
           linestyle="dashed",
           label="Test")

  ax2.plot(
      lambdas,
      np.array(interp_eval_vs_epoch[epoch]["clever_train_acc_interp"]),
      color=highlight_color,
      #   marker="^",
      linewidth=5,
      #  label="Ours",
  )
  ax2.plot(
      lambdas,
      np.array(interp_eval_vs_epoch[epoch]["clever_test_acc_interp"]),
      color=highlight_color,
      #    marker="^",
      linestyle="dashed",
      linewidth=5)

  # Prevent this from changing from frame to frame, messing up the spacing
  # See https://stackoverflow.com/questions/29188757/matplotlib-specify-format-of-floats-for-tick-labels
  ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

  ax2.yaxis.tick_right()
  ax2.yaxis.set_label_position("right")

  # ax2.set_ylim(0.8, 1.01)

  #   ax2.set_xlabel("$\lambda$")
  ax2.set_xticks([0, 1])
  ax2.set_xticklabels(["Model $A$", "Model $B$"])

  # 0.7, 0.725, ..., 0.975, 1.0
  allowed_ticks = 1.0 - np.arange(13)[::-1] * 0.025

  # For some reason the first/last ticks reported here are actually invisible...
  actual_ticks = ax2.get_yticks()[1:-1]
  ax2.set_yticks(
      [x for x in allowed_ticks if min(actual_ticks) - 1e-3 <= x <= max(actual_ticks) + 1e-3])

  ax2.set_ylabel("Accuracy", rotation=270, labelpad=30, fontsize=20)
  # ax2.set_title("Accuracy")
  ax2.legend(loc="lower right", framealpha=0.5)

  fig.suptitle(f"Merging NNs before/after permuting weights (epoch {epoch+1})")
  # fig.tight_layout()

  plt.savefig(f"tmp/mnist_video_{epoch:05d}.png", dpi=300)
  plt.close(fig)

subprocess.run([
    "ffmpeg", "-r", "10", "-i", "tmp/mnist_video_%05d.png", "-vcodec", "libx264", "-crf", "15",
    "-pix_fmt", "yuv420p", "-y", "mnist_video.mp4"
],
               check=True)
subprocess.run([
    "ffmpeg", "-f", "image2", "-r", "10", "-i", "tmp/mnist_video_%05d.png", "-loop", "0", "-y",
    "mnist_video.gif"
],
               check=True)
