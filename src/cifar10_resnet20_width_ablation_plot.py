import matplotlib.pyplot as plt
import numpy as np
import wandb

import matplotlib_style as _
from plot_utils import loss_barrier_is_nonnegative

if __name__ == "__main__":
  api = wandb.Api()
  wm1_run = api.run("skainswo/git-re-basin/61rq8cq7")
  wm2_run = api.run("skainswo/git-re-basin/2xer5nly")
  wm4_run = api.run("skainswo/git-re-basin/3gtdjmnf")
  wm8_run = api.run("skainswo/git-re-basin/3gq93t7g")
  wm16_run = api.run("skainswo/git-re-basin/2zeq7qy2")
  wm32_run = api.run("skainswo/git-re-basin/223t7txl")

  all_runs = [wm1_run, wm2_run, wm4_run, wm8_run, wm16_run, wm32_run]

  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # lambdas = np.linspace(0, 1, 25)
  # wm_glyphs = ["⅛", "¼", "½", "1", "2", "4"]
  # cmap = plt.get_cmap("YlOrRd")
  # for i, wm_glyph, run in zip(range(len(all_runs)), wm_glyphs, all_runs):
  #   ys = np.array(run.summary["train_loss_interp_clever"])
  #   ys = ys - 0.5 * (ys[0] + ys[-1])
  #   ax.plot(lambdas,
  #           ys,
  #           color=cmap(0.25 + 0.75 * i / len(all_runs)),
  #           linewidth=2,
  #           label=f"{wm_glyph}× width")
  # ax.set_xlabel("$\lambda$")
  # ax.set_xticks([0, 1])
  # ax.set_xticklabels(["Model $A$", "Model $B$"])
  # ax.set_ylabel("Training loss barrier")
  # ax.set_title(f"CIFAR-10 ResNet Width Ablation")
  # ax.legend(loc="upper right", framealpha=0.5)
  # fig.tight_layout()

  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_train_loss.png", dpi=300)
  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_train_loss.pdf")
  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_train_loss.eps")

  # ############
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # lambdas = np.linspace(0, 1, 25)
  # wm_glyphs = ["⅛", "¼", "½", "1", "2", "4"]
  # cmap = plt.get_cmap("YlOrRd")
  # for i, wm_glyph, run in zip(range(len(all_runs)), wm_glyphs, all_runs):
  #   ys = np.array(run.summary["test_loss_interp_clever"])
  #   ys = ys - 0.5 * (ys[0] + ys[-1])
  #   ax.plot(lambdas,
  #           ys,
  #           color=cmap(0.25 + 0.75 * i / len(all_runs)),
  #           linewidth=2,
  #           label=f"{wm_glyph}× width")
  # ax.set_xlabel("$\lambda$")
  # ax.set_xticks([0, 1])
  # ax.set_xticklabels(["Model $A$", "Model $B$"])
  # ax.set_ylabel("Test loss barrier")
  # ax.set_title(f"CIFAR-10 ResNet Width Ablation")
  # ax.legend(loc="upper right", framealpha=0.5)
  # fig.tight_layout()

  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_test_loss.png", dpi=300)
  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_test_loss.pdf")
  # plt.savefig("figs/cifar10_resnet20_width_ablation_plot_test_loss.eps")

  ###
  fig = plt.figure()
  #   fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  lambdas = np.linspace(0, 1, 25)
  wm_glyphs = ["1", "2", "4", "8", "16", "32"]

  train_barriers = [
      max(run.summary["train_loss_interp_clever"]) - 0.5 *
      (run.summary["train_loss_interp_clever"][0] + run.summary["train_loss_interp_clever"][-1])
      for run in all_runs
  ]
  test_barriers = [
      max(run.summary["test_loss_interp_clever"]) - 0.5 *
      (run.summary["test_loss_interp_clever"][0] + run.summary["test_loss_interp_clever"][-1])
      for run in all_runs
  ]

  ax.plot(
      train_barriers,
      marker="o",
      linewidth=2,
      label="Train",
  )
  ax.plot(
      test_barriers,
      marker="^",
      linestyle="dashed",
      linewidth=2,
      label="Test",
  )

  ax.arrow(5, 0, -0.75, 0.5, alpha=0.25)
  ins2 = ax.inset_axes((0.7, 0.3, 0.25, 0.25))
  ins2.plot(all_runs[-1].summary["train_loss_interp_clever"])
  ins2.plot(all_runs[-1].summary["test_loss_interp_clever"], linestyle="dashed")
  ins2.set_xticks([])
  ins2.set_yticks([])
  ymin, ymax = ins2.get_ylim()
  ins2.set_ylim((ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))

  loss_barrier_is_nonnegative(ax)

  ax.set_xlabel("Width multiplier")
  ax.set_xticks(range(len(all_runs)))
  ax.set_xticklabels([f"{x}×" for x in wm_glyphs])
  ax.set_ylabel("Loss barrier")
  ax.set_title("ResNet20")
  ax.legend(loc="upper right", framealpha=0.5)
  fig.tight_layout()

  plt.savefig("figs/cifar10_resnet20_width_ablation_line_plot.png", dpi=300)
  plt.savefig("figs/cifar10_resnet20_width_ablation_line_plot.pdf")
