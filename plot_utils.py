def loss_barrier_is_nonnegative(ax):
  ax.axhline(y=0, color="tab:grey", linestyle=":", alpha=0, zorder=-1)
  ylim = ax.get_ylim()
  # See https://stackoverflow.com/a/5197426/3880977
  ax.axhspan(-0.1, 0, color="tab:grey", alpha=0.25, zorder=-2)
  ax.axhspan(-0.1, 0, facecolor="none", edgecolor="tab:grey", alpha=0.25, hatch="//", zorder=-1)
  ax.set_ylim(ylim)
