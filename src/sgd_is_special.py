import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
from jax import random, tree_map
from matplotlib.colors import ListedColormap

import matplotlib_style as _
from utils import unflatten_params

rng = random.PRNGKey(0)

class Model(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2)(x)
    x = nn.PReLU()(x)
    x = nn.Dense(2)(x)
    x = nn.PReLU()(x)
    x = nn.Dense(1)(x)
    return x

model = Model()

# dense kernel shape: (in, out)
dtype = jnp.float32
paramsA = {
    "PReLU_0/negative_slope": jnp.array(0.01),
    "PReLU_1/negative_slope": jnp.array(0.01),
    "Dense_0/kernel": jnp.array([[-1, 0], [0, -1]], dtype=dtype),
    "Dense_0/bias": jnp.array([1, 0], dtype=dtype),
    "Dense_1/kernel": jnp.array([[-1, 0], [0, 1]], dtype=dtype),
    "Dense_1/bias": jnp.array([1, 0], dtype=dtype),
    "Dense_2/kernel": jnp.array([[-1], [-1]], dtype=dtype),
    "Dense_2/bias": jnp.array([0], dtype=dtype),
}
paramsB1 = {
    "PReLU_0/negative_slope": jnp.array(0.01),
    "PReLU_1/negative_slope": jnp.array(0.01),
    "Dense_0/kernel": jnp.array([[1, 0], [0, 1]], dtype=dtype),
    "Dense_0/bias": jnp.array([0, 1], dtype=dtype),
    "Dense_1/kernel": jnp.array([[1, 0], [0, -1]], dtype=dtype),
    "Dense_1/bias": jnp.array([0, 1], dtype=dtype),
    "Dense_2/kernel": jnp.array([[-1], [-1]], dtype=dtype),
    "Dense_2/bias": jnp.array([0], dtype=dtype),
}

def swap_layer(layer: int, params):
  ix = jnp.array([1, 0])
  return {
      **params,
      f"Dense_{layer}/kernel": params[f"Dense_{layer}/kernel"][:, ix],
      f"Dense_{layer}/bias": params[f"Dense_{layer}/bias"][ix],
      f"Dense_{layer+1}/kernel": params[f"Dense_{layer+1}/kernel"][ix, :],
  }

swap_first_layer = lambda params: swap_layer(0, params)
swap_second_layer = lambda params: swap_layer(1, params)

paramsB2 = swap_first_layer(paramsB1)
paramsB3 = swap_second_layer(paramsB1)
paramsB4 = swap_first_layer(swap_second_layer(paramsB1))

# Assert that [swapfirst, swapsecond] is the same as [swapsecond, swapfirst].
assert jnp.all(
    jnp.array(
        list(
            tree_map(jnp.allclose, swap_first_layer(swap_second_layer(paramsB1)),
                     swap_second_layer(swap_first_layer(paramsB1))).values())))

num_examples = 1024
testX = random.uniform(rng, (num_examples, 2), dtype=dtype, minval=-1, maxval=1)
testY = (testX[:, 0] <= 0) & (testX[:, 1] >= 0)

def accuracy(params):
  return jnp.sum((model.apply({"params": unflatten_params(params)}, testX) >= 0
                  ).flatten() == testY) / num_examples

# assert accuracy(paramsA) == 1.0
# assert accuracy(paramsB1) == 1.0
# assert accuracy(paramsB2) == 1.0
# assert accuracy(paramsB3) == 1.0
# assert accuracy(paramsB4) == 1.0

def interp_params(lam, pA, pB):
  return tree_map(lambda a, b: lam * a + (1 - lam) * b, pA, pB)

def plot_interp_loss():
  lambdas = jnp.linspace(0, 1, num=10)
  interp1 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB1)) for lam in lambdas])
  interp2 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB2)) for lam in lambdas])
  interp3 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB3)) for lam in lambdas])
  interp4 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB4)) for lam in lambdas])

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  # We make losses start at 0, since that intuitively makes more sense.
  ax.plot(lambdas, -interp1 + 1, linewidth=2, marker="o", label="Identity")
  ax.plot(lambdas, -interp2 + 1, linewidth=2, marker=".", label="Swap layer 1")
  ax.plot(lambdas, -interp3 + 1, linewidth=2, marker="x", label="Swap layer 2")
  ax.plot(lambdas, -interp4 + 1, linewidth=2, marker="*", label="Swap both")
  ax.plot([-1, 2], [0, 0], linestyle=":", color="tab:grey", alpha=0.5, label="Perfect performance")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_xlim(-0.05, 1.05)
  ax.set_ylabel("Loss")
  ax.set_title("All possible permutations")
  ax.legend(framealpha=0.5)
  fig.tight_layout()
  return fig

fig = plot_interp_loss()
plt.savefig(f"figs/sgd_is_special_loss_interp.png", dpi=300)
plt.savefig(f"figs/sgd_is_special_loss_interp.eps")
plt.savefig(f"figs/sgd_is_special_loss_interp.pdf")
plt.close(fig)

def plot_interp_loss_zoom(max_lambda):
  lambdas = jnp.linspace(0, max_lambda, num=10)
  interp1 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB1)) for lam in lambdas])
  interp2 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB2)) for lam in lambdas])
  interp3 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB3)) for lam in lambdas])
  interp4 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB4)) for lam in lambdas])

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  # We make losses start at 0, since that intuitively makes more sense.
  ax.plot(lambdas, -interp1 + 1, linewidth=2, marker="o", label="Identity")
  ax.plot(lambdas, -interp2 + 1, linewidth=2, marker=".", label="Swap layer 1")
  ax.plot(lambdas, -interp3 + 1, linewidth=2, marker="x", label="Swap layer 2")
  ax.plot(lambdas, -interp4 + 1, linewidth=2, marker="*", label="Swap both")
  ax.plot([-1, 2], [0, 0],
          linestyle="dashed",
          color="tab:grey",
          alpha=0.5,
          label="Perfect performance")

  # ax.set_xscale("log")
  # ax.set_yscale("log")

  ax.set_xlabel("$\lambda$")
  # ax.set_xlim(-0.05, max_lambda * 1.05)
  ax.set_ylabel("Loss")
  ax.set_title("All possible permutations between two globally optimal models (zoom)")
  ax.legend(framealpha=0.5)
  fig.tight_layout()
  return fig

fig = plot_interp_loss_zoom(max_lambda=1e-6)
plt.savefig(f"figs/sgd_is_special_loss_interp_zoom.png", dpi=300)
# plt.savefig(f"figs/sgd_is_special_loss_interp.pdf")
plt.close(fig)

def plot_data():
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(testX[testY, 0],
             testX[testY, 1],
             edgecolor="tab:green",
             facecolor="none",
             marker="o",
             label="$y=1$")
  ax.scatter(testX[~testY, 0], testX[~testY, 1], color="tab:red", marker="x", label="$y=0$")
  ax.axhline(0, color="tab:grey", alpha=0.25)
  ax.axvline(0, color="tab:grey", alpha=0.25)
  ax.set_xlabel("$x_1$")
  ax.set_ylabel("$x_2$")
  ax.set_title("Data")
  # ax.legend(framealpha=0.5)
  fig.tight_layout()
  return fig

fig = plot_data()
plt.savefig(f"figs/sgd_is_special_data.png", dpi=300)
plt.savefig(f"figs/sgd_is_special_data.eps")
plt.savefig(f"figs/sgd_is_special_data.pdf")
plt.close()

extrema = model.apply({"params": unflatten_params(paramsA)},
                      jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=dtype))
min_score = jnp.min(extrema)
max_score = jnp.max(extrema)

def plot_detailed_view():
  lambdas = jnp.linspace(0, 1, num=9)

  s = 2
  fig, ax = plt.subplots(4, len(lambdas), figsize=(len(lambdas) * s, 4 * s))

  ticks = jnp.linspace(-1, 1, num=100)
  xx1, xx2 = jnp.meshgrid(ticks, ticks)
  meshX = jnp.stack([xx1.flatten(), xx2.flatten()], axis=1)

  def asdf(row, paramsB):
    for i in range(len(lambdas)):
      params = interp_params(lambdas[i], paramsA, paramsB)
      meshY = model.apply({"params": unflatten_params(params)}, meshX).flatten()
      decision_boundary = (meshY >= 0).astype(float)
      ax[row, i].contourf(xx1,
                          xx2,
                          meshY.reshape(xx1.shape),
                          levels=jnp.linspace(min_score, max_score, num=25),
                          cmap="copper")
      # ax[row, i].contourf(xx1,
      #                     xx2,
      #                     decision_boundary.reshape(xx1.shape),
      #                     cmap=ListedColormap(np.array([[0, 0, 0, 0.0], [0, 1, 0, 0.5]])))
      ax[row, i].set_xticks([])
      ax[row, i].set_yticks([])

  asdf(0, paramsB1)
  asdf(1, paramsB2)
  asdf(2, paramsB3)
  asdf(3, paramsB4)

  ax[0, 0].set_title("Model A", fontweight="bold")
  ax[0, -1].set_title("Model B", fontweight="bold")
  ax[0, 4].set_title("⟵ $\\lambda$ ⟶")

  ax[0, 0].set_ylabel("Identity")
  ax[1, 0].set_ylabel("Swap layer 1")
  ax[2, 0].set_ylabel("Swap layer 2")
  ax[3, 0].set_ylabel("Swap both")

  fig.tight_layout()
  return fig

fig = plot_detailed_view()
plt.savefig(f"figs/sgd_is_special_detailed_view.png", dpi=300)
plt.savefig(f"figs/sgd_is_special_detailed_view.eps")
plt.savefig(f"figs/sgd_is_special_detailed_view.pdf")
plt.close(fig)
