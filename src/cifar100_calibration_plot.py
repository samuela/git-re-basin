import pickle

import matplotlib.pyplot as plt
import numpy as np
from jax import nn
from tqdm import tqdm

import matplotlib_style as _

# See https://github.com/google/jax/issues/696#issuecomment-642457347
# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

NUM_CLASSES = 100

data = pickle.load(open("../cifar100_interp_logits.pkl", "rb"))

num_bins = 15
bins = np.linspace(0, 1, num_bins + 1)
bin_locations = 0.5 * (bins[:-1] + bins[1:])

def one(bin_ix, probs, labels):
  lo, hi = bins[bin_ix], bins[bin_ix + 1]
  mask = (lo <= probs) & (probs <= hi)
  y_onehot = nn.one_hot(labels, NUM_CLASSES)
  return np.mean(y_onehot[mask])

### Plotting
plt.figure(figsize=(12, 6))

# Train
plt.subplot(1, 2, 1)
plotting_ds_name = "train"
plotting_ds = data[f"{plotting_ds_name}_dataset"]

a_probs = nn.softmax(data[f"a_{plotting_ds_name}_logits"])
b_probs = nn.softmax(data[f"b_{plotting_ds_name}_logits"])
clever_probs = nn.softmax(data[f"clever_{plotting_ds_name}_logits"])
naive_probs = nn.softmax(data[f"naive_{plotting_ds_name}_logits"])
ensemble_probs = nn.softmax(0.5 * (data[f"a_{plotting_ds_name}_logits"] + data[f"b_{plotting_ds_name}_logits"]))

model_a_ys = [one(ix, a_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
model_b_ys = [one(ix, b_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
wm_ys = [one(ix, clever_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
naive_ys = [one(ix, naive_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
ensemble_ys = [one(ix, ensemble_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")
plt.plot(bin_locations, model_a_ys, alpha=0.5, label="Model A")
plt.plot(bin_locations, model_b_ys, alpha=0.5, label="Model B")
plt.plot(bin_locations, naive_ys, color="tab:grey", marker=".", label="Naïve merging")
plt.plot(bin_locations, ensemble_ys, color="tab:purple", marker="2", label="Model ensemble")
plt.plot(bin_locations, wm_ys, color="tab:green", marker="^", linewidth=2, label="Weight matching")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.axis("equal")
plt.legend()
plt.title("Train")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.linspace(0, 1, 5))
plt.yticks(np.linspace(0, 1, 5))

# Test
plt.subplot(1, 2, 2)
plotting_ds_name = "test"
plotting_ds = data[f"{plotting_ds_name}_dataset"]

a_probs = nn.softmax(data[f"a_{plotting_ds_name}_logits"])
b_probs = nn.softmax(data[f"b_{plotting_ds_name}_logits"])
clever_probs = nn.softmax(data[f"clever_{plotting_ds_name}_logits"])
naive_probs = nn.softmax(data[f"naive_{plotting_ds_name}_logits"])
ensemble_probs = nn.softmax(0.5 * (data[f"a_{plotting_ds_name}_logits"] + data[f"b_{plotting_ds_name}_logits"]))

model_a_ys = [one(ix, a_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
model_b_ys = [one(ix, b_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
wm_ys = [one(ix, clever_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
naive_ys = [one(ix, naive_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
ensemble_ys = [one(ix, ensemble_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")
plt.plot(bin_locations, model_a_ys, alpha=0.5, linestyle="dashed", label="Model A")
plt.plot(bin_locations, model_b_ys, alpha=0.5, linestyle="dashed", label="Model B")
plt.plot(bin_locations,
         naive_ys,
         color="tab:grey",
         marker=".",
         linestyle="dashed",
         label="Naïve merging")
plt.plot(bin_locations,
         ensemble_ys,
         color="tab:purple",
         marker="2",
         linestyle="dashed",
         label="Model ensemble")
plt.plot(bin_locations,
         wm_ys,
         color="tab:green",
         marker="^",
         linewidth=2,
         linestyle="dashed",
         label="Weight matching")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.axis("equal")
plt.title("Test")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.linspace(0, 1, 5))
plt.yticks(np.linspace(0, 1, 5))

plt.suptitle("CIFAR-100 Split Datasets, Calibration")
plt.tight_layout()
plt.savefig("figs/cifar100_calibration_plot.png", dpi=300)
plt.savefig("figs/cifar100_calibration_plot.pdf")
