import numpy as np
import tensorflow_datasets as tfds

def load_cifar10():
  """Return the training and test datasets, as jnp.array's."""
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("cifar10", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("cifar10", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def load_cifar100():
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("cifar100", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("cifar100", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def _split_cifar(train_ds, label_split: int):
  """Split a CIFAR-ish dataset into two biased subsets."""
  assert train_ds["images_u8"].shape[0] == 50_000
  assert train_ds["labels"].shape[0] == 50_000

  # We randomly permute the training data, just in case there's some kind of
  # non-iid ordering coming out of tfds.
  perm = np.random.default_rng(123).permutation(50_000)
  train_images_u8 = train_ds["images_u8"][perm, :, :, :]
  train_labels = train_ds["labels"][perm]

  # This just so happens to be a clean 25000/25000 split.
  lt_images_u8 = train_images_u8[train_labels < label_split]
  lt_labels = train_labels[train_labels < label_split]
  gte_images_u8 = train_images_u8[train_labels >= label_split]
  gte_labels = train_labels[train_labels >= label_split]
  s1 = {
      "images_u8": np.concatenate((lt_images_u8[:5000], gte_images_u8[5000:]), axis=0),
      "labels": np.concatenate((lt_labels[:5000], gte_labels[5000:]), axis=0)
  }
  s2 = {
      "images_u8": np.concatenate((gte_images_u8[:5000], lt_images_u8[5000:]), axis=0),
      "labels": np.concatenate((gte_labels[:5000], lt_labels[5000:]), axis=0)
  }
  return s1, s2

def load_cifar10_split():
  train_ds, test_ds = load_cifar10()
  s1, s2 = _split_cifar(train_ds, label_split=5)
  return s1, s2, test_ds

def load_cifar100_split():
  train_ds, test_ds = load_cifar100()
  s1, s2 = _split_cifar(train_ds, label_split=50)
  return s1, s2, test_ds
