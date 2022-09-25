import operator
import re
import time
from contextlib import contextmanager

import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax import random, tree_map
from jax.tree_util import tree_reduce

rngmix = lambda rng, x: random.fold_in(rng, hash(x))

@contextmanager
def timeblock(name):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    print(f"{name} took {end - start:.5f} seconds")

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""

  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key

def l1prox(x, alpha):
  return jnp.sign(x) * jnp.maximum(0, jnp.abs(x) - alpha)

def ec2_get_instance_type():
  # See also https://stackoverflow.com/questions/51486405/aws-ec2-command-line-display-instance-type/51486782
  return open("/sys/devices/virtual/dmi/id/product_name").read().strip()

# Utilities for dealing with flax model parameters
def partition(pred, iterable):
  trues = []
  falses = []
  for item in iterable:
    if pred(item):
      trues.append(item)
    else:
      falses.append(item)
  return trues, falses

def partition_dict(pred, d):
  trues = {}
  falses = {}
  for k, v in d.items():
    if pred(k):
      trues[k] = v
    else:
      falses[k] = v
  return trues, falses

def flatten_params(params):
  return {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}

def unflatten_params(flat_params):
  return freeze(
      traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in flat_params.items()}))

def merge_params(a, b):
  return unflatten_params({**a, **b})

def kmatch(pattern, key):
  regex = "^"
  i = 0
  while i < len(pattern):
    if pattern[i] == "*":
      if i + 1 < len(pattern) and pattern[i + 1] == "*":
        regex += "(.*)"
        i += 2
      else:
        regex += "([^\/]*)"
        i += 1
    else:
      regex += pattern[i]
      i += 1
  regex += "$"
  return re.fullmatch(regex, key)

assert kmatch("*", "a") is not None
assert kmatch("*", "a").group(0) == "a"
assert kmatch("*", "a").group(1) == "a"
assert kmatch("abc", "def") is None
assert kmatch("abc/*/ghi", "abc/def/ghi").group(1) == "def"
assert kmatch("abc/**/jkl", "abc/def/ghi/jkl").group(1) == "def/ghi"
assert kmatch("abc/*/jkl", "abc/def/ghi/jkl") is None
assert kmatch("**/*", "abc/def/ghi/jkl").group(1) == "abc/def/ghi"
assert kmatch("**/*", "abc/def/ghi/jkl").group(2) == "jkl"

def lerp(lam, t1, t2):
  return tree_map(lambda a, b: (1 - lam) * a + lam * b, t1, t2)

def tree_norm(t):
  return jnp.sqrt(tree_reduce(operator.add, tree_map(lambda x: jnp.sum(x**2), t)))

def tree_l2(t1, t2):
  return tree_norm(tree_map(lambda x, y: x - y, t1, t2))

def slerp(lam, t1, t2):
  # See https://en.wikipedia.org/wiki/Slerp
  om = jnp.arccos(
      tree_reduce(operator.add, tree_map(lambda x, y: jnp.sum(x * y), t1, t2)) /
      (tree_norm(t1) * tree_norm(t2)))
  sinom = jnp.sin(om)
  return tree_map(
      lambda x, y: jnp.sin((1 - lam) * om) / sinom * x + jnp.sin(lam * om) / sinom * y,
      t1,
      t2,
  )
