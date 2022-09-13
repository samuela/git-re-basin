"""Online-ish Pearson correlation of all n x n variable pairs simultaneously."""
from typing import NamedTuple

import jax.numpy as jnp


class OnlineMean(NamedTuple):
  sum: jnp.ndarray
  count: int

  @staticmethod
  def init(num_features: int):
    return OnlineMean(sum=jnp.zeros(num_features), count=0)

  def update(self, batch: jnp.ndarray):
    return OnlineMean(self.sum + jnp.sum(batch, axis=0), self.count + batch.shape[0])

  def mean(self):
    return self.sum / self.count

class OnlineCovariance(NamedTuple):
  a_mean: jnp.ndarray  # (d, )
  b_mean: jnp.ndarray  # (d, )
  cov: jnp.ndarray  # (d, d)
  var_a: jnp.ndarray  # (d, )
  var_b: jnp.ndarray  # (d, )
  count: int

  @staticmethod
  def init(a_mean: jnp.ndarray, b_mean: jnp.ndarray):
    assert a_mean.shape == b_mean.shape
    assert len(a_mean.shape) == 1
    d = a_mean.shape[0]
    return OnlineCovariance(a_mean,
                            b_mean,
                            cov=jnp.zeros((d, d)),
                            var_a=jnp.zeros((d, )),
                            var_b=jnp.zeros((d, )),
                            count=0)

  def update(self, a_batch, b_batch):
    assert a_batch.shape == b_batch.shape
    batch_size, _ = a_batch.shape
    a_res = a_batch - self.a_mean
    b_res = b_batch - self.b_mean
    return OnlineCovariance(a_mean=self.a_mean,
                            b_mean=self.b_mean,
                            cov=self.cov + a_res.T @ b_res,
                            var_a=self.var_a + jnp.sum(a_res**2, axis=0),
                            var_b=self.var_b + jnp.sum(b_res**2, axis=0),
                            count=self.count + batch_size)

  def covariance(self):
    return self.cov / (self.count - 1)

  def a_variance(self):
    return self.var_a / (self.count - 1)

  def b_variance(self):
    return self.var_b / (self.count - 1)

  def a_stddev(self):
    return jnp.sqrt(self.a_variance())

  def b_stddev(self):
    return jnp.sqrt(self.b_variance())

  def E_ab(self):
    return self.covariance() + jnp.outer(self.a_mean, self.b_mean)

  def pearson_correlation(self):
    # Note that the 1/(n-1) normalization terms cancel out nicely here.
    # TODO: clip?
    eps = 0
    # Dead units will have zero variance, which produces NaNs. Convert those to
    # zeros with nan_to_num.
    return jnp.nan_to_num(self.cov / (jnp.sqrt(self.var_a[:, jnp.newaxis]) + eps) /
                          (jnp.sqrt(self.var_b) + eps))

class OnlineInnerProduct(NamedTuple):
  val: jnp.ndarray  # (d, d)

  @staticmethod
  def init(d: int):
    return OnlineInnerProduct(val=jnp.zeros((d, d)))

  def update(self, a_batch, b_batch):
    assert a_batch.shape == b_batch.shape
    return OnlineInnerProduct(val=self.val + a_batch.T @ b_batch)

# def online_pearson_init_state(n):
#   return {
#       "Exy": jnp.zeros((n, n)),
#       "Ex": jnp.zeros((n, )),
#       "Ey": jnp.zeros((n, )),
#       "Ex2": jnp.zeros((n, )),
#       "Ey2": jnp.zeros((n, )),
#       "samples": 0,
#   }

# def online_pearson_update(state, x_batch, y_batch):
#   """Online-ish Pearson update.

#   x_batch and y_batch are assumed to be of shape (batch_size, n)."""
#   assert x_batch.shape == y_batch.shape
#   batch_size = x_batch.shape[0]
#   return {
#       "Exy": state["Exy"] + x_batch.T @ y_batch,
#       "Ex": state["Ex"] + jnp.sum(x_batch, axis=0),
#       "Ey": state["Ey"] + jnp.sum(y_batch, axis=0),
#       "Ex2": state["Ex2"] + jnp.sum(x_batch**2, axis=0),
#       "Ey2": state["Ey2"] + jnp.sum(y_batch**2, axis=0),
#       "samples": state["samples"] + batch_size,
#   }

# def online_pearson_finalize(state):
#   samples = state["samples"]
#   Exy = state["Exy"] / samples
#   Ex = state["Ex"] / samples
#   Ey = state["Ey"] / samples
#   Ex2 = state["Ex2"] / samples
#   Ey2 = state["Ey2"] / samples

#   print("finalizing pearson")
#   print((Exy - Ex[jnp.newaxis, :] * Ey).min(), (Exy - Ex[jnp.newaxis, :] * Ey).max())
#   print(jnp.sqrt(Ex2 - Ex**2).min(), jnp.sqrt(Ex2 - Ex**2).max())
#   print(jnp.sqrt(Ey2 - Ey**2).min(), jnp.sqrt(Ey2 - Ey**2).max())
#   # Note that this will not be symmetric in general
#   # return (Exy - Ex[jnp.newaxis, :] * Ey) / jnp.sqrt(Ex2 - Ex**2)[jnp.newaxis, :] / jnp.sqrt(Ey2 -
#   # Ey**2)
#   return (Exy - Ex[jnp.newaxis, :] * Ey)

# # def online_

# def test_one_batch():
#   rp = RngPooper(random.PRNGKey(123))
#   n = 3
#   x_batch = random.normal(rp.poop(), (1024, n))
#   y_batch = random.normal(rp.poop(), (1024, n))
#   state = online_pearson_init_state(n)
#   state = online_pearson_update(state, x_batch, y_batch)
#   pred = online_pearson_finalize(state)
#   print(pred)
#   gt = jnp.corrcoef(x_batch, y_batch, rowvar=False)[:n, n:]
#   print(gt)
#   np.testing.assert_allclose(pred, gt, rtol=1e0, atol=1e-2)

# def test_multiple_batches():
#   rp = RngPooper(random.PRNGKey(123))
#   n = 3
#   x_batch = random.normal(rp.poop(), (1024, n))
#   y_batch = random.normal(rp.poop(), (1024, n))
#   state = online_pearson_init_state(n)
#   state = online_pearson_update(state, x_batch[:500, :], y_batch[:500, :])
#   state = online_pearson_update(state, x_batch[500:750, :], y_batch[500:750, :])
#   state = online_pearson_update(state, x_batch[750:, :], y_batch[750:, :])
#   pred = online_pearson_finalize(state)
#   print(pred)
#   gt = jnp.corrcoef(x_batch, y_batch, rowvar=False)[:n, n:]
#   print(gt)
#   np.testing.assert_allclose(pred, gt, rtol=1e0, atol=1e-2)

# if __name__ == "__main__":
#   test_one_batch()
#   test_multiple_batches()
