from einops import reduce
from flax import linen as nn

def reverse_compose(x, fs):
  for f in fs:
    x = f(x)
  return x

class Block(nn.Module):
  num_channels: int = None
  strides: int = None

  def setup(self):
    self.conv1 = nn.Conv(features=self.num_channels,
                         kernel_size=(3, 3),
                         strides=self.strides,
                         use_bias=False)
    self.norm1 = nn.LayerNorm()
    self.conv2 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=1, use_bias=False)
    self.norm2 = nn.LayerNorm()

    # When strides != 1, then it's 2, which means that we halve the width and height of the input, while doubling the
    # number of channels. Therefore we need to correspondingly halve the width and height of the residuals/shortcut.
    if self.strides != 1:
      assert self.strides == 2

      # Supposedly this is the original description, but it is not easily comaptible with our weight matching stuff
      # since it plays games with the channel structure by padding things around.
      # self.shortcut = lambda x: jnp.pad(x[:, ::2, ::2, :], (
      #     (0, 0), (0, 0), (0, 0), (self.num_channels // 4, self.num_channels // 4)),
      #                                   "constant",
      #                                   constant_values=0)

      # This is not the original, but is fairly common based on other implementations.
      self.shortcut = nn.Sequential([
          nn.Conv(features=self.num_channels,
                  kernel_size=(3, 3),
                  strides=self.strides,
                  use_bias=False),
          nn.LayerNorm()
      ])
    else:
      self.shortcut = lambda x: x

  def __call__(self, x):
    y = x
    y = self.conv1(y)
    y = self.norm1(y)
    y = nn.relu(y)
    y = self.conv2(y)
    y = self.norm2(y)
    return nn.relu(y + self.shortcut(x))

class BlockGroup(nn.Module):
  num_channels: int = None
  num_blocks: int = None
  strides: int = None

  def setup(self):
    assert self.num_blocks > 0
    self.blocks = (
        [Block(num_channels=self.num_channels, strides=self.strides)] +
        [Block(num_channels=self.num_channels, strides=1) for _ in range(self.num_blocks - 1)])

  def __call__(self, x):
    return reverse_compose(x, self.blocks)

class ResNet(nn.Module):
  blocks_per_group: int = None
  num_classes: int = None
  width_multiplier: int = 1

  def setup(self):
    wm = self.width_multiplier

    self.conv1 = nn.Conv(features=16 * wm, kernel_size=(3, 3), use_bias=False)
    self.norm1 = nn.LayerNorm()

    channels_per_group = (16 * wm, 32 * wm, 64 * wm)
    strides_per_group = (1, 2, 2)
    self.blockgroups = [
        BlockGroup(num_channels=c, num_blocks=b, strides=s)
        for c, b, s in zip(channels_per_group, self.blocks_per_group, strides_per_group)
    ]

    self.dense = nn.Dense(self.num_classes)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = nn.relu(x)
    x = reverse_compose(x, self.blockgroups)
    x = reduce(x, "n h w c -> n c", "mean")
    x = self.dense(x)
    x = nn.log_softmax(x)
    return x

BLOCKS_PER_GROUP = {
    "resnet20": (3, 3, 3),
    "resnet32": (5, 5, 5),
    "resnet44": (7, 7, 7),
    "resnet56": (9, 9, 9),
    "resnet110": (18, 18, 18),
}
