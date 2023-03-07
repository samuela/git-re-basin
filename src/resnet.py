"""
This code is copied (almost) verbatim from https://github.com/n2cholas/jax-resnet/
It is only used for our resnet50 experiments on ImageNet.
"""

from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import ConvBlock, ModuleDef
# from .splat import SplAtConv2d

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(64,
                                   kernel_size=(7, 7),
                                   strides=(2, 2),
                                   padding=[(3, 3), (3, 3)])(x)


class ResNetDStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 32

    # If True, n_filters for first conv is (input_channels + 1) * 8
    adaptive_first_width: bool = False

    @nn.compact
    def __call__(self, x):
        cls = partial(self.conv_block_cls, kernel_size=(3, 3), padding=((1, 1), (1, 1)))
        first_width = (8 * (x.shape[-1] + 1)
                       if self.adaptive_first_width else self.stem_width)
        x = cls(first_width, strides=(2, 2))(x)
        x = cls(self.stem_width, strides=(1, 1))(x)
        x = cls(self.stem_width * 2, strides=(1, 1))(x)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x)
        return x


class ResNetDSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        if x.shape[-1] != out_shape[-1]:
            x = self.conv_block_cls(out_shape[-1], (1, 1), activation=lambda y: y)(x)
        return x


class ResNeStSkipConnection(ResNetDSkipConnection):
    # Inheritance to ensures our variables dict has the right names.
    pass


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1  # cardinality
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        # Downsampling strides in 3x3 conv instead of 1x1 conv, which improves accuracy.
        # This variant is called ResNet V1.5 (matches torchvision).
        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=((1, 1), (1, 1)))(y)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNetDBlock(ResNetBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNetDBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNeStBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNeStSkipConnection
    avg_pool_first: bool = False
    radix: int = 2

    splat_cls: ModuleDef = None # SplAtConv2d

    @nn.compact
    def __call__(self, x):
        assert self.radix == 2  # TODO: implement radix != 2

        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)

        if self.strides != (1, 1) and self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.splat_cls(group_width,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding=[(1, 1), (1, 1)],
                           groups=self.groups,
                           radix=self.radix)(y)

        if self.strides != (1, 1) and not self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)

        return self.activation(y + skip_cls(self.strides)(x, y.shape))


def ResNet(
    block_cls: ModuleDef,
    *,
    stage_sizes: Sequence[int],
    n_classes: int,
    hidden_sizes: Sequence[int] = (64, 128, 256, 512),
    conv_cls: ModuleDef = nn.Conv,
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9),
    conv_block_cls: ModuleDef = ConvBlock,
    stem_cls: ModuleDef = ResNetStem,
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1))),
) -> nn.Sequential:
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    layers = [stem_cls(), pool_fn]

    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for b in range(n_blocks):
            strides = (1, 1) if i == 0 or b != 0 else (2, 2)
            layers.append(block_cls(n_hidden=hsize, strides=strides))

    layers.append(partial(jnp.mean, axis=(1, 2)))  # global average pool
    layers.append(nn.Dense(n_classes))
    return nn.Sequential(layers)


# yapf: disable
ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                   stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)

WideResNet50 = partial(ResNet50, hidden_sizes=(128, 256, 512, 1024),
                       block_cls=partial(ResNetBottleneckBlock, expansion=2))
WideResNet101 = partial(ResNet101, hidden_sizes=(128, 256, 512, 1024),
                        block_cls=partial(ResNetBottleneckBlock, expansion=2))

ResNeXt50 = partial(ResNet50,
                    block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=4))
ResNeXt101 = partial(ResNet101,
                     block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=8))

ResNetD18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)

ResNeSt50Fast = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                        stem_cls=ResNetDStem,
                        block_cls=partial(ResNeStBottleneckBlock, avg_pool_first=True))
ResNeSt50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNeStBottleneckBlock)
ResNeSt101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt269 = partial(ResNet, stage_sizes=STAGE_SIZES[269],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
