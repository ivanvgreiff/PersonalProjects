import jax
from jax import numpy as jnp
from flax import linen as nn
from functools import partial
from typing import Any, Tuple
from collections.abc import Callable, Sequence

__all__ = ["lenet5"]

#! The final layer doesn't have Softmax activation becasue it's implemented when calculating loss.
class LeNet5(nn.Module):
    num_classes: int
    @nn.compact
    def __call__(self, x, train:bool):
        x = nn.Conv(features=6, kernel_size=(5, 5), padding="SAME", kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.activation.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID", kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.activation.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(features=84, kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(features=self.num_classes, kernel_init=nn.initializers.xavier_normal())(x) #! num_classes = 10
        return x
            
def lenet5(num_classes):
    return LeNet5(num_classes)