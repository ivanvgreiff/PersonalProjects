from ml_collections import FrozenConfigDict, ConfigDict, FieldReference
from chex import Shape, ArrayTree, Scalar, PRNGKey, PyTreeDef
from jax.typing import ArrayLike, DTypeLike
from optax import GradientTransformation, OptState
from collections.abc import Callable
OptaxLoss = Callable[[ArrayTree, ArrayLike], ArrayLike]
KeyArray = ArrayLike
CustomOptState = dict[str, ArrayTree | ArrayLike]

