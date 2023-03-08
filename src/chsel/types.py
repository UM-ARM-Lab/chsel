import torch
import enum
import numpy as np
from typing import List, NamedTuple, Union


# compatible with pytorch3d; duplicated here to avoid dependence

# named tuples for inputs/outputs
class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


class ICPSolution(NamedTuple):
    converged: bool
    rmse: Union[torch.Tensor, None]
    Xt: torch.Tensor
    RTs: SimilarityTransform
    t_history: List[SimilarityTransform]


class SemanticsClass(enum.Enum):
    FREE = 0
    OCCUPIED = 1


# for known SDF value, it is the actual float value
Semantics = Union[float, SemanticsClass]
