import torch
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
