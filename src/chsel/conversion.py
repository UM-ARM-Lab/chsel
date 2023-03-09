import torch

from arm_pytorch_utilities.tensor_utils import ensure_tensor
from pytorch_kinematics import matrix_to_rotation_6d, rotation_6d_to_matrix


def RT_to_continuous_representation(R, T):
    q = matrix_to_rotation_6d(R)
    x = torch.cat([q, T], dim=-1)
    return x


def continuous_representation_to_RT(x, device='cpu', dtype=torch.float):
    q_ = x[..., :6]
    t = x[..., 6:]
    qq, TT = ensure_tensor(device, dtype, q_, t)
    RR = rotation_6d_to_matrix(qq)
    return RR, TT
