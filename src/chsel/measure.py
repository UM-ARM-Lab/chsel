import abc

import numpy as np
import torch

import pytorch_kinematics as pk
from chsel.conversion import RT_to_continuous_representation, continuous_representation_to_RT
from chsel.se2 import xyz_to_uv, uv_to_xyz, construct_plane_basis


class MeasureFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def grad(self, x):
        pass

    def __init__(self, measure_dim, dtype=torch.float32, device="cpu"):
        self.measure_dim = measure_dim
        self.dtype = dtype
        self.device = device

    def get_numpy_x(self, R, T):
        return RT_to_continuous_representation(R, T).cpu().numpy()

    def get_torch_RT(self, x):
        return continuous_representation_to_RT(x, self.device, self.dtype)


class RotMeasure(MeasureFunction):
    def __init__(self, measure_dim, offset=0, **kwargs):
        super().__init__(measure_dim, **kwargs)
        self.offset = offset

    def __call__(self, x):
        return x[..., self.offset:self.measure_dim]

    def grad(self, x):
        grad = np.zeros((self.measure_dim, x.shape[-1]))
        grad[:, self.offset:self.measure_dim] = np.eye(self.measure_dim)
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad


class PositionMeasure(MeasureFunction):
    """
    :param measure_dim: The number of translation dimensions to use for the QD measure in the order of XYZ -
    this is what is ensured diversity across. For example, if you use qd_measure_dim=2, only diversity of estimated
    transforms across X and Y translation terms will be ensured. This may be useful if you have an upright prior
    that the object lies on a plane normal to Z. Empirically, we found no significant difference in performance
    between 2 and 3 dimensions.
    """

    def __call__(self, x):
        return x[..., 6:6 + self.measure_dim]

    def grad(self, x):
        grad = np.zeros((self.measure_dim, x.shape[-1]))
        grad[:, 6:6 + self.measure_dim] = np.eye(self.measure_dim)
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad


class SE2Measure(MeasureFunction):
    def __init__(self, *args, axis_of_rotation, offset_along_axis=0, **kwargs):
        self.axis_of_rotation = axis_of_rotation
        self.offset_along_axis = offset_along_axis
        self.origin = self.axis_of_rotation * self.offset_along_axis
        self.axis_u, self.axis_v = construct_plane_basis(self.axis_of_rotation)

        super().__init__(*args, **kwargs)
        self.dtype = self.axis_of_rotation.dtype
        self.device = self.axis_of_rotation.device

    def get_numpy_x(self, R, T):
        # projection of rotation down to axis of rotation
        axis_angle = pk.matrix_to_axis_angle(R)
        # dot product against the given axis of rotation to get the angle
        theta = axis_angle @ self.axis_of_rotation
        uv = xyz_to_uv(T, self.origin, self.axis_of_rotation, self.axis_u, self.axis_v)
        x = torch.cat((theta.view(-1, 1), uv), dim=-1)
        return x.cpu().numpy()

    def get_torch_RT(self, x):
        # convert back to R, T
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device, dtype=self.dtype)
        R = pk.axis_and_angle_to_matrix_33(self.axis_of_rotation, x[..., 0])
        T = uv_to_xyz(x[..., 1:], self.origin, self.axis_u, self.axis_v)
        return R, T


class SE2PositionMeasure(SE2Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

    def __call__(self, x):
        return x[..., 1:]

    def grad(self, x):
        grad = np.zeros((2, x.shape[-1]))
        grad[:, 1:] = np.eye(2)
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad


class SE2AngleMeasure(SE2Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    def __call__(self, x):
        # need to keep last dimension for compatibility with other measures
        # need to clip/wrap to avoid having really high measure; we will restrict it to [-pi, pi]
        theta = x[..., 0:1]
        # ensure [0, 2pi]
        theta = np.mod(theta, 2 * np.pi)
        # convert to [-pi, pi]
        theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)

        return theta

    def grad(self, x):
        grad = np.zeros((1, x.shape[-1]))
        grad[:, 0] = 1
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad
