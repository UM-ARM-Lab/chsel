import enum

import numpy as np
import torch

from pytorch_kinematics import transforms as tf


class InitMethod(enum.Enum):
    ORIGIN = 0
    CONTACT_CENTROID = 1
    RANDOM = 2


def initialize_transform_estimates(B, freespace_ranges, init_method: InitMethod, contact_points,
                                   device="cpu", dtype=torch.float):
    # translation is 0,0,0
    best_tsf_guess = random_upright_transforms(B, dtype, device)
    if init_method == InitMethod.ORIGIN:
        pass
    elif init_method == InitMethod.CONTACT_CENTROID:
        centroid = contact_points.mean(dim=0).to(device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = centroid
    elif init_method == InitMethod.RANDOM:
        trans = np.random.uniform(freespace_ranges[:, 0], freespace_ranges[:, 1], (B, 3))
        trans = torch.tensor(trans, device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = trans
    else:
        raise RuntimeError(f"Unsupported initialization method: {init_method}")
    return best_tsf_guess


def reinitialize_transform_estimates(B, best_tsf_guess, radian_sigma=0.3, translation_sigma=0.05):
    # sample rotation and translation around the previous best solution to reinitialize
    dtype = best_tsf_guess.dtype
    device = best_tsf_guess.device

    # sample delta rotations in axis angle form
    T_init = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)

    delta_R = random_rotation_perturbations(B, dtype, device, radian_sigma)
    T_init[:, :3, :3] = delta_R @ best_tsf_guess[:3, :3]
    T_init[:, :3, 3] = best_tsf_guess[:3, 3]

    delta_t = torch.randn((B, 3), dtype=dtype, device=device) * translation_sigma
    T_init[:, :3, 3] += delta_t
    # ensure one of them (first of the batch) has the exact transform
    T_init[0] = best_tsf_guess
    best_tsf_guess = T_init
    return best_tsf_guess


def random_rotation_perturbations(B, dtype, device, radian_sigma):
    delta_R = torch.randn((B, 3), dtype=dtype, device=device) * radian_sigma
    delta_R = tf.axis_angle_to_matrix(delta_R)
    return delta_R


def random_upright_transforms(B, dtype, device, translation=None):
    # initialize guesses with a prior; since we're trying to retrieve an object, it's reasonable to have the prior
    # that the object only varies in yaw (and is upright)
    axis_angle = torch.zeros((B, 3), dtype=dtype, device=device)
    axis_angle[:, -1] = torch.rand(B, dtype=dtype, device=device) * 2 * np.pi
    R = tf.axis_angle_to_matrix(axis_angle)
    init_pose = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    init_pose[:, :3, :3] = R
    if translation is not None:
        init_pose[:, :3, 3] = translation
    return init_pose
