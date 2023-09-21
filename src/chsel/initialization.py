import enum

import numpy as np
import torch
from chsel import conversion

from pytorch_kinematics import transforms as tf


class InitMethod(enum.Enum):
    ORIGIN = 0
    CONTACT_CENTROID = 1
    RANDOM = 2


def initialize_transform_estimates(B, freespace_ranges, init_method: InitMethod, contact_points,
                                   device="cpu", dtype=torch.float, trans_noise=0.05):
    # translation is 0,0,0
    best_tsf_guess = random_upright_transforms(B, dtype, device)
    if init_method == InitMethod.ORIGIN:
        pass
    elif init_method == InitMethod.CONTACT_CENTROID:
        centroid = contact_points.mean(dim=0).to(device=device, dtype=dtype)
        trans = np.random.uniform(np.ones(3) * -trans_noise, np.ones(3) * trans_noise, (B, 3))
        best_tsf_guess[:, :3, 3] = centroid + torch.tensor(trans, device=device, dtype=dtype)
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


def reinitialize_transform_around_elites(tsfs, rmse, elite_percentile=0.2, keep_elite=True):
    # find elite tsfs and reinitialize around them
    # sample rotation and translation around the previous best solution to reinitialize
    dtype = tsfs.dtype
    device = tsfs.device

    B = tsfs.shape[0]

    # find elite tsfs
    k = int(elite_percentile * B)
    elite_idx = np.argpartition(rmse.cpu().numpy(), k)[:k]
    elite_tsfs = tsfs[elite_idx]

    # sample in the continuous 9D space
    elite_x = conversion.RT_to_continuous_representation(elite_tsfs[:, :3, :3], elite_tsfs[:, :3, 3])
    # sample B 9D vectors around the mean of elite_x with std matching its std
    elite_x_mean = elite_x.mean(dim=0)
    elite_x_std = elite_x.std(dim=0)
    x = torch.randn((B, 9), dtype=dtype, device=device) * elite_x_std + elite_x_mean
    if keep_elite:
        x[:k] = elite_x

    R, T = conversion.continuous_representation_to_RT(x)
    tsfs = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    tsfs[:, :3, :3] = R
    tsfs[:, :3, 3] = T
    return tsfs


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
