from typing import Optional
import math
import torch
from chsel.costs import VolumetricCost
from chsel.registration_util import plot_poke_losses, plot_sgd_losses
from chsel.se2 import construct_plane_basis, xyz_to_uv, uv_to_xyz
from chsel.types import SimilarityTransform, ICPSolution
import pytorch_kinematics as pk
from pytorch_kinematics import random_rotations, matrix_to_rotation_6d, rotation_6d_to_matrix

import logging
import typing

logger = logging.getLogger(__file__)


def volumetric_registration_sgd(
        volumetric_cost: VolumetricCost,
        batch=30,
        init_transform: Optional[SimilarityTransform] = None,
        max_iterations: int = 10,  # quite robust to this parameter (number of restarts)
        relative_rmse_thr: float = 1e-6,
        estimate_scale: bool = False,
        verbose: bool = False,
        save_loss_plot=False,
        **kwargs,
) -> ICPSolution:
    """
    Executes the iterative closest point (ICP) algorithm [1, 2] in order to find
    a similarity transformation (rotation `R`, translation `T`, and
    optionally scale `s`) between two given differently-sized sets of
    `d`-dimensional points `X` and `Y`, such that:

    `s[i] X[i] R[i] + T[i] = Y[NN[i]]`,

    for all batch indices `i` in the least squares sense. Here, Y[NN[i]] stands
    for the indices of nearest neighbors from `Y` to each point in `X`.
    Note, however, that the solution is only a local optimum.

    Args:
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        **init_transform**: A named-tuple `SimilarityTransform` of tensors
            `R`, `T, `s`, where `R` is a batch of orthonormal matrices of
            shape `(minibatch, d, d)`, `T` is a batch of translations
            of shape `(minibatch, d)` and `s` is a batch of scaling factors
            of shape `(minibatch,)`.
        **max_iterations**: The maximum number of ICP iterations.
        **relative_rmse_thr**: A threshold on the relative root mean squared error
            used to terminate the algorithm.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes the identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **sgd_iterations**: Number of epochs to run SGD for computing alignment
        **sgd_lr**: Learning rate of SGD for computing alignment
        **verbose**: If `True`, prints status messages during each ICP iteration.

    Returns:
        A named tuple `ICPSolution` with the following fields:
        **converged**: A boolean flag denoting whether the algorithm converged
            successfully (=`True`) or not (=`False`).
        **rmse**: Attained root mean squared error after termination of ICP.
        **Xt**: The point cloud `X` transformed with the final transformation
            (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
            instance of `Pointclouds`, otherwise returns `torch.Tensor`.
        **RTs**: A named tuple `SimilarityTransform` containing
        a batch of similarity transforms with fields:
            **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
            **T**: Batch of translations of shape `(minibatch, d)`.
            **s**: batch of scaling factors of shape `(minibatch, )`.
        **t_history**: A list of named tuples `SimilarityTransform`
            the transformation parameters after each ICP iteration.
    """
    dtype = volumetric_cost.dtype
    device = volumetric_cost.device
    dim = 3

    if init_transform is not None:
        b = init_transform.R.shape[0]
        try:
            R, T, s = init_transform
            R = R.clone()
            T = T.clone()
            s = s.clone()
            assert (
                    R.shape == torch.Size((b, dim, dim))
                    and T.shape == torch.Size((b, dim))
                    and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            )
    else:
        b = batch
        # initialize the transformation with identity
        R = torch.eye(dim, device=device, dtype=dtype).repeat(b, 1, 1)
        T = torch.zeros((b, dim), device=device, dtype=dtype)
        s = torch.ones(b, device=device, dtype=dtype)

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []
    losses = []

    # --- SGD
    for iteration in range(max_iterations):
        # get the alignment of the nearest neighbors from Yt with Xt_init
        sim_transform, rmse = volumetric_points_alignment(
            volumetric_cost,
            batch=batch,
            estimate_scale=estimate_scale,
            R=R,
            T=T,
            s=s,
            save_loss_plot=save_loss_plot,
            **kwargs
        )
        R, T, s = sim_transform
        losses.append(rmse)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                    f"registration iteration {iteration}: mean/max rmse = "
                    + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                    + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            logger.info(rmse_msg)

        # check for convergence
        if relative_rmse.mean() <= relative_rmse_thr:
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if save_loss_plot:
        plot_poke_losses(losses)

    if verbose:
        if converged:
            logger.info(f"ICP has converged in {iteration + 1} iterations.")
        else:
            logger.info(f"ICP has not converged in {max_iterations} iterations.")

    return ICPSolution(converged, rmse, torch.empty(0), SimilarityTransform(R, T, s), t_history)


def volumetric_points_alignment(
        volumetric_cost: VolumetricCost,
        axis_of_rotation=None,
        offset_along_axis=0,
        batch=30,
        estimate_scale: bool = False,
        R: torch.Tensor = None, T: torch.tensor = None, s: torch.tensor = None,
        iterations: int = 50,
        lr: float = 0.01,
        save_loss_plot=True,
        verbose=False
) -> typing.Tuple[SimilarityTransform, torch.tensor]:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense using gradient descent

    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **sgd_iterations**: Number of epochs to run
        **lr**: Learning rate

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.
    """
    if axis_of_rotation is not None:
        return volumetric_points_alignment_2d(volumetric_cost, axis_of_rotation,
                                              offset_along_axis=offset_along_axis, batch=batch,
                                              estimate_scale=estimate_scale, R=R, T=T, s=s,
                                              iterations=iterations, lr=lr, save_loss_plot=save_loss_plot,
                                              verbose=verbose)

    device = volumetric_cost.device
    dtype = volumetric_cost.dtype
    # works for only 3D transforms
    dim = 3

    if R is None:
        R = random_rotations(batch, dtype=dtype, device=device)
        T = torch.randn((batch, dim), dtype=dtype, device=device)
        s = torch.ones(batch, dtype=dtype, device=device)

    # convert to non-redundant representation
    q = matrix_to_rotation_6d(R)

    # set them up as parameters for training
    q.requires_grad = True
    T.requires_grad = True
    if estimate_scale:
        s.requires_grad = True

    optimizer = torch.optim.Adam([q, T, s], lr=lr)

    def get_usable_transform_representation():
        nonlocal T
        # we get a more usable representation of R
        R = rotation_6d_to_matrix(q)
        return R, T

    losses = []

    for epoch in range(iterations):
        R, T = get_usable_transform_representation()

        total_loss = volumetric_cost(R, T, s)
        total_loss.mean().backward()
        losses.append(total_loss.detach())

        # visualize gradients on the losses
        volumetric_cost.visualize(R, T, s)

        optimizer.step()
        optimizer.zero_grad()

    if save_loss_plot:
        plot_sgd_losses(losses)

    R, T = get_usable_transform_representation()
    return SimilarityTransform(R.detach().clone(), T.detach().clone(),
                               s.detach().clone()), total_loss.detach().clone()


def volumetric_points_alignment_2d(
        volumetric_cost: VolumetricCost,
        axis_of_rotation: torch.tensor,
        offset_along_axis=0,
        batch=30,
        estimate_scale: bool = False,
        R: torch.Tensor = None, T: torch.tensor = None, s: torch.tensor = None,
        iterations: int = 50,
        lr: float = 0.01,
        save_loss_plot=True,
        verbose=False
) -> typing.Tuple[SimilarityTransform, torch.tensor]:
    """
    Similar to above but for 2D transforms with a fixed axis of rotation
    """
    # ensure axis is normalized
    axis_of_rotation = axis_of_rotation / torch.norm(axis_of_rotation)
    origin = axis_of_rotation * offset_along_axis
    axis_u, axis_v = construct_plane_basis(axis_of_rotation)

    device = volumetric_cost.device
    dtype = volumetric_cost.dtype
    # works for only 3D transforms

    if R is None:
        theta = torch.rand(batch, dtype=dtype, device=device) * math.pi * 2
        R = pk.axis_and_angle_to_matrix_33(axis_of_rotation, theta)
        T = torch.randn((batch, 3), dtype=dtype, device=device)
        s = torch.ones(batch, dtype=dtype, device=device)

    # projection of rotation down to axis of rotation
    axis_angle = pk.matrix_to_axis_angle(R)
    # dot product against the given axis of rotation to get the angle
    theta = axis_angle @ axis_of_rotation

    # extract xy
    # project them onto the SE(2) plane (offset along the axis)
    uv = xyz_to_uv(T, origin, axis_of_rotation, axis_u, axis_v)

    # set them up as parameters for training
    theta.requires_grad = True
    uv.requires_grad = True
    if estimate_scale:
        s.requires_grad = True

    optimizer = torch.optim.Adam([theta, uv, s], lr=lr)

    def get_usable_transform_representation():
        nonlocal T
        RR = pk.axis_and_angle_to_matrix_33(axis_of_rotation, theta)
        T = uv_to_xyz(uv, origin, axis_u, axis_v)
        return RR, T

    losses = []

    for epoch in range(iterations):
        R, T = get_usable_transform_representation()

        total_loss = volumetric_cost(R, T, s)
        total_loss.mean().backward()
        losses.append(total_loss.detach())

        # visualize gradients on the losses
        volumetric_cost.visualize(R, T, s)

        optimizer.step()
        optimizer.zero_grad()

    if save_loss_plot:
        plot_sgd_losses(losses)

    if verbose:
        print(f"pose loss {total_loss.mean().item()}")
    R, T = get_usable_transform_representation()
    return SimilarityTransform(R.detach().clone(), T.detach().clone(),
                               s.detach().clone()), total_loss.detach().clone()
