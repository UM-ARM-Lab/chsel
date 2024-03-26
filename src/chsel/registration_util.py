import os
from typing import Optional

import torch
from matplotlib import pyplot as plt, cm as cm
from ribs.visualize import grid_archive_heatmap
from chsel.types import SimilarityTransform, ICPSolution
import logging

logger = logging.getLogger(__file__)

poke_index = 0
sgd_index = 0

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


def _savefig(directory, fig, suffix=""):
    savepath = os.path.join(directory, f"{poke_index}{suffix}.png")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    fig.clf()
    plt.close(fig)


def plot_poke_losses(losses, savedir=ROOT_DIR, loss_name='loss', ylabel='cost', xlabel='iteration', logy=True):
    global poke_index, sgd_index
    losses = torch.stack(losses).cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    fig.suptitle(f"poke {poke_index}")

    for b in range(losses.shape[1]):
        c = (b + 1) / losses.shape[1]
        ax.plot(losses[:, b], c=cm.GnBu(c))
    _savefig(os.path.join(savedir, 'img', loss_name), fig)
    sgd_index = 0


def plot_qd_archive(archive, savedir=ROOT_DIR):
    global poke_index
    fig, ax = plt.subplots()
    grid_archive_heatmap(archive, ax=ax, vmin=-4, vmax=0)
    # # for MUSTARD task
    # ax.scatter(0.25, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.suptitle(f"poke {poke_index} QD: {archive.stats.norm_qd_score}")
    _savefig(os.path.join(savedir, 'img/qd'), fig)


def plot_sgd_losses(losses, savedir=ROOT_DIR):
    global poke_index, sgd_index
    losses = torch.stack(losses).cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_xlabel('sgd iteration')
    ax.set_ylabel('cost')
    ax.set_yscale('log')
    fig.suptitle(f"poke {poke_index} restart {sgd_index}")

    for b in range(losses.shape[1]):
        c = (b + 1) / losses.shape[1]
        ax.plot(losses[:, b], c=cm.PuRd(c))

    _savefig(os.path.join(savedir, 'img/sgd'), fig, suffix=f"_{sgd_index}")
    sgd_index += 1


def apply_init_transform(Xt, init_transform: Optional[SimilarityTransform] = None):
    b, size_X, dim = Xt.shape
    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
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
        # apply the init transform to the input point cloud
        Xt = apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = torch.eye(dim, device=Xt.device, dtype=Xt.dtype).repeat(b, 1, 1)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)
    return Xt, R, T, s


def apply_similarity_transform(
        X: torch.Tensor, R: torch.Tensor, T: torch.Tensor = None, s: torch.Tensor = None
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    if s is not None:
        R = s[:, None, None] * R
    X = R @ X.transpose(-1, -2)
    if T is not None:
        X = X + T[:, :, None]
    return X.transpose(-1, -2)


def solution_to_world_to_link_matrix(res: ICPSolution, invert_rot_matrix=False):
    batch = res.RTs.T.shape[0]
    device = res.RTs.T.device
    dtype = res.RTs.T.dtype
    T = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)
    if invert_rot_matrix:
        T[:, :3, :3] = res.RTs.R.transpose(-1, -2)
    else:
        T[:, :3, :3] = res.RTs.R
    T[:, :3, 3] = res.RTs.T
    return T


