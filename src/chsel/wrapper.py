from typing import Sequence, Optional

import math
import chsel
import chsel.sgd
import torch
import numpy as np

from chsel import costs
from chsel import types
from chsel import quality_diversity
from chsel import registration_util

import pytorch_volumetric as pv
import pytorch_kinematics as pk

import logging

logger = logging.getLogger(__file__)


class CHSEL:
    def __init__(self,
                 obj_sdf: pv.ObjectFrameSDF,
                 positions: torch.tensor, semantics: Sequence[types.Semantics],
                 cost=costs.VolumetricDirectSDFCost,
                 free_voxels: Optional[pv.Voxels] = None,
                 occupied_voxels: Optional[pv.Voxels] = None,
                 known_sdf_voxels: Optional[pv.Voxels] = None,
                 free_voxels_resolution=0.01,
                 sgd_solution_outlier_rejection_ratio=5.0,
                 archive_range_sigma=3,
                 bins=40,
                 qd_iterations=100,
                 qd_alg=quality_diversity.CMAMEGA,
                 qd_measure_dim=2,
                 qd_measure_fn=None,
                 qd_measure_grad=None,
                 savedir=registration_util.ROOT_DIR,
                 debug=False,
                 qd_alg_kwargs=None,
                 **cost_kwargs):
        """

        :param obj_sdf: Object centered signed distance field (SDF)
        :param positions: World frame positions of the observed points
        :param semantics: Semantics of the observed points
        :param cost: Class that implements Eq. 6
        :param free_voxels: Explicit specification of free space voxels; if not specified they will be extracted
        from the given points that have the FREE semantics. If the observed points are derived from free voxels,
        it will save recreating them if these are specified directly.
        :param occupied_voxels: Similarly to the above
        :param known_sdf_voxels: Similarly to the above
        :param free_voxels_resolution: Resolution in meters of the side-length of each voxel; this should scale with
        the object's size; it is a good idea that at least 10 voxels span each dimension of the object
        :param sgd_solution_outlier_rejection_ratio: The ratio of CHSEL cost function to the best one found, above which
        to be considered an outlier and rejected. This is used to prevent the archive range being polluted with outliers
        :param archive_range_sigma: b_sigma the number of standard deviations to consider for the archive
        :param bins: How many bins each dimension of the archive will have
        :param qd_iterations: n_o number of quality diversity optimization iterations
        :param qd_alg: The quality diversity optimization algorithm to use
        :param qd_measure_dim: The number of translation dimensions to use for the QD measure in the order of XYZ -
        this is what is ensured diversity across. For example, if you use qd_measure_dim=2, only diversity of estimated
        transforms across X and Y translation terms will be ensured. This may be useful if you have an upright prior
        that the object lies on a plane normal to Z. Empirically, we found no significant difference in performance
        between 2 and 3 dimensions.
        :param qd_measure_fn: The function to use for the QD measure. If not specified, the position is used.
        :param qd_measure_grad: The gradient of the QD measure function. If not specified, the one associated with
        position is used.
        :param savedir: Directory to save loss plots
        :param debug:
        """
        self.obj_sdf = obj_sdf
        self.outlier_rejection_ratio = sgd_solution_outlier_rejection_ratio

        self.dtype = positions.dtype
        self.device = positions.device

        self.positions = positions
        # to allow batch indexing
        semantics = np.asarray(semantics, dtype=object)
        self.semantics = semantics

        self.debug = debug
        self.archive_range_sigma = archive_range_sigma
        self.bins = bins
        self.qd_iterations = qd_iterations
        self.qd_alg = qd_alg
        self.savedir = savedir
        # extract indices
        self._free = torch.tensor([s == chsel.SemanticsClass.FREE for s in semantics])
        self._occupied = torch.tensor([s == chsel.SemanticsClass.OCCUPIED for s in semantics])
        self._known = ~self._free & ~self._occupied

        # intermediate results for use outside for debugging
        self.res_init = None
        self.qd = None
        self.res_history = []
        self.res_init_history = []

        self._qd_alg_kwargs = {"measure_dim": qd_measure_dim,
                               "measure_fn": qd_measure_fn,
                               "measure_grad": qd_measure_grad}
        if qd_alg_kwargs is not None:
            self._qd_alg_kwargs.update(qd_alg_kwargs)

        # extract the known free voxels from the given free points
        if free_voxels is None:
            free_voxels = pv.ExpandingVoxelGrid(free_voxels_resolution, [(0, 0) for _ in range(3)], dtype=self.dtype,
                                                device=self.device)
            free_voxels[positions[self._free]] = 1
        # TODO extract the known occupied points (this has been unused in any experiments so far)
        # extract the known SDF points
        if known_sdf_voxels is None:
            known_sdf_values = semantics[self._known].astype(float)
            known_sdf_values = torch.tensor(known_sdf_values, dtype=self.dtype, device=self.device)
            known_sdf_voxels = pv.VoxelSet(positions[self._known], known_sdf_values)

        cost_options = {
            "scale_known_freespace": 20,
            "debug": self.debug,
            "surface_threshold": free_voxels_resolution
        }
        cost_options.update(cost_kwargs)
        # construct the cost function
        self.volumetric_cost = cost(free_voxels, known_sdf_voxels, self.obj_sdf, dtype=self.dtype, device=self.device,
                                    **cost_options)

    def evaluate_homogeneous(self, H_world_to_link: torch.tensor, use_scale=False):
        """
        Evaluate the discrepency between the observed semantic point cloud and the given transforms (world to link)
        :return: the cost for each transform
        """
        if use_scale:
            s = H_world_to_link[:, -1, -1]
        else:
            s = None
        return self.evaluate(H_world_to_link[:, :3, :3], H_world_to_link[:, :3, 3], s)

    def evaluate(self, R: torch.tensor, T: torch.tensor, s: torch.tensor = None):
        """
        Evaluate the discrepency between the observed semantic point cloud and the given transforms (world to link)
        :return: the cost for each transform
        """
        return self.volumetric_cost(R, T, s)

    def update(self, positions, semantics):
        """
        Update the observed point cloud and semantics
        """
        if len(positions) == 0:
            return
        _free = torch.tensor([s == chsel.SemanticsClass.FREE for s in semantics])
        _occupied = torch.tensor([s == chsel.SemanticsClass.OCCUPIED for s in semantics])
        _known = ~_free & ~_occupied
        self._free = torch.cat([self._free, _free])
        self._occupied = torch.cat([self._occupied, _occupied])
        self._known = torch.cat([self._known, _known])
        self.positions = torch.cat([self.positions, positions])
        self.volumetric_cost.free_voxels[positions[_free]] = 1
        semantics = np.asarray(semantics, dtype=object)
        self.semantics = np.concatenate([self.semantics, semantics])
        if torch.any(_known):
            # special edge case for updating with only a single point, np interprets true as index 1
            if len(_known) == 1:
                # if we directly use 0 as the index, the return is a scalar, so we need to use a slice
                _known = slice(0, 1)
            self.volumetric_cost.sdf_voxels[positions[_known]] = torch.tensor(semantics[_known].astype(float),
                                                                              dtype=self.dtype,
                                                                              device=self.device)

    def undo_update(self, num_pts):
        """
        Undo the latest update
        :param num_pts: number of points to remove, starting from the last added ones
        """
        free = self._free[-num_pts:]
        pos = self.positions[-num_pts:]
        # occupied = self._occupied[-num_pts:]
        self.volumetric_cost.free_voxels[pos[free]] = 0

        self._free = self._free[:-num_pts]
        self._occupied = self._occupied[:-num_pts]
        self._known = self._known[:-num_pts]
        self.positions = self.positions[:-num_pts]

        known_sdf_values = self.semantics[self._known].astype(float)
        known_sdf_values = torch.tensor(known_sdf_values, dtype=self.dtype, device=self.device)
        self.volumetric_cost.sdf_voxels = pv.VoxelSet(self.positions[self._known], known_sdf_values)

    def register(self, iterations=1, initial_tsf=None, low_cost_transform_set=None, **kwargs):
        """
        Register the semantic point cloud to the given object SDF
        :param iterations: number of iterations to run
        :param initial_tsf: T_0 initial world to link transform to use for the optimization
        :param low_cost_transform_set: T_l low cost transform set such as from the previous iteration
        :param kwargs: arguments to pass to register_single
        :return: the registration result and all the archive solutions
        """
        res = None
        all_solutions = None
        self.res_history = []
        self.res_init_history = []
        for i in range(iterations):
            res, all_solutions = self.register_single(initial_tsf=initial_tsf,
                                                      low_cost_transform_set=low_cost_transform_set,
                                                      **kwargs)
            self.res_history.append(res)
            self.res_init_history.append(self.res_init)
            world_to_link = registration_util.solution_to_world_to_link_matrix(res)

            # TODO auto select reinitialization policy based on RMSE distribution
            # reinitialize world_to_link around elites
            initial_tsf = chsel.reinitialize_transform_around_elites(world_to_link, res.rmse)

            low_cost_transform_set = all_solutions
        return res, all_solutions

    def register_single(self, initial_tsf=None, low_cost_transform_set=None, batch=30, debug_func_after_sgd_init=None):
        """

        :param initial_tsf: T_0 initial transform to use for the optimization
        :param low_cost_transform_set: T_l low cost transform set such as from the previous iteration
        :param batch: number of transforms to estimate, only used if initial_tsf is not specified
        :param debug_func_after_sgd_init: debug function to call after the initial sgd optimization with the CHSEL
        object passed in
        :return: the registration result (world to link transform matrix) and all the archive solutions
        """
        dim_pts = self.positions.shape[-1]
        known_pts_world_frame = self.positions[self._known]
        if initial_tsf is not None:
            batch = initial_tsf.shape[0]

        initial_tsf = init_random_transform_with_given_init(dim_pts, batch, self.dtype, self.device,
                                                            initial_tsf=initial_tsf)
        initial_tsf = types.SimilarityTransform(initial_tsf[:, :3, :3],
                                                initial_tsf[:, :3, 3],
                                                torch.ones(batch, device=self.device, dtype=self.dtype))

        # feed it the result of SGD optimization
        self.res_init = chsel.sgd.volumetric_registration_sgd(self.volumetric_cost, batch=batch,
                                                              init_transform=initial_tsf)

        # create range based on SGD results (where are good fits)
        # filter outliers out based on RMSE
        T = registration_util.solution_to_world_to_link_matrix(self.res_init)
        archive_range = registration_util.initialize_qd_archive(T, self.res_init.rmse,
                                                                outlier_ratio=self.outlier_rejection_ratio,
                                                                range_sigma=self.archive_range_sigma,
                                                                measure_fn=self._qd_alg_kwargs.get('measure_fn', None))
        logger.info("QD position bins %s %s", self.bins, archive_range)

        if debug_func_after_sgd_init is not None:
            debug_func_after_sgd_init(self)

        # run QD
        self.qd = self.qd_alg(self.volumetric_cost, known_pts_world_frame.repeat(batch, 1, 1),
                              init_transform=initial_tsf,
                              outlier_ratio=self.outlier_rejection_ratio,
                              iterations=self.qd_iterations, num_emitters=1, bins=self.bins,
                              ranges=archive_range, savedir=self.savedir, **self._qd_alg_kwargs)

        # \hat{T}_0
        x = self.qd.get_numpy_x(self.res_init.RTs.R, self.res_init.RTs.T)
        self.qd.add_solutions(x)
        # \hat{T}_l (such as from the previous iteration's get_all_elite_solutions())
        self.qd.add_solutions(low_cost_transform_set)

        res = self.qd.run()

        return res, self.qd.get_all_elite_solutions()


def init_random_transform_with_given_init(m, batch, dtype, device, initial_tsf=None):
    # apply some random initial poses
    if m > 2:
        R = pk.random_rotations(batch, dtype=dtype, device=device)
    else:
        theta = torch.rand(batch, dtype=dtype, device=device) * math.pi * 2
        Rtop = torch.cat([torch.cos(theta).view(-1, 1), -torch.sin(theta).view(-1, 1)], dim=1)
        Rbot = torch.cat([torch.sin(theta).view(-1, 1), torch.cos(theta).view(-1, 1)], dim=1)
        R = torch.cat((Rtop.unsqueeze(-1), Rbot.unsqueeze(-1)), dim=-1)

    init_pose = torch.eye(m + 1, dtype=dtype, device=device).repeat(batch, 1, 1)
    init_pose[:, :m, :m] = R[:, :m, :m]
    if initial_tsf is not None:
        # check if it's given as a batch
        if len(initial_tsf.shape) == 3:
            init_pose = initial_tsf.clone()
        else:
            init_pose[0] = initial_tsf
    return init_pose
