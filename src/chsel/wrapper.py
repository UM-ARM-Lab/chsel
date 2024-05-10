from typing import Sequence, Optional, Union

import math
import chsel
import chsel.measure
import chsel.quality_diversity
import chsel.sgd
import torch
import numpy as np

from chsel import costs
from chsel import types
from chsel import quality_diversity
from chsel import registration_util
from chsel.se2 import project_transform

import pytorch_volumetric as pv
import pytorch_kinematics as pk

import logging

logger = logging.getLogger(__file__)


def ensure_tensor(device, dtype, *args):
    tensors = tuple(
        x.to(device=device, dtype=dtype) if torch.is_tensor(x) else
        torch.tensor(x, device=device, dtype=dtype)
        for x in args)
    return tensors if len(tensors) > 1 else tensors[0]


class CHSEL:
    def __init__(self,
                 obj_sdf: pv.ObjectFrameSDF,
                 positions: torch.tensor, semantics: Sequence[types.Semantics],
                 resolution=None,
                 known_sdf_values: Optional[torch.tensor] = None,
                 cost=costs.VolumetricDirectSDFCost,
                 free_voxels: Optional[pv.Voxels] = None,
                 occupied_voxels: Optional[pv.Voxels] = None,
                 known_sdf_voxels: Optional[pv.Voxels] = None,
                 # parameters for SE2 registration
                 axis_of_rotation=None,
                 offset_along_axis=0,
                 # for removing duplicate points
                 free_voxels_resolution=None,
                 occupied_voxels_resolution=None,
                 sgd_solution_outlier_rejection_ratio=5.0,
                 archive_range_sigma=3,
                 archive_min_size=2e-4,
                 bins=40,
                 do_qd=True,
                 qd_iterations=100,
                 qd_alg=quality_diversity.CMAMEGA,
                 qd_measure: Optional[chsel.measure.MeasureFunction] = None,
                 savedir=registration_util.ROOT_DIR,
                 debug=False,
                 qd_alg_kwargs=None,
                 **cost_kwargs):
        """

        :param obj_sdf: Object centered signed distance field (SDF)
        :param positions: World frame positions of the observed points
        :param semantics: Semantics of the observed points
        :param resolution: Resolution in meters of the side-length of each voxel; this controls what points
        are considered duplicates and removed when calling remove_duplicate_points(); if None is given, it will be
        chosen such that there are at least 20 voxels per dimension of the object
        :param known_sdf_values: Known SDF values of the observed points; if not specified, all known points will be
        assumed to have SDF=0 (surface points)
        :param cost: Class that implements Eq. 6
        :param free_voxels: Explicit specification of free space voxels; if not specified they will be extracted
        from the given points that have the FREE semantics. If the observed points are derived from free voxels,
        it will save recreating them if these are specified directly.
        :param occupied_voxels: Similarly to the above
        :param known_sdf_voxels: Similarly to the above
        :param axis_of_rotation: If given, optimize over SE(2) instead of SE(3) by fixing the rotation around the given axis
        :param offset_along_axis: If axis_of_rotation is given, the SE(2) optimization is over a plane defined by the
        axis_of_rotation and offset from origin along the axis_of_rotation by this value
        :param free_voxels_resolution: Resolution as above; this should scale with
        the object's size; it is a good idea that at least 10 voxels span each dimension of the object. If set to None
        by default, the duplicate resolution will be used
        :param occupied_voxels_resolution: Similarly to the above
        :param sgd_solution_outlier_rejection_ratio: The ratio of CHSEL cost function to the best one found, above which
        to be considered an outlier and rejected. This is used to prevent the archive range being polluted with outliers
        :param archive_range_sigma: b_sigma the number of standard deviations to consider for the archive
        :param archive_min_size: b_min the minimum size of the archive (range of the bin)
        :param bins: How many bins each dimension of the archive will have
        :param do_qd: Whether to do quality diversity optimization
        :param qd_iterations: n_o number of quality diversity optimization iterations
        :param qd_alg: The quality diversity optimization algorithm to use
        :param qd_measure: The function to use for the QD measure. If not specified, the position is used.
        :param savedir: Directory to save loss plots
        :param debug:
        """
        self.obj_sdf = obj_sdf
        self.outlier_rejection_ratio = sgd_solution_outlier_rejection_ratio
        self.resolution = resolution
        if self.resolution is None:
            bb = self.obj_sdf.surface_bounding_box()
            r = bb[:, 1] - bb[:, 0]
            self.resolution = (min(r) / 20).cpu().item()

        self.axis_of_rotation = axis_of_rotation
        self.offset_along_axis = offset_along_axis

        self.dtype = positions.dtype
        self.device = positions.device

        self.positions = positions
        # to allow batch indexing
        # store their numeric values rather than enum for better operations
        self.semantics = ensure_tensor(self.device, torch.long, semantics)

        self.debug = debug
        self.archive_range_sigma = archive_range_sigma
        self.archive_min_size = archive_min_size
        self.bins = bins
        self.do_qd = do_qd
        self.qd_iterations = qd_iterations
        self.qd_alg = qd_alg
        self.savedir = savedir
        # extract indices
        idx = CHSEL.get_separate_semantic_indices(self.semantics)
        self._free = idx['free']
        self._occupied = idx['occupied']
        self._known = idx['known']

        # intermediate results for use outside for debugging
        self.res_init = None
        self.qd = None
        self.res_history = []
        self.res_init_history = []

        self._qd_alg_kwargs = {"measure": qd_measure, }
        if qd_alg_kwargs is not None:
            self._qd_alg_kwargs.update(qd_alg_kwargs)

        # extract the known free voxels from the given free points
        if free_voxels is None:
            free_voxels = pv.ExpandingVoxelGrid(free_voxels_resolution or self.resolution, [(0, 0) for _ in range(3)],
                                                dtype=self.dtype,
                                                device=self.device)
            free_voxels[positions[self._free]] = 1
        # extract the known occupied voxels from the given occupied points
        if occupied_voxels is None:
            occupied_voxels = pv.ExpandingVoxelGrid(occupied_voxels_resolution or self.resolution,
                                                    [(0, 0) for _ in range(3)],
                                                    dtype=self.dtype, device=self.device)
            occupied_voxels[positions[self._occupied]] = 1
        # extract the known SDF points
        if known_sdf_voxels is None:
            if known_sdf_values is None:
                known_sdf_pts = positions[self._known]
                known_sdf_values = torch.zeros(len(known_sdf_pts), dtype=self.dtype, device=self.device)
            known_sdf_voxels = pv.VoxelSet(positions[self._known], known_sdf_values)

        cost_options = {
            "scale_known_freespace": 20,
            "debug": self.debug,
            "surface_threshold": self.resolution
        }
        cost_options.update(cost_kwargs)
        # construct the cost function
        self.volumetric_cost = cost(free_voxels, known_sdf_voxels, self.obj_sdf,
                                    occ_voxels=occupied_voxels,
                                    dtype=self.dtype, device=self.device,
                                    **cost_options)

    @staticmethod
    def get_separate_semantic_indices(semantics):
        free = semantics == chsel.SemanticsClass.FREE.value
        occupied = semantics == chsel.SemanticsClass.OCCUPIED.value
        known = ~free & ~occupied
        return {'free': free, 'occupied': occupied, 'known': known}

    def evaluate_homogeneous(self, H_world_to_link: Union[torch.tensor, pk.Transform3d], use_scale=False):
        """
        Evaluate the discrepency between the observed semantic point cloud and the given transforms (world to link)
        :return: the cost for each transform
        """
        if not torch.is_tensor(H_world_to_link):
            H_world_to_link = H_world_to_link.get_matrix()
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

    def debug_last_cost_call(self):
        return self.volumetric_cost.last_call_info

    def update(self, positions: torch.tensor, semantics: torch.tensor, known_sdf_values=None):
        """
        Update the observed point cloud and semantics
        """
        if positions is None or len(positions) == 0:
            return
        idx = CHSEL.get_separate_semantic_indices(semantics)
        _free = idx['free']
        _occupied = idx['occupied']
        _known = idx['known']
        self._free = torch.cat([self._free, _free])
        self._occupied = torch.cat([self._occupied, _occupied])
        self._known = torch.cat([self._known, _known])
        self.positions = torch.cat([self.positions, positions])
        self.volumetric_cost.free_voxels[positions[_free]] = 1
        self.semantics = torch.cat([self.semantics, semantics])
        if torch.any(_known):
            # special edge case for updating with only a single point, np interprets true as index 1
            if len(_known) == 1:
                # if we directly use 0 as the index, the return is a scalar, so we need to use a slice
                _known = slice(0, 1)
            known_sdf_positions = positions[_known]
            if known_sdf_values is None:
                known_sdf_values = torch.zeros(len(known_sdf_positions), dtype=self.dtype, device=self.device)
            self.volumetric_cost.sdf_voxels[known_sdf_positions] = known_sdf_values

    def reset_free_points(self, still_free_points):
        # don't touch the other semantics
        untouched_semantics = self.semantics[~self._free]
        positions = [self.positions[~self._free], still_free_points]
        semantics = [untouched_semantics, torch.ones(len(still_free_points), dtype=torch.long,
                                                     device=self.device) * chsel.SemanticsClass.FREE.value]

        self.positions = torch.cat(positions)
        self.semantics = torch.cat(semantics).reshape(-1)
        self._build_semantic_indices()
        self._sync_free_points()

    def remove_duplicate_points(self, duplicate_distance=None, range_per_dim=None):
        if duplicate_distance is None:
            duplicate_distance = self.resolution

        if range_per_dim is None:
            range_per_dim = np.stack(
                (self.positions.min(dim=0)[0].cpu().numpy(), self.positions.max(dim=0)[0].cpu().numpy())).T

        all_pts = []
        all_sem = []
        for s in chsel.SemanticsClass:
            # potentially different resolution per semantic class
            if type(duplicate_distance) == dict:
                resolution = duplicate_distance[s]
            else:
                resolution = duplicate_distance
            idx = self.semantics == s.value
            pts = self.positions[idx]
            pts = pv.voxel_down_sample(pts, resolution, range_per_dim)
            all_pts.append(pts)
            all_sem.append(torch.ones(len(pts), dtype=torch.long, device=self.device) * s.value)
        self.positions = torch.cat(all_pts)
        self.semantics = torch.cat(all_sem).reshape(-1)
        self._build_semantic_indices()

        self._sync_sdf_points()
        self._sync_free_points()

    def _build_semantic_indices(self):
        idx = CHSEL.get_separate_semantic_indices(self.semantics)
        self._free = idx['free']
        self._occupied = idx['occupied']
        self._known = idx['known']

    def _sync_free_points(self):
        """Ensure the wrapper's free points are in sync with the volumetric cost function's free points"""
        self.volumetric_cost.free_voxels = pv.ExpandingVoxelGrid(self.volumetric_cost.free_voxels.resolution,
                                                                 [(0, 0) for _ in range(3)], dtype=self.dtype,
                                                                 device=self.device)
        self.volumetric_cost.free_voxels[self.positions[self._free]] = 1

    def _sync_sdf_points(self):
        known_sdf_positions = self.positions[self._known]
        known_sdf_values = torch.zeros(len(known_sdf_positions), dtype=self.dtype, device=self.device)
        self.volumetric_cost.sdf_voxels = pv.VoxelSet(known_sdf_positions, known_sdf_values)

    def register(self, iterations=1, initial_tsf=None, low_cost_transform_set=None, **kwargs):
        """
        Register the semantic point cloud to the given object SDF
        :param iterations: number of iterations to run
        :param initial_tsf: T_0 initial world to link transform to use for the optimization
        :param low_cost_transform_set: T_l low cost transform set such as from the previous iteration
        :param kwargs: arguments to pass to register_single
        :return: the registration result and all the archive solutions
        """
        if initial_tsf is not None and isinstance(initial_tsf, pk.Transform3d):
            initial_tsf = initial_tsf.get_matrix()
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

    def enforce_transform_constraints(self, res):
        if self.axis_of_rotation is not None:
            R, T = project_transform(self.axis_of_rotation, self.offset_along_axis, res.RTs.R, res.RTs.T)
            res = types.ICPSolution(res.converged, res.rmse, res.Xt, types.SimilarityTransform(R, T, res.RTs.s),
                                    res.t_history)
        return res

    def register_single(self, initial_tsf=None, low_cost_transform_set=None, batch=30, debug_func_after_sgd_init=None,
                        skip_qd=False):
        """

        :param initial_tsf: T_0 initial world to link transform to use for the optimization
        :param low_cost_transform_set: T_l low cost transform set such as from the previous iteration
        :param batch: number of transforms to estimate, only used if initial_tsf is not specified
        :param debug_func_after_sgd_init: debug function to call after the initial sgd optimization with the CHSEL
        object passed in
        :param skip_qd: whether to skip the quality diversity optimization
        :return: the registration result (world to link transform matrix) and all the archive solutions
        """
        if initial_tsf is not None and isinstance(initial_tsf, pk.Transform3d):
            initial_tsf = initial_tsf.get_matrix()

        dim_pts = self.positions.shape[-1]
        known_pts_world_frame = self.positions[self._known]
        if initial_tsf is not None:
            batch = initial_tsf.shape[0]

        initial_tsf = init_random_transform_with_given_init(dim_pts, batch, self.dtype, self.device,
                                                            initial_tsf=initial_tsf,
                                                            axis_of_rotation=self.axis_of_rotation)
        initial_tsf = types.SimilarityTransform(initial_tsf[:, :3, :3],
                                                initial_tsf[:, :3, 3],
                                                torch.ones(batch, device=self.device, dtype=self.dtype))

        # feed it the result of SGD optimization
        self.res_init = chsel.sgd.volumetric_registration_sgd(self.volumetric_cost, batch=batch,
                                                              init_transform=initial_tsf,
                                                              axis_of_rotation=self.axis_of_rotation,
                                                              offset_along_axis=self.offset_along_axis, )

        if skip_qd or not self.do_qd:
            self.res_init = self.enforce_transform_constraints(self.res_init)
            return self.res_init, None

        # create range based on SGD results (where are good fits)
        # filter outliers out based on RMSE
        T = registration_util.solution_to_world_to_link_matrix(self.res_init)
        measure_fn = self._qd_alg_kwargs.get('measure', None)
        if measure_fn is None:
            if self.axis_of_rotation is None:
                measure_fn = chsel.measure.PositionMeasure(2, device=self.device, dtype=self.dtype)
            else:
                measure_fn = chsel.measure.SE2AngleMeasure(axis_of_rotation=self.axis_of_rotation,
                                                           offset_along_axis=self.offset_along_axis)
        self._qd_alg_kwargs['measure'] = measure_fn
        archive_range = chsel.quality_diversity.initialize_qd_archive(T, self.res_init.rmse, measure_fn,
                                                                      outlier_ratio=self.outlier_rejection_ratio,
                                                                      range_sigma=self.archive_range_sigma,
                                                                      min_std=self.archive_min_size * 0.5)
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
        x = self.qd.measure.get_numpy_x(self.res_init.RTs.R, self.res_init.RTs.T)
        self.qd.add_solutions(x)
        # \hat{T}_l (such as from the previous iteration's get_all_elite_solutions())
        self.qd.add_solutions(low_cost_transform_set)

        res = self.qd.run()
        res = self.enforce_transform_constraints(res)

        return res, self.qd.get_all_elite_solutions()


def init_random_transform_with_given_init(m, batch, dtype, device, initial_tsf=None, axis_of_rotation=None):
    # apply some random initial poses
    if m > 2 and axis_of_rotation is None:
        R = pk.random_rotations(batch, dtype=dtype, device=device)
    else:
        theta = torch.rand(batch, dtype=dtype, device=device) * math.pi * 2
        if axis_of_rotation is not None:
            # rotate around the given axis
            R = pk.axis_and_angle_to_matrix_33(axis_of_rotation, theta)
        else:
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
