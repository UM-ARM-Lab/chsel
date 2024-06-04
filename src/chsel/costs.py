import matplotlib.colors, matplotlib.cm
import torch
import typing

import pytorch_volumetric as pv
from chsel.registration_util import apply_similarity_transform
from typing import Any
import logging

logger = logging.getLogger(__name__)


class RegistrationCost:
    def __call__(self, R, T, s, other_info=None):
        """Cost for given pose guesses of ICP

        :param R: B x 3 x 3
        :param T: B x 3
        :param s: B
        :return: scalar that is the mean of the costs
        """
        return 0

    def visualize(self, R, T, s):
        pass

    @property
    def last_call_info(self):
        return {}


class ComposeCost(RegistrationCost):
    def __init__(self, *args):
        self.costs = args

    def __call__(self, *args, **kwargs):
        return sum(cost(*args, **kwargs) for cost in self.costs)

    @property
    def last_call_info(self):
        return {f"{i}": cost.last_call_info for i, cost in enumerate(self.costs)}


# Lookup versions of costs do SDF lookups directly which are more expensive but more accurate
# VoxelDiff versions of costs are faster, approximate versions using cached voxel values
class FreeSpaceVoxelDiffCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, world_frame_interior_points: torch.tensor, world_frame_interior_gradients: torch.tensor,
                interior_point_weights: torch.tensor, world_frame_voxels: pv.VoxelGrid) -> torch.tensor:
        # interior points should not be occupied
        # B x Nfree
        occupied = world_frame_voxels[world_frame_interior_points]
        # voxels should be 1 where it is known free space, otherwise 0
        # interior point weights > 0 (negative of their SDF value)
        loss = occupied * interior_point_weights
        loss = loss.sum(dim=-1)
        ctx.occupied = occupied
        ctx.save_for_backward(world_frame_interior_gradients, interior_point_weights)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # need to output the gradient of the loss w.r.t. all the inputs of forward
        dl_dpts = None
        dl_dgrad = None
        dl_dweights = None
        dl_dvoxels = None
        if ctx.needs_input_grad[0]:
            world_frame_interior_gradients, interior_point_weights = ctx.saved_tensors
            # SDF grads point away from the surface; in this case we want to move the surface away from the occupied
            # free space point, so the surface needs to go in the opposite direction
            grads = ctx.occupied[:, :, None] * world_frame_interior_gradients * interior_point_weights[None, :, None]
            # TODO consider averaging the gradients out across all points?
            dl_dpts = grad_outputs[:, None, None] * grads

        # gradients for the other inputs not implemented
        return dl_dpts, dl_dgrad, dl_dweights, dl_dvoxels


class FreeSpaceLookupCost(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, sdf: pv.ObjectFrameSDF,
                model_frame_free_pos: torch.tensor, surface_threshold=0.01) -> torch.tensor:
        # alpha < 0 to tolerate some amount of penetration to compensate for resolution and precision issues
        # surface_threshold = -alpha
        interior_threshold = -surface_threshold
        definitely_not_violating = sdf.outside_surface(model_frame_free_pos,
                                                       surface_level=interior_threshold)
        violating = ~definitely_not_violating
        # this full lookup is much, much slower than the cached version with points, but are about equivalent
        sdf_value, sdf_grad = sdf(model_frame_free_pos[violating])
        # interior points will have sdf_value < 0
        loss = torch.zeros(model_frame_free_pos.shape[:-1], dtype=model_frame_free_pos.dtype,
                           device=model_frame_free_pos.device)
        violation = interior_threshold - sdf_value
        loss[violating] = violation
        ctx.save_for_backward(violating, violation, sdf_value, sdf_grad)
        # returns Batch x Points; sum over dim=-1 to get per transform
        return loss

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # need to output the gradient of the loss w.r.t. all the inputs of forward
        dl_dsdf = None
        dl_dvoxels = None
        dl_dthreshold = None
        if ctx.needs_input_grad[1]:
            violating, violation, sdf_value, sdf_grad = ctx.saved_tensors
            # SDF grads point away from the surface; in this case we want to move the surface away from the occupied
            grads = torch.zeros(list(violating.shape) + [3], dtype=grad_outputs.dtype, device=grad_outputs.device)
            # free space point, so the surface needs to go in the opposite direction
            grads[violating] = violation.unsqueeze(-1) * sdf_grad
            dl_dvoxels = grad_outputs[:, :, None] * -grads

        # gradients for the other inputs not implemented
        return dl_dsdf, dl_dvoxels, dl_dthreshold


class OccupiedLookupCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, sdf: pv.ObjectFrameSDF,
                model_frame_occ_pos: torch.tensor, surface_threshold=0.01) -> torch.tensor:
        interior_threshold = - surface_threshold
        violating = sdf.outside_surface(model_frame_occ_pos, surface_level=surface_threshold)
        # this full lookup is much, much slower than the cached version with points, but are about equivalent
        sdf_value, sdf_grad = sdf(model_frame_occ_pos[violating])
        # interior points will have sdf_value < 0
        loss = torch.zeros(model_frame_occ_pos.shape[:-1], dtype=model_frame_occ_pos.dtype,
                           device=model_frame_occ_pos.device)
        violation = -interior_threshold + sdf_value
        loss[violating] = violation
        ctx.save_for_backward(violating, violation, sdf_value, sdf_grad)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # need to output the gradient of the loss w.r.t. all the inputs of forward
        dl_dsdf = None
        dl_dvoxels = None
        dl_dthreshold = None
        if ctx.needs_input_grad[1]:
            violating, violation, sdf_value, sdf_grad = ctx.saved_tensors
            # SDF grads point away from the surface; in this case we want to move the surface away from the occupied
            grads = torch.zeros(list(violating.shape) + [3], dtype=grad_outputs.dtype, device=grad_outputs.device)
            # free space point, so the surface needs to go in the opposite direction
            grads[violating] = violation.unsqueeze(-1) * sdf_grad
            dl_dvoxels = grad_outputs[:, :, None] * grads

        # gradients for the other inputs not implemented
        return dl_dsdf, dl_dvoxels, dl_dthreshold


class KnownSDFLookupCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, sdf: pv.ObjectFrameSDF, model_frame_positions: torch.tensor,
                expected_sdf_values: torch.tensor, diff_tolerance=0) -> torch.tensor:
        # should be fast since they should be in cache
        sdf_value, sdf_grad = sdf(model_frame_positions)
        diff = sdf_value - expected_sdf_values

        # allow for up to diff_tolerance of diff
        if diff_tolerance > 0:
            pos = diff > 0
            diff[pos] = torch.clamp(diff[pos] - diff_tolerance, min=0)
            diff[~pos] = torch.clamp(diff[~pos] + diff_tolerance, max=0)

        # interior points will have sdf_value < 0
        ctx.save_for_backward(diff, sdf_grad)
        # return Batch x Points shape; caller should sum over points
        return diff.abs()

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # need to output the gradient of the loss w.r.t. all the inputs of forward
        dl_dsdf = None
        dl_dx = None
        dl_dv = None
        dl_thresh = None
        if ctx.needs_input_grad[1]:
            diff, sdf_grad = ctx.saved_tensors
            grads = sdf_grad * diff.unsqueeze(-1)
            dl_dx = grad_outputs[:, :, None] * grads

        # gradients for the other inputs not implemented
        return dl_dsdf, dl_dx, dl_dv, dl_thresh


class KnownSDFVoxelDiffCost:
    @staticmethod
    def apply(all_points: torch.tensor, all_point_weights: torch.tensor,
              known_voxel_centers: torch.tensor, known_voxel_values: torch.tensor, epsilon=0.01) -> torch.tensor:
        # all_points and known_voxel_centers must be in the same frame (world or model frame)
        # difference between current SDF value at each point and the desired one
        # M number of all SDF voxels
        # Nknown number of points with known SDF values
        # M x Nknown
        sdf_diff = torch.cdist(all_point_weights.view(-1, 1), known_voxel_values.view(-1, 1))

        # vector from each known voxel's center with known value to each point
        # B x M x Nknown x 3
        known_voxel_to_pt = all_points.unsqueeze(-2) - known_voxel_centers
        # B x M x Nknown
        known_voxel_to_pt_dist = known_voxel_to_pt.norm(dim=-1)

        # loss for each point, corresponding to each known voxel center
        # only consider points with sdf_diff less than epsilon between desired and model (take the level set)
        mask = sdf_diff < epsilon
        # ensure we have at least one element included in the mask
        while torch.any(mask.sum(dim=0) == 0):
            epsilon *= 2
            mask = sdf_diff < epsilon
        # low distance should have low difference
        # remove those not masked from contention
        known_voxel_to_pt_dist[:, ~mask] = 10000

        loss = known_voxel_to_pt_dist
        # each point may not satisfy two targets simulatenously, so we just care about the best one
        # B x Nknown
        loss = loss.min(dim=1).values
        loss = loss.mean(dim=-1)

        return loss


class VolumetricCost(RegistrationCost):
    """Cost of transformed model pose intersecting with known freespace voxels"""

    def __init__(self, free_voxels: pv.Voxels, sdf_voxels: pv.Voxels,
                 obj_sdf: pv.ObjectFrameSDF,
                 surface_threshold=0.01,
                 occ_voxels: pv.Voxels = None,
                 # cost scales
                 scale=1, scale_known_freespace=1., scale_known_sdf=1., scale_known_occ=0,
                 device='cpu', dtype=torch.float,
                 # for some cost approximations for acceleration
                 query_voxel_grid: typing.Optional[pv.VoxelGrid] = None,
                 # for debugging only
                 vis=None, obj_factory=None,
                 debug=False, debug_known_sgd=False, debug_freespace=False):
        """
        :param free_voxels: representation of freespace
        :param sdf_voxels: voxels for which we know the exact SDF values for
        :param obj_sdf: signed distance function of the target object in object frame
        :param surface_threshold: alpha in meters the tolerance for penetration
        :param scale:
        """

        self.device = device
        self.dtype = dtype

        self.free_voxels = free_voxels
        self.sdf_voxels = sdf_voxels
        self.occ_voxels = occ_voxels

        self.scale = scale
        self.scale_known_freespace = scale_known_freespace
        self.scale_known_sdf = scale_known_sdf
        self.scale_known_occ = scale_known_occ

        # SDF gives us a volumetric representation of the target object
        self.sdf = obj_sdf
        self.surface_threshold = surface_threshold

        # batch
        self.B = None

        # intermediate products for visualization purposes
        self._last_call_info = {}
        self._pts_interior = None
        self._grad = None

        self._pts_all = None
        self.debug = debug
        self.debug_known_sgd = debug_known_sgd
        self.debug_freespace = debug_freespace

        self.vis = vis
        self.obj_factory = obj_factory

        # model points for inverting the lookup of freespace cost
        self.model_interior_points_orig = None
        self.model_interior_normals_orig = None
        self.model_interior_weights = None
        self.model_all_points = None
        self.model_all_weights = None
        self.model_all_normals = None
        self.model_interior_points = None
        self.model_interior_normals = None
        self.init_model_points(query_voxel_grid=query_voxel_grid)

    @property
    def last_call_info(self):
        return self._last_call_info

    def init_model_points(self, query_voxel_grid):
        # ---- for +, known free space points, we just have to care about interior points of the object
        # to facilitate comparison between volumes that are rotated, we sample points at the center of the object voxels
        interior_threshold = -self.surface_threshold
        bb = self.sdf.surface_bounding_box(padding=0.1).cpu().numpy()
        if query_voxel_grid is None:
            query_voxel_grid = pv.VoxelGrid(self.surface_threshold or 0.01,
                                            bb,
                                            dtype=self.dtype, device=self.device)

        self.model_interior_points_orig = self.sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < interior_threshold,
                                                                       voxels=query_voxel_grid)
        if self.model_interior_points_orig.shape[0] == 0:
            raise RuntimeError("Something's wrong with the SDF since there are no interior points")

        self.model_interior_weights, self.model_interior_normals_orig = self.sdf(self.model_interior_points_orig)
        self.model_interior_weights *= -1

        # for some reason the above filtering query can sometimes be wrong
        valid = self.model_interior_weights > 0
        self.model_interior_points_orig = self.model_interior_points_orig[valid]
        self.model_interior_normals_orig = self.model_interior_normals_orig[valid]
        self.model_interior_weights = self.model_interior_weights[valid]

        self.model_all_points = self.sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < self.surface_threshold,
                                                             voxels=query_voxel_grid)
        self.model_all_weights, self.model_all_normals = self.sdf(self.model_all_points)

        if self.model_interior_points_orig.shape[0] < 25:
            logger.warning(
                f"Only {self.model_interior_points_orig.shape[0]} interior points for the model; consider decreasing "
                f"the surface threshold {self.surface_threshold} such as by decreasing resolution; "
                f"the object bounding box is {bb}")
        if self.model_interior_points_orig.shape[0] == self.model_all_points.shape[0]:
            raise RuntimeError("The voxelgrid to query points is too small and only interior points have been "
                               "extracted. Resolve this by increasing the range the voxel grid is over")

    def build_model_points(self, R, T, s):
        self.B = R.shape[0]
        self.model_interior_points = self.model_interior_points_orig.repeat(self.B, 1, 1)
        self.model_interior_normals = self.model_interior_normals_orig.repeat(self.B, 1, 1)

    def __call__(self, R, T, s, other_info=None):
        self._last_call_info = {}
        # assign batch and reuse for later for efficiency
        if self.B is None or self.B != R.shape[0]:
            self.build_model_points(R, T, s)

        # voxels are in world frame
        # need points transformed into world frame
        # transform the points via the given similarity transformation parameters, then evaluate their occupancy
        # should transform the interior points from link frame to world frame
        self._transform_model_to_world_frame(R, T, s)
        # self.visualize(R, T, s)

        loss = torch.zeros(self.B, device=self.device, dtype=self.dtype)

        if self.scale_known_freespace != 0:
            loss += self._cost_freespace(R, T, s) * self.scale_known_freespace
        if self.scale_known_occ != 0:
            loss += self._cost_occ(R, T, s) * self.scale_known_occ
        if self.scale_known_sdf != 0:
            loss += self._cost_sdf(R, T, s) * self.scale_known_sdf

        return loss * self.scale

    def _cost_freespace(self, R, T, s):
        known_free_space_loss = FreeSpaceVoxelDiffCost.apply(self._pts_interior, self._grad,
                                                             self.model_interior_weights,
                                                             self.free_voxels)
        self._last_call_info['unscaled_known_free_space_loss'] = known_free_space_loss
        self._last_call_info["per_point_free_loss"] = None
        return known_free_space_loss

    def _cost_sdf(self, R, T, s):
        known_sdf_voxel_centers, known_sdf_voxel_values = self.sdf_voxels.get_known_pos_and_values()
        known_sdf_loss = KnownSDFVoxelDiffCost.apply(self._pts_all, self.model_all_weights,
                                                     known_sdf_voxel_centers, known_sdf_voxel_values)
        self._last_call_info['unscaled_known_sdf_loss'] = known_sdf_loss
        return known_sdf_loss

    def _cost_occ(self, R, T, s):
        world_frame_occ_voxels, known_occ = self.occ_voxels.get_known_pos_and_values()
        world_frame_occ_voxels = world_frame_occ_voxels[known_occ.view(-1) == 1]
        model_frame_occ_voxels = self._transform_world_frame_points_to_model_frame(R, T, s,
                                                                                   world_frame_occ_voxels)
        known_occ_loss = OccupiedLookupCost.apply(self.sdf, model_frame_occ_voxels, self.surface_threshold)
        known_occ_loss_per_tf = known_occ_loss.sum(dim=-1)
        self._last_call_info["unscaled_known_occ_loss"] = known_occ_loss_per_tf
        self._last_call_info["per_point_occ_loss"] = known_occ_loss
        return known_occ_loss_per_tf

    def _transform_model_to_world_frame(self, R, T, s):
        Rt = R.transpose(-1, -2)
        tt = (-Rt @ T.reshape(-1, 3, 1)).squeeze(-1)
        self._pts_interior = apply_similarity_transform(self.model_interior_points, Rt, tt, s)
        self._pts_all = apply_similarity_transform(self.model_all_points, Rt, tt, s)
        if self.debug and self._pts_interior.requires_grad:
            self._pts_interior.retain_grad()
            self._pts_all.retain_grad()
        self._grad = apply_similarity_transform(self.model_interior_normals, Rt)

    @staticmethod
    def _transform_world_frame_points_to_model_frame(R, T, s, points):
        return apply_similarity_transform(points, R, T, s)

    def visualize(self, R, T, s):
        if not self.debug:
            return
        if self.vis is not None:
            if self._pts_interior is None:
                self._transform_model_to_world_frame(R, T, s)
            with torch.no_grad():
                # occupied = self.voxels.voxels.raw_data > 0
                # indices = occupied.nonzero()
                # coord = self.voxels.voxels.ensure_value_key(indices)
                # for i, xyz in enumerate(coord):
                #     self.vis.draw_point(f"free.{i}", xyz, color=(1, 0, 0), scale=5)
                point_grads = self._pts_all.grad
                have_gradients = point_grads.sum(dim=-1).sum(dim=-1) != 0

                batch_with_gradients = have_gradients.nonzero()
                for b in batch_with_gradients:
                    b = b.item()
                    # only visualize it for one sample at a time
                    if b != 0:
                        continue

                    i = 0

                    # select what to visualize
                    if self.debug_known_sgd:
                        # visualize the known SDF loss directly
                        known_voxel_centers, known_voxel_values = self.sdf_voxels.get_known_pos_and_values()
                        world_frame_all_points = self._pts_all
                        all_point_weights = self.model_all_weights
                        # difference between current SDF value at each point and the desired one
                        sdf_diff = torch.cdist(all_point_weights.view(-1, 1), known_voxel_values.view(-1, 1))
                        # vector from each known voxel's center with known value to each point
                        known_voxel_to_pt = world_frame_all_points.unsqueeze(-2) - known_voxel_centers
                        known_voxel_to_pt_dist = known_voxel_to_pt.norm(dim=-1)

                        epsilon = 0.01
                        # loss for each point, corresponding to each known voxel center
                        # only consider points with sdf_diff less than epsilon between desired and model (take the level set)
                        mask = sdf_diff < epsilon
                        # ensure we have at least one element included in the mask
                        while torch.any(mask.sum(dim=0) == 0):
                            epsilon *= 2
                            mask = sdf_diff < epsilon
                        # low distance should have low difference
                        # remove those not masked from contention
                        known_voxel_to_pt_dist[:, ~mask] = 10000

                        loss = known_voxel_to_pt_dist

                        # closest of each known SDF; try to satisfy each target as best as possible
                        min_values, min_idx = loss.min(dim=1)

                        # just visualize the first one
                        min_values = min_values[b]
                        min_idx = min_idx[b]
                        for k in range(len(min_values)):
                            self.vis.draw_point(f"to_match", known_voxel_centers[k], color=(1, 0, 0), length=0.003,
                                                scale=10)
                            self.vis.draw_point(f"closest", world_frame_all_points[b, min_idx[k]], color=(0, 1, 0),
                                                length=0.003, scale=10)
                            self.vis.draw_2d_line(f"closest_grad",
                                                  self._pts_all[b, min_idx[k]],
                                                  -self._pts_all.grad[b, min_idx[k]],
                                                  color=(0, 1, 0), size=2., scale=5)

                            # draw all losses corresponding to this to match voxel and color code their values
                            each_loss = loss[0, :, k].detach().cpu()
                            # draw all the masked out ones
                            each_loss[each_loss > 1000] = 0

                            error_norm = matplotlib.colors.Normalize(vmin=0, vmax=each_loss.max())
                            color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
                            rgb = color_map.to_rgba(each_loss.reshape(-1))
                            rgb = rgb[:, :-1]

                            for i in range(world_frame_all_points[0].shape[0]):
                                self.vis.draw_point(f"each_loss_pt.{i}", world_frame_all_points[0, i], color=rgb[i],
                                                    length=0.003 if each_loss[i] > 0 else 0.0001)

                            print(min_values[k])
                    elif self.debug_freespace:
                        # draw all points that are in violation larger
                        # interior points should not be occupied
                        occupied = self.free_voxels[self._pts_interior]
                        for i, pt in enumerate(self._pts_interior[b]):
                            self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1),
                                                length=0.005 if occupied[b, i] else 0.0005,
                                                scale=4 if occupied[b, i] else 1)
                            # gradient descend goes along negative gradient so best to show the direction of movement
                            self.vis.draw_2d_line(f"mingrad.{i}", pt, -self._pts_interior.grad[b, i], color=(0, 1, 0),
                                                  size=5.,
                                                  scale=10)
                    else:
                        # visualize all points and losses
                        for i, pt in enumerate(self._pts_all[b]):
                            self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
                            # gradient descend goes along negative gradient so best to show the direction of movement
                            self.vis.draw_2d_line(f"mingrad.{i}", pt, -self._pts_all.grad[b, i], color=(0, 1, 0),
                                                  size=5.,
                                                  scale=10)
                        for j, pt in enumerate(self._pts_interior[b]):
                            self.vis.draw_2d_line(f"intgrad.{j}", pt, -self._pts_interior.grad[b, j], color=(0, 1, 0),
                                                  size=5.,
                                                  scale=10)

                        self.vis.clear_visualization_after("mipt", i + 1)
                        self.vis.clear_visualization_after("min", i + 1)
                        self.vis.clear_visualization_after("mingrad", i + 1)
                        self.vis.clear_visualization_after("intgrad", j + 1)


class VolumetricDirectSDFCost(VolumetricCost):
    """Use SDF queries for the known SDF points instead of using cached grads"""

    def __init__(self, *args, known_sdf_tolerance=0., **kwargs):
        self.known_sdf_tolerance = known_sdf_tolerance
        super().__init__(*args, **kwargs)

    def _cost_sdf(self, R, T, s):
        world_frame_known_sdf_voxels, known_sdf_values = self.sdf_voxels.get_known_pos_and_values()
        known_sdf_model_frame = self._transform_world_frame_points_to_model_frame(R, T, s,
                                                                                  world_frame_known_sdf_voxels)

        known_sdf_loss = KnownSDFLookupCost.apply(self.sdf, known_sdf_model_frame, known_sdf_values,
                                                  self.known_sdf_tolerance)
        known_sdf_loss_per_tf = known_sdf_loss.sum(dim=-1)
        self._last_call_info["unscaled_known_sdf_loss"] = known_sdf_loss_per_tf
        self._last_call_info["per_point_sdf_loss"] = known_sdf_loss
        return known_sdf_loss_per_tf


class VolumetricDoubleDirectCost(VolumetricDirectSDFCost):
    """Cost of transformed model pose intersecting with known freespace voxels
    (slower than the voxelized version above)"""

    def init_model_points(self, query_voxel_grid):
        pass

    def build_model_points(self, R, T, s):
        self.B = R.shape[0]

    def _transform_model_to_world_frame(self, R, T, s):
        # no model points needed
        pass

    def _cost_freespace(self, R, T, s):
        world_frame_free_voxels, known_free = self.free_voxels.get_known_pos_and_values()
        world_frame_free_voxels = world_frame_free_voxels[known_free.view(-1) == 1]
        model_frame_free_voxels = self._transform_world_frame_points_to_model_frame(R, T, s,
                                                                                    world_frame_free_voxels)
        known_free_space_loss = FreeSpaceLookupCost.apply(self.sdf, model_frame_free_voxels, self.surface_threshold)
        known_free_loss_per_tf = known_free_space_loss.sum(dim=-1)
        self._last_call_info["unscaled_known_free_space_loss"] = known_free_loss_per_tf
        self._last_call_info["per_point_free_loss"] = known_free_space_loss
        return known_free_loss_per_tf


class DiscreteNondifferentiableCost(RegistrationCost):
    """Flat high cost for any known free space point violations"""

    def __init__(self, free_voxels: pv.Voxels, sdf_voxels: pv.Voxels,
                 obj_sdf: pv.ObjectFrameSDF, scale=1,
                 vis=None, cmax=20., penetration_tolerance=0.01,
                 obj_factory=None):
        """
        :param free_voxels: representation of freespace
        :param sdf_voxels: voxels for which we know the exact SDF values for
        :param obj_sdf: signed distance function of the target object in object frame
        :param scale:
        """

        self.free_voxels = free_voxels
        self.sdf_voxels = sdf_voxels

        self.scale = scale
        self.scale_known_freespace = cmax
        self.scale_known_sdf = 1
        self.penetration_tolerance = penetration_tolerance

        # SDF gives us a volumetric representation of the target object
        self.sdf = obj_sdf

        # batch
        self.B = None

        self.vis = vis
        self.obj_factory = obj_factory

        self._last_call_info = {}

    @property
    def last_call_info(self):
        return self._last_call_info

    def __call__(self, R, T, s, other_info=None):
        self._last_call_info = {}
        # assign batch and reuse for later for efficiency
        if self.B is None or self.B != R.shape[0]:
            self.B = R.shape[0]

        loss = torch.zeros(self.B, device=R.device, dtype=R.dtype)

        # voxels are in world frame, translate them to model frame
        if self.scale_known_freespace != 0:
            world_frame_free_voxels, known_free = self.free_voxels.get_known_pos_and_values()
            world_frame_free_voxels = world_frame_free_voxels[known_free.view(-1) == 1]
            model_frame_free_voxels = self._transform_world_frame_points_to_model_frame(R, T, s,
                                                                                        world_frame_free_voxels)
            # interior points should not be occupied, set some threshold for interior points
            free_voxels_in_free_space = self.sdf.outside_surface(model_frame_free_voxels, -self.penetration_tolerance)
            occupied = ~free_voxels_in_free_space
            # interior points will have sdf_value < 0
            known_free_space_loss = occupied
            known_free_space_loss = known_free_space_loss.sum(dim=-1) / occupied.shape[-1]
            loss += known_free_space_loss * self.scale_known_freespace
            self._last_call_info["unscaled_known_free_space_loss"] = known_free_space_loss
        if self.scale_known_sdf != 0:
            world_frame_known_sdf_voxels, known_sdf_values = self.sdf_voxels.get_known_pos_and_values()
            model_frame_known_sdf_voxels = self._transform_world_frame_points_to_model_frame(R, T, s,
                                                                                             world_frame_known_sdf_voxels)
            sdf_values, _ = self.sdf(model_frame_known_sdf_voxels)

            known_sdf_loss = (sdf_values - known_sdf_values).abs()
            known_sdf_loss = known_sdf_loss.mean(dim=-1)
            loss += known_sdf_loss * self.scale_known_sdf
            self._last_call_info["unscaled_known_sdf_loss"] = known_sdf_loss

        return loss * self.scale

    def _transform_world_frame_points_to_model_frame(self, R, T, s, points):
        return apply_similarity_transform(points, R, T, s)


class ICPPoseCostMatrixInputWrapper:
    def __init__(self, cost: RegistrationCost, action_cost_scale=1.0):
        self.cost = cost
        self.action_cost_scale = action_cost_scale

    def __call__(self, H, dH=None):
        N = H.shape[0]
        H = H.view(N, 4, 4)
        R = H[:, :3, :3]
        T = H[:, :3, 3]
        s = torch.ones(N, dtype=T.dtype, device=T.device)
        state_cost = self.cost.__call__(R, T, s, None)
        action_cost = torch.norm(dH, dim=1) if dH is not None else 0
        return state_cost + action_cost * self.action_cost_scale
