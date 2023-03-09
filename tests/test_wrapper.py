import copy
import os
import numpy as np
import torch
import open3d as o3d

import chsel
import pytorch_volumetric as pv
from pytorch_seed import seed

import logging

TEST_DIR = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def test_chsel_on_drill():
    visualize = True
    d = "cuda" if torch.cuda.is_available() else "cpu"
    seed(1)
    # supposing we have an object mesh (most formats supported) - from https://github.com/eleramp/pybullet-object-models
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "YcbPowerDrill/textured_simple_reoriented.obj"))
    sdf = pv.MeshSDF(obj)

    # get some points in a grid in the object frame (can get points through other means)
    # this is only around one part of the object to simulate the local nature of contacts
    query_range = np.array([
        [0.04, 0.15],
        [0.04, 0.15],
        [0.1, 0.25],
    ])

    coords, pts = pv.get_coordinates_and_points_in_grid(0.005, query_range, device=d)

    # randomly permute points so they are not in order
    pts = pts[torch.randperm(len(pts))]

    sdf_val, sdf_grad = sdf(pts)
    # assuming we only observe surface points, set thresholds for known free and known occupied
    # note that this is the most common case, but you can know non-zero (non-surface) SDF values
    known_free = sdf_val > 0.005
    known_occupied = sdf_val < -0.005
    known_sdf = ~known_free & ~known_occupied

    # randomly downsample (we randomly permutated before) to simulate getting partial observations
    N = 100
    # retrieve batch of 30 transforms
    B = 30

    # group and stack the points together
    # note that it will still work even if you don't have all 3 or even 2 classes
    # for example with only known surface points, it degenerates to a form of ICP
    # we also don't need balanced data from each class
    pts_free = pts[known_free][:N]
    pts_occupied = pts[known_occupied][:N]
    pts_sdf = pts[known_sdf][:N]
    sem_free = [chsel.SemanticsClass.FREE for _ in pts_free]
    sem_occupied = [chsel.SemanticsClass.OCCUPIED for _ in pts_occupied]
    sem_sdf = sdf_val[known_sdf][:N]
    # sem_sdf = torch.zeros(len(pts_sdf), device=d)

    positions = torch.cat((pts_free, pts_occupied, pts_sdf))
    semantics = sem_free + sem_occupied + sem_sdf.tolist()

    first_rotate = False

    def rotate_view(vis):
        nonlocal first_rotate
        ctr = vis.get_view_control()
        if first_rotate is False:
            ctr.rotate(0.0, -540.0)
            first_rotate = True
        ctr.rotate(5.0, 0.0)
        return False

    if visualize:
        # plot object and points
        # have to convert the pts to o3d point cloud
        pc_free = o3d.geometry.PointCloud()
        pc_free.points = o3d.utility.Vector3dVector(pts_free.cpu().numpy())
        pc_free.paint_uniform_color([1, 0.706, 0])

        pc_occupied = o3d.geometry.PointCloud()
        pc_occupied.points = o3d.utility.Vector3dVector(pts_occupied.cpu().numpy())
        pc_occupied.paint_uniform_color([0, 0.706, 0])

        pc_sdf = o3d.geometry.PointCloud()
        pc_sdf.points = o3d.utility.Vector3dVector(pts_sdf.cpu().numpy())
        pc_sdf.paint_uniform_color([0, 0.706, 1])

        o3d.visualization.draw_geometries_with_animation_callback([obj._mesh, pc_free, pc_occupied, pc_sdf],
                                                                  rotate_view)
        first_rotate = False

    import pytorch_kinematics as pk

    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), device=d)
    positions = gt_tf.transform_points(positions)

    # visualize the transformed mesh and points
    if visualize:
        link_to_world_gt = gt_tf.get_matrix()[0]
        tf_mesh = copy.deepcopy(obj._mesh).transform(link_to_world_gt.cpu().numpy())

        # only plotting the transformed known SDF points for clarity
        pc_free.points = o3d.utility.Vector3dVector(positions[:N].cpu())
        pc_sdf.points = o3d.utility.Vector3dVector(positions[2 * N:].cpu())
        o3d.visualization.draw_geometries_with_animation_callback([tf_mesh, pc_free, pc_sdf], rotate_view)
        first_rotate = False

    def visualize_transforms(link_to_world):
        # visualize the transformed mesh and points
        if visualize:
            # est_tf = pk.Transform3d(matrix=world_to_link.inverse())
            geo = [tf_mesh, pc_free, pc_sdf]
            for i in range(B):
                tfd_mesh = copy.deepcopy(obj._mesh).transform(link_to_world[i].cpu().numpy())
                # paint tfd_mesh a color corresponding to the index in the batch
                tfd_mesh.paint_uniform_color([i / (B * 2), 0, 1 - i / (B * 2)])
                geo.append(tfd_mesh)
            o3d.visualization.draw_geometries_with_animation_callback(geo, rotate_view)

    registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100, free_voxels_resolution=0.005)
    # with no initial transform guess (starts guesses with random rotations at the origin)
    # we want a set of 30 transforms
    random_init_tsf = pk.Transform3d(pos=torch.randn((B, 3), device=d), rot=pk.random_rotations(B, device=d), device=d)
    random_init_tsf = random_init_tsf.get_matrix()

    # try to use the ground truth transform as the initial guess
    # gt_init = chsel.reinitialize_transform_estimates(B, gt_tf.inverse().get_matrix()[0])
    # res, all_solutions = registration.register(terations=15, batch=B, initial_tsf=gt_init)

    res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=random_init_tsf)
    for i in range(len(registration.res_history)):
        # print the sorted RMSE for each iteration
        print(torch.sort(registration.res_history[i].rmse).values)

        # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
        # get 30 4x4 transform matrix for homogeneous coordinates
        world_to_link = chsel.solution_to_world_to_link_matrix(registration.res_history[i])
        link_to_world = world_to_link.inverse()
        visualize_transforms(link_to_world)

    # manually do single step registrations (iterations=1 essentially)
    """
    world_to_link = chsel.solution_to_world_to_link_matrix(res)
    link_to_world = world_to_link.inverse()
    # can evaluate the cost of each transform
    print(res.rmse)
    visualize_transforms(link_to_world)

    # refine the solutions
    iterations = 10
    for i in range(iterations):
        # reinitialize world_to_link around elites
        world_to_link = chsel.reinitialize_transform_around_elites(world_to_link, res.rmse)

        # assuming the object hasn't moved, use the output of the previous iteration as the initial estimate
        res, all_solutions = registration.register(initial_tsf=world_to_link, batch=B,
                                                   low_cost_transform_set=all_solutions)

        world_to_link = chsel.solution_to_world_to_link_matrix(res)
        link_to_world = world_to_link.inverse()
        # print sorted rmse
        print(torch.sort(res.rmse))
        visualize_transforms(link_to_world)
    """


if __name__ == "__main__":
    test_chsel_on_drill()