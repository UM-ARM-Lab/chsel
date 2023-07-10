import copy
import os
import sys
import numpy as np
import torch
import open3d as o3d

import chsel
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from pytorch_seed import seed

import subprocess
import logging

from chsel.types import SimilarityTransform
from chsel.wrapper import init_random_transform_with_given_init
from timeit import default_timer as timer
import time

TEST_DIR = os.path.dirname(__file__)

logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

visualize = True
visualize_after_each_iteration = True
compare_against_icp = True
# number of transforms to estimate at one time
B = 30
# if record video is true then the visualization will always rotate one rotation to allow for looping gifs
# if it's false then it will be interactable and you can rotate it yourself or close the window
record_video = False
d = "cuda" if torch.cuda.is_available() else "cpu"

# for rotating the visualization the first time we enter to get a better angle
first_rotate = False


def rotate_view(vis):
    global first_rotate
    ctr = vis.get_view_control()
    if first_rotate is False:
        ctr.rotate(0.0, -540.0)
        first_rotate = True
    ctr.rotate(5.0, 0.0)
    return False


def draw_geometries_one_rotation(geometries):
    global first_rotate
    if not record_video:
        o3d.visualization.draw_geometries_with_animation_callback(geometries, rotate_view)
    else:
        # open3d visualize geo non-blocking
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for item in geometries:
            vis.add_geometry(item)

        # unfortunately the rotation isn't specified in terms of angles but in terms of mouse drag units
        orbit_period = 7.5 * 224 / 242

        # run the recording in the background
        time.sleep(0.1)
        recorder = subprocess.Popen(
            ["python", os.path.join(TEST_DIR, "record_video.py"), "Open3D", str(orbit_period)])

        start = timer()
        while True:
            vis.poll_events()
            vis.update_renderer()
            if rotate_view(vis):
                break
            if timer() - start > orbit_period + 0.1:
                break
        first_rotate = False
        recorder.wait()
        logger.info("recording finished")


def test_chsel_on_obj(obj, sdf, positions_obj_frame, semantics):
    _free = torch.tensor([s == chsel.SemanticsClass.FREE for s in semantics])
    _occupied = torch.tensor([s == chsel.SemanticsClass.OCCUPIED for s in semantics])
    _known = ~_free & ~_occupied

    if visualize:
        pts_free = positions_obj_frame[_free]
        pts_occupied = positions_obj_frame[_occupied]
        pts_sdf = positions_obj_frame[_known]
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

        print(f"visualize object mesh, free space points (orange), and known SDF points (blue) (press Q or the close window button to move on)")
        draw_geometries_one_rotation([obj._mesh, pc_free, pc_occupied, pc_sdf])


    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), device=d)
    positions = gt_tf.transform_points(positions_obj_frame)

    # visualize the transformed mesh and points
    if visualize:
        link_to_world_gt = gt_tf.get_matrix()[0]
        tf_mesh = copy.deepcopy(obj._mesh).transform(link_to_world_gt.cpu().numpy())
        pts_free = positions[_free]
        pts_sdf = positions[_known]

        # only plotting the transformed known SDF points for clarity
        pc_free.points = o3d.utility.Vector3dVector(pts_free.cpu())
        pc_sdf.points = o3d.utility.Vector3dVector(pts_sdf.cpu())
        print(f"visualize the transformed object mesh, free space points, and known SDF points (press Q or the close window button to move on)")
        draw_geometries_one_rotation([tf_mesh, pc_free, pc_sdf])

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

            draw_geometries_one_rotation(geo)

    registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100, free_voxels_resolution=0.005)
    # with no initial transform guess (starts guesses with random rotations at the origin)
    # we want a set of 30 transforms
    random_init_tsf = pk.Transform3d(pos=torch.randn((B, 3), device=d), rot=pk.random_rotations(B, device=d), device=d)
    random_init_tsf = random_init_tsf.get_matrix()

    # try to use the ground truth transform as the initial guess
    # gt_init = chsel.reinitialize_transform_estimates(B, gt_tf.inverse().get_matrix()[0])
    # res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=gt_init)
    iterations = 15
    if visualize_after_each_iteration:
        world_to_link = random_init_tsf
        all_solutions = None
        # refine the solutions
        for i in range(iterations):
            print(f"iteration {i}")
            start = timer()
            # assuming the object hasn't moved, use the output of the previous iteration as the initial estimate
            res, all_solutions = registration.register(initial_tsf=world_to_link, batch=B,
                                                       low_cost_transform_set=all_solutions)
            print(f"registration took {timer() - start} seconds")

            world_to_link = chsel.solution_to_world_to_link_matrix(res)
            link_to_world = world_to_link.inverse()
            # print sorted rmse
            print(torch.sort(res.rmse))
            visualize_transforms(link_to_world)

            # reinitialize world_to_link around elites
            world_to_link = chsel.reinitialize_transform_around_elites(world_to_link, res.rmse)
    else:
        res, all_solutions = registration.register(iterations=iterations, batch=B, initial_tsf=random_init_tsf)
        print("Showing each iteration of the registration result (press Q or the close window button to move on)")
        for i in range(len(registration.res_history)):
            # print the sorted RMSE for each iteration
            print(f"iteration {i}")
            print(torch.sort(registration.res_history[i].rmse).values)

            # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
            # get 30 4x4 transform matrix for homogeneous coordinates
            world_to_link = chsel.solution_to_world_to_link_matrix(registration.res_history[i])
            link_to_world = world_to_link.inverse()
            visualize_transforms(link_to_world)

    # try ICP on this problem for comparison
    try:
        if compare_against_icp:
            from pytorch3d.ops.points_alignment import iterative_closest_point

            print("Compare against ICP")
            pcd = obj._mesh.sample_points_uniformly(number_of_points=500)
            model_points_register = torch.tensor(np.asarray(pcd.points), device=d, dtype=torch.float)

            # select initial transform from ground truth or from our random initialization
            # initial_tsf = gt_tf.get_matrix()[0].repeat(B, 1, 1)
            initial_tsf = random_init_tsf
            initial_tsf = SimilarityTransform(initial_tsf[:, :3, :3],
                                              initial_tsf[:, :3, 3],
                                              torch.ones(B, device=d, dtype=model_points_register.dtype))

            # known_sdf_pts represent the partial observation
            known_sdf_pts = positions[_known]
            # test with full observation (1-1 correspondence to the registered points)
            full_model_points_in_world = gt_tf.transform_points(model_points_register)

            for i in range(10):
                res = iterative_closest_point(known_sdf_pts.repeat(B, 1, 1), model_points_register.repeat(B, 1, 1),
                                              init_transform=initial_tsf,
                                              allow_reflection=False)
                world_to_link = chsel.solution_to_world_to_link_matrix(res, invert_rot_matrix=True)
                visualize_transforms(world_to_link.inverse())
                # refine initial_tsf using elites
                # doesn't get much better
                initial_tsf = chsel.reinitialize_transform_around_elites(world_to_link, res.rmse)
                initial_tsf = SimilarityTransform(initial_tsf[:, :3, :3],
                                                  initial_tsf[:, :3, 3],
                                                  torch.ones(B, device=d, dtype=model_points_register.dtype))
    except ImportError:
        print("Install pytorch3d to run ICP")

    # manually do single step registrations (iterations=1 essentially)



def test_chsel_on_drill(rng_seed=3):
    seed(rng_seed)
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
    test_chsel_on_obj(obj, sdf, positions, semantics)


if __name__ == "__main__":
    rng_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Using random seed {rng_seed}; try running with different seeds to see different results with\npython tests/test_wrapper.py <seed>")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU which may be much slower")
    test_chsel_on_drill(rng_seed)
