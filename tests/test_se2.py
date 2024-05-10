import copy

import numpy as np

import chsel
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from pytorch_seed import seed
import torch
import os
from chsel.se2 import project_onto_plane, construct_plane_basis, xyz_to_uv, uv_to_xyz
import matplotlib.pyplot as plt
import open3d as o3d
import time
from timeit import default_timer as timer
import subprocess
import logging

# from mpl_toolkits.mplot3d import Axes3D

TEST_DIR = os.path.dirname(__file__)

logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

visualize_after_each_iteration = True
record_video = False
visualize = True
B = 30
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


def plot_plane_and_points(axis, offset, points, plane_size=10):
    # Project points
    projected_points = project_onto_plane(axis, offset, points)

    # Plot original and projected points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points', s=50, c='r')
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], label='Projected Points', s=25,
               c='b')
    # draw line from point to projected point
    for i in range(points.shape[0]):
        ax.plot([points[i, 0], projected_points[i, 0]], [points[i, 1], projected_points[i, 1]],
                [points[i, 2], projected_points[i, 2]], c='g')

    # Create a grid to represent the plane
    xx, yy = torch.meshgrid(torch.linspace(-plane_size, plane_size, 10), torch.linspace(-plane_size, plane_size, 10))
    z = (-axis[1] * yy - axis[0] * xx + offset) * 1. / axis[2]

    # Plot the plane
    ax.plot_surface(xx.numpy(), yy.numpy(), z.numpy(), alpha=0.5)
    # ensure the aspect ratio is correct as otherwise the plane will look skewed
    ax.set_aspect('equal', adjustable='box')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


def test_projection_simple():
    N = 100
    axis = torch.tensor([0, 0, 1], dtype=torch.float32)
    axis_u, axis_v = construct_plane_basis(axis)
    # should be x and y axis
    assert torch.allclose(axis_u, torch.tensor([1, 0, 0], dtype=torch.float32))
    assert torch.allclose(axis_v, torch.tensor([0, 1, 0], dtype=torch.float32))

    offset = 0
    origin = axis * offset
    points = torch.randn(N, 3)
    projected = project_onto_plane(axis, offset, points)
    # check that the points are actually on the plane
    assert projected[:, 2].abs().max() < 1e-6
    assert projected[:, :2].allclose(points[:, :2])

    # convert to uv coordinates and back
    uv = xyz_to_uv(points, origin, axis, axis_u, axis_v)
    points_reconstructed = uv_to_xyz(uv, origin, axis_u, axis_v)
    assert points_reconstructed.allclose(projected)

    if visualize:
        plot_plane_and_points(axis, offset, points, plane_size=3)


def test_projection():
    N = 100
    axis = torch.rand(3)
    axis /= torch.linalg.norm(axis)
    axis_u, axis_v = construct_plane_basis(axis)
    offset = torch.rand(1).item()
    origin = axis * offset
    points = torch.randn(N, 3)
    projected = project_onto_plane(axis, offset, points)
    # convert to uv coordinates and back
    uv = xyz_to_uv(points, origin, axis, axis_u, axis_v)
    points_reconstructed = uv_to_xyz(uv, origin, axis_u, axis_v)
    assert points_reconstructed.allclose(projected, atol=1e-6)

    if visualize:
        plot_plane_and_points(axis, offset, points, plane_size=2)


def test_se2(axis_of_rotation=(0, 0, 1)):
    seed(5)
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "YcbPowerDrill/textured_simple_reoriented.obj"))
    sdf = pv.MeshSDF(obj)
    N = 200

    query_range = np.array([
        [0.01, 0.15],
        [0.01, 0.15],
        [0.05, 0.2],
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

    # group and stack the points together
    # note that it will still work even if you don't have all 3 or even 2 classes
    # for example with only known surface points, it degenerates to a form of ICP
    # we also don't need balanced data from each class
    pts_free = pts[known_free][:N]
    pts_occupied = pts[known_occupied][:N]
    pts_sdf = pts[known_sdf][:N]
    sem_free = [chsel.SemanticsClass.FREE for _ in pts_free]
    sem_occupied = [chsel.SemanticsClass.OCCUPIED for _ in pts_occupied]
    sem_sdf = [chsel.SemanticsClass.SURFACE for _ in pts_sdf]

    positions = torch.cat((pts_free, pts_occupied, pts_sdf))
    semantics = sem_free + sem_occupied + sem_sdf

    _free = torch.tensor([s == chsel.SemanticsClass.FREE for s in semantics])
    _occupied = torch.tensor([s == chsel.SemanticsClass.OCCUPIED for s in semantics])
    _known = ~_free & ~_occupied

    if visualize:
        pts_free = positions[_free]
        pts_occupied = positions[_occupied]
        pts_sdf = positions[_known]
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

        print(
            f"visualize object mesh, free space points (orange), and known SDF points (blue) (press Q or the close window button to move on)")
        draw_geometries_one_rotation([obj._mesh, pc_free, pc_occupied, pc_sdf])
    # we create some random initializations

    random_init_world_to_link = pk.Transform3d(pos=torch.randn((B, 3), device=d), rot=pk.random_rotations(B, device=d),
                                               device=d)

    axis_of_rotation = torch.tensor(axis_of_rotation, device=d, dtype=positions.dtype)
    axis_of_rotation /= torch.linalg.norm(axis_of_rotation)

    R = pk.axis_and_angle_to_matrix_33(axis_of_rotation, torch.tensor(0.5, device=d))
    gt_link_to_world_tf = pk.Rotate(R, device=d)
    gt_link_to_world_tf = gt_link_to_world_tf.translate(0., 0., 0.005)
    gt_world_to_link_tf = gt_link_to_world_tf.inverse()

    positions = gt_link_to_world_tf.transform_points(positions)

    # visualize the transformed mesh and points
    if visualize:
        link_to_world_gt = gt_link_to_world_tf.get_matrix()[0]
        tf_mesh = copy.deepcopy(obj._mesh).transform(link_to_world_gt.cpu().numpy())
        pts_free = positions[_free]
        pts_sdf = positions[_known]

        # only plotting the transformed known SDF points for clarity
        pc_free.points = o3d.utility.Vector3dVector(pts_free.cpu())
        pc_sdf.points = o3d.utility.Vector3dVector(pts_sdf.cpu())
        print(
            f"visualize the transformed object mesh, free space points, and known SDF points (press Q or the close window button to move on)")
        draw_geometries_one_rotation([tf_mesh, pc_free, pc_sdf])

    offset = gt_world_to_link_tf.get_matrix()[0, :3, 3] @ axis_of_rotation
    # try the position measure to enforce position diversity (bound to uv within the SE(2) plane)
    measure_fn = chsel.measure.SE2PositionMeasure(axis_of_rotation=axis_of_rotation, offset_along_axis=offset)
    registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100,
                               axis_of_rotation=axis_of_rotation, offset_along_axis=offset,
                               qd_measure=measure_fn,
                               do_qd=True)

    # alternatively the angular measure enforces rotational diversity (default)
    measure_fn = chsel.measure.SE2AngleMeasure(axis_of_rotation=axis_of_rotation, offset_along_axis=offset)
    registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100,
                               axis_of_rotation=axis_of_rotation, offset_along_axis=offset,
                               archive_range_sigma=1.5, archive_min_size=1e-6,
                               qd_measure=measure_fn,
                               do_qd=True)
    print(registration.resolution)

    # by default, the 2D position measure is used
    # if you know the position quite well, but its rotation is ambiguous, it is advantageous to use rotation measures
    def visualize_transforms(link_to_world):
        # visualize the transformed mesh and points
        if visualize:
            # est_tf = pk.Transform3d(matrix=world_to_link.inverse())
            # create plane at z=0
            plane = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.4, depth=0.01)
            plane = plane.translate((-0.2, -0.2, 0))
            geo = [tf_mesh, pc_free, pc_sdf, plane]
            for i in range(B):
                tfd_mesh = copy.deepcopy(obj._mesh).transform(link_to_world[i].cpu().numpy())
                # paint tfd_mesh a color corresponding to the index in the batch
                tfd_mesh.paint_uniform_color([i / (B * 2), 0, 1 - i / (B * 2)])
                geo.append(tfd_mesh)

            draw_geometries_one_rotation(geo)

    # for example in SE(2), we can evaluate rotation about the axis of rotation
    iterations = 15
    if visualize_after_each_iteration:
        world_to_link = random_init_world_to_link
        all_solutions = None
        # refine the solutions
        for i in range(iterations):
            print(f"iteration {i}")
            start = timer()
            # assuming the object hasn't moved, use the output of the previous iteration as the initial estimate
            res, all_solutions = registration.register(initial_tsf=world_to_link, batch=B,
                                                       low_cost_transform_set=all_solutions)
            print(f"registration took {timer() - start} seconds")
            # print the RMSE for each cost type

            world_to_link = chsel.solution_to_world_to_link_matrix(res)
            link_to_world = world_to_link.inverse()
            # print sorted rmse
            print(torch.sort(res.rmse))
            fl = registration.debug_last_cost_call().get('unscaled_known_free_space_loss')
            sl = registration.debug_last_cost_call().get('unscaled_known_sdf_loss')
            ol = registration.debug_last_cost_call().get('unscaled_known_occ_loss')
            print(f"free space loss {fl} \nsdf loss {sl} \nocc loss {ol}")
            visualize_transforms(link_to_world)

            # reinitialize world_to_link around elites
            world_to_link = chsel.reinitialize_transform_around_elites(world_to_link, res.rmse)
    else:
        res, all_solutions = registration.register(iterations=iterations, batch=B,
                                                   initial_tsf=random_init_world_to_link)
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


def test_se2_measure():
    seed(5)
    # check that the angle wrapping works by evaluating the measure of a group of transforms
    axis_of_rotation = torch.tensor([0, 0, 1], dtype=torch.float32)
    offset = 0
    measure = chsel.measure.SE2AngleMeasure(axis_of_rotation=axis_of_rotation, offset_along_axis=offset)

    N = 100
    x = torch.randn(N, 3)

    # perturb the angle systematically
    perturbation = torch.linspace(0, 2 * np.pi, N)
    range_sigma = 3
    min_std = 1e-4
    diffs = []
    for delta_angel in perturbation:
        delta = torch.cat((delta_angel.repeat(N, 1), torch.zeros(N, 2)), dim=-1)
        x_perturbed = x + delta
        m = measure(x_perturbed)
        m = m.reshape(-1, measure.measure_dim)

        centroid, m_std = measure.compute_moments(m)
        m_std = np.maximum(m_std, min_std)

        ranges = np.array((centroid - m_std * range_sigma, centroid + m_std * range_sigma)).T
        ranges = ranges.reshape(-1, 2)
        diff = ranges[:, 1] - ranges[:, 0]
        diffs.append(diff)
        # logger.info(f"{ranges} {diff.sum()}")
    diffs = np.array(diffs)
    # check that the diffs are about the same
    assert np.allclose(diffs, diffs[0], atol=1e-4)


if __name__ == '__main__':
    test_projection_simple()
    test_projection()
    test_se2_measure()
    test_se2()
