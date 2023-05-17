## CHSEL: Producing Diverse Plausible Pose Estimates from Contact and Free Space Data

![combined demo](https://i.imgur.com/T4DnKDu.gif)

Demo CHSEL results on drill with sparse contact data (top) and mug with dense vision data (bottom).

## Installation
```bash
pip install chsel
```

For development, clone repository somewhere, then `pip3 install -e .` to install in editable mode.

## Links
- [arxiv](https://arxiv.org/abs/2305.08042)
- [website](https://johnsonzhong.me/projects/chsel/)

## Citation
If you use it, please cite

```bibtex
@inproceedings{zhong2023chsel,
  title={CHSEL: Producing Diverse Plausible Pose Estimates from Contact and Free Space Data},
  author={Zhong, Sheng and Fazeli, Nima and Berenson, Dmitry},
  booktitle={Robotics science and systems},
  year={2023}
}
```

To reproduce the results from the paper, see the 
[experiments repository](https://github.com/UM-ARM-Lab/chsel_experiments).

## Usage
CHSEL registers an observed semantic point cloud against a target object's signed distance field (SDF).
It is agnostic to how the semantic point cloud is obtained, which can come from cameras and tactile sensors for example.

Example code is given in `tests/test_wrapper.py`.

First you need an object frame SDF which you can generate from its 3D mesh
```python
import pytorch_volumetric as pv

# supposing we have an object mesh (most formats supported) - from https://github.com/eleramp/pybullet-object-models
obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")
sdf = pv.MeshSDF(obj)
```
or for (much) faster queries at the cost of some accuracy and memory usage with a cached SDF
```python
import pytorch_volumetric as pv

obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")
sdf = pv.MeshSDF(obj)
# caching the SDF via a voxel grid to accelerate queries
cached_sdf = pv.CachedSDF('drill', resolution=0.01, range_per_dim=obj.bounding_box(padding=0.1), gt_sdf=sdf)
```

Next you need the semantic point cloud which for each point has a position (numpy array or torch tensor)
and semantics (`chsel.Semantics.FREE`, `chsel.Semantics.OCCUPIED`, or `float`).
This is normally observed through various means, but for this tutorial we will generate it from the SDF.
Note that you don't need points from each semantics, but the more information you provide the better the estimate.
For example, you can use CHSEL even with only known surface points. In this case CHSEL will
behave similarly to ICP.

```python
import numpy as np
import torch
import chsel
from pytorch_seed import seed

# get some points in a grid in the object frame (can get points through other means)
# this is only around one part of the object to simulate the local nature of contacts
query_range = np.array([
    [0.04, 0.15],
    [0.04, 0.15],
    [0.1, 0.25],
])

# use CUDA to accelerate if available
d = "cuda" if torch.cuda.is_available() else "cpu"
# seed the RNG for reproducibility
seed(1)

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

positions = torch.cat((pts_free, pts_occupied, pts_sdf))
semantics = sem_free + sem_occupied + sem_sdf.tolist()
```

You can visualize the points with open3d. Orange points are
known free, blue points are known SDF, and green points (inside the object)
are known occupied.
```python
import open3d as o3d

# plot object and points
# have to convert the pts to o3d point cloud
pc_free = o3d.geometry.PointCloud()
pc_free.points = o3d.utility.Vector3dVector(pts_free.numpy())
pc_free.paint_uniform_color([1, 0.706, 0])

pc_occupied = o3d.geometry.PointCloud()
pc_occupied.points = o3d.utility.Vector3dVector(pts_occupied.numpy())
pc_occupied.paint_uniform_color([0, 0.706, 0])

pc_sdf = o3d.geometry.PointCloud()
pc_sdf.points = o3d.utility.Vector3dVector(pts_sdf.numpy())
pc_sdf.paint_uniform_color([0, 0.706, 1])

first_rotate = False

def rotate_view(vis):
    nonlocal first_rotate
    ctr = vis.get_view_control()
    if first_rotate is False:
        ctr.rotate(0.0, -540.0)
        first_rotate = True
    ctr.rotate(5.0, 0.0)
    return False

o3d.visualization.draw_geometries_with_animation_callback([obj._mesh, pc_free, pc_occupied, pc_sdf], rotate_view)
```

![points](https://i.imgur.com/GpQf0w1.gif)

We transform those points from object frame to some random world frame
```python
import pytorch_kinematics as pk

gt_tf = pk.Transform3d(pos=torch.randn(3), rot=pk.random_rotation())
positions = gt_tf.transform_points(positions)
```

We can visualize the transform (see `tests/test_wrapper.py` for more detail)
```python
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

# plot the transformed mesh and points
link_to_world_gt = gt_tf.get_matrix()[0]
tf_mesh = copy.deepcopy(obj._mesh).transform(link_to_world_gt.cpu().numpy())

# only plotting the transformed known SDF points for clarity
pc_free.points = o3d.utility.Vector3dVector(positions[:N].cpu())
pc_sdf.points = o3d.utility.Vector3dVector(positions[2 * N:].cpu())
o3d.visualization.draw_geometries_with_animation_callback([tf_mesh, pc_free, pc_sdf], rotate_view)
```

![transformed points](https://i.imgur.com/aS8oaZO.gif)


We now apply CHSEL for some iterations with an initial random guesses of the transform
```python
import chsel
registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100, free_voxels_resolution=0.005)
# we want a set of 30 transforms
B = 30

random_init_tsf = pk.Transform3d(pos=torch.randn((B, 3), device=d), rot=pk.random_rotations(B, device=d), device=d)
random_init_tsf = random_init_tsf.get_matrix()

res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=random_init_tsf)
```
Starting from a random initial guess, it takes some iterations to converge to plausible estimates,
but you will need fewer iterations (potentially just 1) when warm-starting from the previous estimates
in sequential estimation problems.

We can analyze the results from each iteration and plot the transforms

```python
def visualize_transforms(link_to_world):
    # visualize the transformed mesh and points
    geo = [tf_mesh, pc_free, pc_sdf]
    for i in range(B):
        tfd_mesh = copy.deepcopy(obj._mesh).transform(link_to_world[i].cpu().numpy())
        # paint tfd_mesh a color corresponding to the index in the batch
        tfd_mesh.paint_uniform_color([i / (B * 2), 0, 1 - i / (B * 2)])
        geo.append(tfd_mesh)
    o3d.visualization.draw_geometries_with_animation_callback(geo, rotate_view)

for i in range(len(registration.res_history)):
    # print the sorted RMSE for each iteration
    print(torch.sort(registration.res_history[i].rmse).values)

    # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
    # get 30 4x4 transform matrix for homogeneous coordinates
    world_to_link = chsel.solution_to_world_to_link_matrix(registration.res_history[i])
    link_to_world = world_to_link.inverse()
    visualize_transforms(link_to_world)
```

We gradually refine implausible transforms to plausible ones

![first](https://i.imgur.com/wlnnJJL.gif)

![final](https://i.imgur.com/bxdpgpk.gif)

### Comparison to ICP
We can use [pytorch3d](https://pytorch3d.org/)'s ICP implementation as comparison. We can see that CHSEL is able to
converge to plausible transforms while ICP is not able to improve its estimates via refinement
since it does not use any free space information.

```python
from pytorch3d.ops.points_alignment import iterative_closest_point

pcd = obj._mesh.sample_points_uniformly(number_of_points=500)
model_points_register = torch.tensor(np.asarray(pcd.points), device=d, dtype=torch.float)
initial_tsf = random_init_tsf
initial_tsf = SimilarityTransform(initial_tsf[:, :3, :3],
                                  initial_tsf[:, :3, 3],
                                  torch.ones(B, device=d, dtype=model_points_register.dtype))

# known_sdf_pts represent the partial observation on the object surface
known_sdf_pts = positions[2 * N:]
for i in range(10):
    # note that internally ICP also iterates on its estimate, but refining around the elite estimates may
    # filter out some outliters
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

```

### Plausible Set Estimation
Starting from initially random transform guesses may get stuck in bad local minima (it is reasonably robust to it though),
but if you have access to the ground truth or an approximate transform, you can use it as the initial guess instead.

```python
gt_init = chsel.reinitialize_transform_estimates(B, gt_tf.inverse().get_matrix()[0])
res, all_solutions = registration.register(terations=15, batch=B, initial_tsf=gt_init)
```

### Updating for Sequential Registration
If we have a sequential registration problem where we collect more semantic points, we can
warm start our registration with the previous results:
```python
# assuming positions and semantics have been updated
registration = chsel.CHSEL(sdf, positions, semantics, qd_iterations=100, free_voxels_resolution=0.005)
# feed all solutions from the previous iteration
# assuming the object hasn't moved, use the output of the previous iteration as the initial estimate
# this res is from the last iteration of the previous registration
world_to_link = chsel.solution_to_world_to_link_matrix(res)
res, all_solutions = registration.register(iterations=5, batch=B, initial_tsf=world_to_link,
                                           low_cost_transform_set=all_solutions)
```
