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

## Intuition
### Cost
* enforce an object's signed distance field (SDF) consistency against point clouds augmented with volumetric semantics
    * whether a point is in free space, inside the object, or has a known SDF value 
* two of those semantics classes encodes uncertainty about the observed point's SDF
    * for example free space just means we know the point has SDF > 0, but makes no claim about its actual value
    * in many observation scenarios, we do not know the observed' point's actual SDF value
* compared to related methods like SDF2SDF that match an SDF against another SDF, this avoids bias

### Optimization
At a high level, our method can be summarized as:
1. start with an initial set of transforms that ideally covers all local minima
   * for every local minima, there exists a transform in the initial set that is in its attraction basin in terms of the local cost landscape
3. find the bounds of the local minima landscape
    * usually much smaller than the whole search space
    * do this by performing gradient descent independently on each initial transform with our given cost
4. consider search in a dimensionality reduced feature space
    * translational component of the transform such that we consider the best rotation for each translation
6. find good local minima with a fine-tuning search with the search space reduced to the local minima bounds that we found to afford higher resolution
    * instead of finding the best local minima, we want to evaluate all local minima, and we do this with Quality Diversity optimization on the feature space
    * we maintain an archive, a grid in feature space, each cell of which holds the best solution given that archive (so the grid is in translation space, and each cell holds a full transform)
    * we populate this archive over the course of QD optimization, which evolutionarily combines the top solutions
7. we return the transforms from the best scoring cells, as many as requested
    * usually same number as in the input transform set

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

Easy way to initialize registration (assuming you have observed world-frame points):
```python
import chsel
# can specify None for any of the classes if you don't have them
registration = chsel.CHSEL(sdf, surface_pts=surface_pts, free_pts=free_pts, occupied_pts=None)
```

Alternatively you can specify the points and semantics directly
```python
import chsel
...
sem_free = [chsel.SemanticsClass.FREE for _ in pts_free]
sem_occupied = [chsel.SemanticsClass.OCCUPIED for _ in pts_occupied]
# lookup the SDF value of the points
sem_sdf = sdf_val[known_sdf][:N]
positions = torch.cat((pts_free, pts_occupied, pts_sdf))
semantics = sem_free + sem_occupied + sem_sdf.tolist()
registration = chsel.CHSEL(sdf, positions=positions, semantics=semantics)
```

First you might want to visualize the object interior and surface points to ensure resolutions are fine (need open3d).
Surface points are in green, interior points are in blue. You can adjust them via the `CHSEL` constructor arguments 
`surface_threshold` and `surface_threshold_model_override`. The default value for them is the `resolution` of the registration.
```python
registration.visualize_input(show_model_points=True, show_input_points=False)
```
![model points](https://i.imgur.com/edRkUog.gif)

You can also visualize the observed input points together with the model points.
For them to show up in the same frame, you need the ground truth transform.
Orange points are known free, red points are surface, dark green is occupied.
```python
registration.visualize_input(show_model_points=True, show_input_points=True, gt_obj_to_world_tf=gt_tf)
```
![model and input points](https://i.imgur.com/DD8zSkR.gif)

Also visualized with the object mesh

![points](https://i.imgur.com/GpQf0w1.gif)


We now apply CHSEL for some iterations with an initial random guesses of the transform
```python
import pytorch_kinematics as pk
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

### SE(2) Constraint
If you know the object lies on a table top for example, you can constrain the optimization to be over
SE(2) instead of SE(3). You do this by specifying the plane via the axis of rotation (normal) and a
scalar offset:
```python
# table top assumed to face positive z
normal = torch.tensor([0, 0, 1])
offset = torch.tensor([0, 0, 0])
registration = chsel.CHSEL(sdf, positions, semantics, axis_of_rotation=normal, offset_along_normal=offset)
```

### Measures
QD optimization works by binning over the measure space. This is how diversity is enforced.
You can specify the measure space by providing a `chsel.measure.MeasureFunction` object.
Note that this also defines what space the QD optimization is performed in. (such as a 9 dimensional continuous 
representation of SE(3) transforms, or a 3 dimensional space for SE(2) transforms).

For example, if you are on a tabletop and know the object's position quite well and want diversity in yaw:

```python
import chsel

normal = torch.tensor([0, 0, 1])
offset = torch.tensor([0, 0, 0])
measure = chsel.SE2AngleMeasure(axis_of_rotation=normal, offset_along_normal=offset)
registration = chsel.CHSEL(sdf, positions, semantics, axis_of_rotation=normal, offset_along_normal=offset)
```