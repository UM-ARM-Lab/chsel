import torch
import pytorch_kinematics as pk


def project_transform_homogeneous(axis, offset, H):
    is_tsf = isinstance(H, pk.Transform3d)
    if is_tsf:
        H = H.get_matrix()
    R, T = H[:, :3, :3], H[:, :3, 3]
    R, T = project_transform(axis, offset, R, T)
    H = torch.eye(4, device=H.device, dtype=H.dtype).repeat(H.shape[0], 1, 1)
    H[:, :3, :3] = R
    H[:, :3, 3] = T
    if is_tsf:
        H = pk.Transform3d(matrix=H, device=H.device, dtype=H.dtype)
    return H


def project_transform(axis, offset, R, T):
    axis_angle = pk.matrix_to_axis_angle(R)
    theta = axis_angle @ axis
    # ensure the rotation is around the given axis
    R = pk.axis_and_angle_to_matrix_33(axis, theta)
    # ensure the translation is on the plane defined by the axis of rotation and the offset
    T = project_onto_plane(axis, offset, T)
    return R, T


def project_onto_plane(axis, offset, points):
    """Project points onto a plane defined by axis and offset
    :param axis: 3 unit length vector originating from the origin
    :param offset scalar offset along the axis to define the plane
    :param points: N x 3 tensor of points to project
    """
    # basically just need to find the perpendicular distance of each point to the plane
    # and subtract that from each point along axis
    # Compute dot product of each point with the normalized axis
    dot_product = torch.einsum('i,bi->b', axis, points) - offset

    # Compute the projection components
    projection = points - dot_product.unsqueeze(-1) * axis.unsqueeze(0)
    return projection


def find_orthogonal_vector(v):
    """Find an orthogonal vector to v that is not the zero vector."""
    if v[0] == 0 and v[1] == 0:
        if v[2] == 0:
            raise ValueError("Zero vector does not have a defined orthogonal vector")
        # v is pointing in the z direction, so crossing with x will not produce the zero vector.
        return torch.tensor([0, -1, 0], device=v.device, dtype=v.dtype)
    # Not pointing in the z direction, so crossing with z is safe
    return torch.tensor([0, 0, -1], device=v.device, dtype=v.dtype)


def construct_plane_basis(normal):
    """Constructs a basis (two orthogonal unit vectors) for the plane given the plane normal."""
    axis_u = torch.cross(normal, find_orthogonal_vector(normal))
    axis_u /= torch.linalg.norm(axis_u)
    axis_v = torch.cross(normal, axis_u)
    return axis_u, axis_v


def xyz_to_uv(xyz, origin, normal, axis_u, axis_v, already_projected=False):
    """Projects a point onto the plane and computes its uv coordinates in the plane's basis."""
    if not already_projected:
        xyz = project_onto_plane(normal, torch.dot(origin, normal), xyz)

    # these are actually in batch
    diff = xyz - origin
    u = diff @ axis_u
    v = diff @ axis_v
    uv = torch.stack([u, v], dim=-1)
    return uv


def uv_to_xyz(uv, origin, axis_u, axis_v):
    """Converts uv coordinates back to xyz coordinates on the plane."""
    point_proj = origin + uv[:, 0].view(-1, 1) * axis_u + uv[:, 1].view(-1, 1) * axis_v
    return point_proj
