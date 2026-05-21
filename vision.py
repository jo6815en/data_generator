import numpy as np
import math

# -----------------------
# Bygg vision matrix
# -----------------------

def unpack_cylinder(cyl):
    if len(cyl) == 4:
        x, y, r, h = cyl
        return x, y, r, h

    if len(cyl) == 5:
        x, y, z, r, h = cyl
        return x, y, r, h

    raise ValueError(f"Unexpected cylinder format with len={len(cyl)}: {cyl}")  

def angle_to_bin(theta, num_bins=128, fov_degrees=90.0):
    fov = math.radians(fov_degrees)
    theta_min = -0.5 * fov
    theta_max = 0.5 * fov

    if theta < theta_min or theta >= theta_max:
        return None

    bin_size = fov / num_bins
    return int((theta - theta_min) / bin_size)


def build_vision_matrix(
    projections,
    cylinders,
    num_bins=32,
    fov_degrees=90.0,
):
    """
    projections: [(idx, u_min, u_max, v_min, v_max, d)]
    cylinders: [(x, y, r, h)]

    Returns:
        mat[:, 0] = occupancy
        mat[:, 1] = radius
        mat[:, 2] = depth
    """

    # [occupancy, radius, depth]
    mat = np.zeros((num_bins, 3), dtype=np.float32)

    for (idx, u0, u1, v_min, v_max, d) in projections:
        x, y, r, h = unpack_cylinder(cylinders[idx])

        theta = math.atan2(y, x)

        b = angle_to_bin(
            theta,
            num_bins=num_bins,
            fov_degrees=fov_degrees,
        )

        if b is None:
            continue

        # välj närmaste objekt om flera hamnar i samma bin
        if mat[b, 0] == 0 or d < mat[b, 2]:
            mat[b, 0] = 1.0  # occupancy
            mat[b, 1] = r    # radius
            mat[b, 2] = d    # depth

    return mat


# -----------------------
# Convenience
# -----------------------

def build_vision_pair(
    proj1,
    proj2,
    cylinders,
    num_bins=32,
    fov_degrees=90.0,
):
    mat1 = build_vision_matrix(
        proj1,
        cylinders,
        num_bins=num_bins,
        fov_degrees=fov_degrees,
    )

    mat2 = build_vision_matrix(
        proj2,
        cylinders,
        num_bins=num_bins,
        fov_degrees=fov_degrees,
    )

    return mat1, mat2
