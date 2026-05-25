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


def _wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def image_u_to_polar_theta(u, camera_theta_xy):
    """
    Gör om bildkoordinaten u till standardpolär vinkel i XY-planet.

    Positiv u ligger åt kamerans höger, alltså medurs från kamerans
    framåtriktning. Därför subtraheras bildvinkeln från camera_theta_xy.
    """
    return camera_theta_xy - math.atan(u)


def angle_to_bin(theta, num_bins=128, fov_degrees=90.0, theta_center=0.0):
    fov = math.radians(fov_degrees)
    theta_rel = _wrap_to_pi(theta - theta_center)
    theta_min = -0.5 * fov
    theta_max = 0.5 * fov

    if theta_rel < theta_min or theta_rel >= theta_max:
        return None

    bin_size = fov / num_bins
    return int((theta_rel - theta_min) / bin_size)


def build_vision_matrix(
    projections,
    cylinders,
    num_bins=32,
    fov_degrees=90.0,
    camera_theta_xy=None,
):
    """
    projections: [(idx, u_min, u_max, v_min, v_max, d)]
    cylinders: [(x, y, r, h)] eller [(x, y, z, r, h)]
    camera_theta_xy:
        Om satt tolkas bin-axeln som standardpolär theta i scenens XY-plan:
        p_xy = cam.c_xy + d * (cos(theta), sin(theta)).
        Om None behålls den gamla relativa bildvinkeln atan(u).

    Returns:
        mat[:, 0] = occupancy
        mat[:, 1] = radius
        mat[:, 2] = depth
        mat[:, 3] = cylinder_id
    """

    mat = np.zeros((num_bins, 4), dtype=np.float32)
    mat[:, 3] = -1

    for (idx, u0, u1, v_min, v_max, d) in projections:
        x, y, r, h = unpack_cylinder(cylinders[idx])

        u_center = 0.5 * (u0 + u1)
        if camera_theta_xy is None:
            theta = math.atan(u_center)
            theta_center = 0.0
        else:
            theta = image_u_to_polar_theta(u_center, camera_theta_xy)
            theta_center = camera_theta_xy

        b = angle_to_bin(
            theta,
            num_bins=num_bins,
            fov_degrees=fov_degrees,
            theta_center=theta_center,
        )

        if b is None:
            continue

        if mat[b, 0] == 0 or d < mat[b, 2]:
            mat[b, 0] = 1.0
            mat[b, 1] = r
            mat[b, 2] = d
            mat[b, 3] = idx

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
    cam1=None,
    cam2=None,
):
    theta1 = cam1.theta_xy if cam1 is not None else None
    theta2 = cam2.theta_xy if cam2 is not None else None

    mat1 = build_vision_matrix(
        proj1,
        cylinders,
        num_bins=num_bins,
        fov_degrees=fov_degrees,
        camera_theta_xy=theta1,
    )

    mat2 = build_vision_matrix(
        proj2,
        cylinders,
        num_bins=num_bins,
        fov_degrees=fov_degrees,
        camera_theta_xy=theta2,
    )

    return mat1, mat2
