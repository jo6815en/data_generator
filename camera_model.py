import numpy as np
import random
from copy import deepcopy

# -----------------------
# Kamera-klass (3D)
# -----------------------
class Camera3D:
    def __init__(self, position, theta_xy, fov_deg=90):
        self.c = np.array(position, dtype=float)
        self.theta_xy = theta_xy

        self.dir = np.array([
            np.cos(theta_xy),
            np.sin(theta_xy),
            0.0
        ])

        self.up = np.array([0.0, 0.0, 1.0])

        # Högerhänt bas
        self.right = np.array([
            np.sin(theta_xy),
            -np.cos(theta_xy),
            0.0
        ])

        self.fov = np.deg2rad(fov_deg)


# -----------------------
# Skapa kamerapar
# -----------------------
def create_camera_pair(
    cylinders,
    bounds,
    seed=None,
    camera_distance=2.0,
    angle_jitter_deg=45,
    pad=2.0,
    min_cam_cyl_dist=1.0,
    max_cam_cyl_dist=7.0,
    min_visible=2,
    max_tries=5000,
):
    if seed is not None:
        random.seed(seed)

    xmin, xmax, ymin, ymax = bounds

    def unpack_cylinder(cyl):
        if len(cyl) == 5:
            x, y, z, r, h = cyl
        elif len(cyl) == 4:
            x, y, r, h = cyl
            z = 0.0
        else:
            raise ValueError(f"Unexpected cylinder format: {cyl}")
        return x, y, z, r, h

    cyls = [unpack_cylinder(c) for c in cylinders]

    center_xy = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
    ])

    if len(cyls) > 0:
        pair_center_z = np.mean([z + h / 2 for (_, _, z, _, h) in cyls])
    else:
        pair_center_z = 0.0

    def valid_camera_distance(cam_pos):
        for x, y, z, r, h in cyls:
            dist_xy = np.linalg.norm(cam_pos[:2] - np.array([x, y]))

            # avstånd till cylinderns yta, inte till centrum
            surface_dist = dist_xy - r

            if surface_dist < min_cam_cyl_dist:
                return False

        nearest_surface_dist = min(
            np.linalg.norm(cam_pos[:2] - np.array([x, y])) - r
            for x, y, z, r, h in cyls
        )

        if nearest_surface_dist > max_cam_cyl_dist:
            return False

        return True

    def sample_theta(cam_pos):
        # sikta ungefär mot cylindrarnas centroid
        target_xy = np.array([
            np.mean([x for (x, y, z, r, h) in cyls]),
            np.mean([y for (x, y, z, r, h) in cyls]),
        ])

        base_angle = np.arctan2(
            target_xy[1] - cam_pos[1],
            target_xy[0] - cam_pos[0],
        )

        jitter = np.deg2rad(random.uniform(-angle_jitter_deg, angle_jitter_deg))
        return base_angle + jitter
        

    for _ in range(max_tries):
        pair_center = np.array([
            random.uniform(xmin - pad, xmax + pad),
            random.uniform(ymin - pad, ymax + pad),
            pair_center_z,
        ])

        angle = random.uniform(0, 2 * np.pi)

        offset = np.array([
            np.cos(angle) * camera_distance,
            np.sin(angle) * camera_distance,
            0.0,
        ])

        cam1_pos = pair_center - 0.5 * offset
        cam2_pos = pair_center + 0.5 * offset

        if not valid_camera_distance(cam1_pos):
            continue
        if not valid_camera_distance(cam2_pos):
            continue

        theta1 = sample_theta(cam1_pos)
        theta2 = sample_theta(cam2_pos)

        cam1 = Camera3D(cam1_pos, theta1)
        cam2 = Camera3D(cam2_pos, theta2)

        proj1 = compute_visibility(cam1, cylinders)
        proj2 = compute_visibility(cam2, cylinders)

        if len(proj1) < min_visible:
            continue
        if len(proj2) < min_visible:
            continue

        return cam1, cam2

    raise RuntimeError(
        "Could not sample a valid camera pair. "
        "Try lowering min_cam_cyl_dist, lowering min_visible, "
        "increasing max_cam_cyl_dist, increasing pad, or increasing max_tries."
    )


def compute_camera_pair(*args, **kwargs):
    return create_camera_pair(*args, **kwargs)


# -----------------------
# Projektion (pinhole)
# -----------------------
def project_point(cam, point):
    p = point - cam.c
    depth = np.dot(p, cam.dir)

    if depth <= 0:
        return None

    u = np.dot(p, cam.right) / depth
    v = np.dot(p, cam.up) / depth
    return np.array([u, v]), depth


# -----------------------
# Projektion av cylinder
# -----------------------
def project_cylinder(cam, cylinder):
    if len(cylinder) == 5:
        x, y, z, r, h = cylinder
    else:
        x, y, r, h = cylinder
        z = 0.0

    perp = np.array([-cam.dir[1], cam.dir[0], 0.0])
    perp = perp / np.linalg.norm(perp)

    base = np.array([x, y, z])
    left = base + perp * r
    right = base - perp * r

    bottom_l = left
    top_l = left + np.array([0.0, 0.0, h])

    bottom_r = right
    top_r = right + np.array([0.0, 0.0, h])

    res_l1 = project_point(cam, bottom_l)
    res_l2 = project_point(cam, top_l)
    res_r1 = project_point(cam, bottom_r)
    res_r2 = project_point(cam, top_r)

    if None in [res_l1, res_l2, res_r1, res_r2]:
        return None

    (u_l, v_l1), d1 = res_l1
    (u_l2, v_l2), d2 = res_l2
    (u_r, v_r1), d3 = res_r1
    (u_r2, v_r2), d4 = res_r2

    u_min = min(u_l, u_l2, u_r, u_r2)
    u_max = max(u_l, u_l2, u_r, u_r2)
    v_min = min(v_l1, v_l2, v_r1, v_r2)
    v_max = max(v_l1, v_l2, v_r1, v_r2)

    # Vision-depth: radialt avstånd i top-down-planet.
    # Med standardpolär theta från vision.image_u_to_polar_theta gäller:
    #   p_xy = cam.c_xy + d * (cos(theta), sin(theta))
    center = np.array([x, y, z], dtype=float)
    relative = center - cam.c
    d_radial = np.linalg.norm(relative[:2])

    return u_min, u_max, v_min, v_max, d_radial


# -----------------------
# Synlighet / projection per kamera
# -----------------------
def compute_visibility(cam, cylinders):
    projections = []

    for i, cyl in enumerate(cylinders):
        if len(cyl) == 5:
            x, y, z, r, h = cyl
        else:
            x, y, r, h = cyl
            z = 0.0

        center = np.array([x, y, z + h / 2])

        if not in_fov(cam, center):
            continue

        res = project_cylinder(cam, cyl)
        if res is None:
            continue

        u_min, u_max, v_min, v_max, d = res
        projections.append((i, u_min, u_max, v_min, v_max, d))

    return projections


def compute_projections(cam1, cam2, cylinders):
    proj1 = compute_visibility(cam1, cylinders)
    proj2 = compute_visibility(cam2, cylinders)
    return proj1, proj2


def in_fov(cam, point):
    p = point - cam.c
    p_norm = np.linalg.norm(p)

    if p_norm < 1e-6:
        return False

    cos_angle = np.dot(p, cam.dir) / p_norm
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    return angle < cam.fov / 2


# ---------------------------
# Flytta allt till kamera 1
# ---------------------------
def get_relative_pose(cam1_n, cam2_n):
    """
    Returnerar cam2:s pose relativt cam1.

    Antagande:
      cam1_n och cam2_n är redan uttryckta i cam1:s koordinatsystem.

    Return:
      - cam2_in_cam1: cam2:s center och orientering i cam1-ramen
      - R_cam1_to_cam2, t_cam1_to_cam2: extrinsic som mappar en punkt i cam1-ramen till cam2-ramen
        p_cam2 = R_cam1_to_cam2 @ p_cam1 + t_cam1_to_cam2
    """

    r2 = cam2_n.right / np.linalg.norm(cam2_n.right)
    d2 = cam2_n.dir / np.linalg.norm(cam2_n.dir)
    u2 = cam2_n.up / np.linalg.norm(cam2_n.up)

    # cam2:s lokala bas uttryckt i cam1-ramen
    R_cam2_in_cam1 = np.column_stack([r2, d2, u2])

    # Transformation från cam1-ram till cam2-ram
    # p_cam2 = R @ p_cam1 + t
    R_cam1_to_cam2 = R_cam2_in_cam1.T
    t_cam1_to_cam2 = -R_cam1_to_cam2 @ cam2_n.c

    relative_pose = {
        "cam2_in_cam1": {
            "c": cam2_n.c.copy(),
            "dir": cam2_n.dir.copy(),
            "up": cam2_n.up.copy(),
            "right": cam2_n.right.copy(),
            "theta_xy": cam2_n.theta_xy,
        },
        "R_cam1_to_cam2": R_cam1_to_cam2,
        "t_cam1_to_cam2": t_cam1_to_cam2,
    }

    return relative_pose

def transform_scene_to_cam1(cam1, cam2, cylinders):
    r = cam1.right / np.linalg.norm(cam1.right)
    d = cam1.dir / np.linalg.norm(cam1.dir)
    u = cam1.up / np.linalg.norm(cam1.up)

    B = np.column_stack([r, d, u])

    def to_local_point(p):
        v = np.asarray(p, dtype=float) - cam1.c
        return B.T @ v

    def to_local_vec(v):
        v = np.asarray(v, dtype=float)
        return B.T @ v

    def transform_camera(cam):
        cam_new = deepcopy(cam)
        cam_new.c = to_local_point(cam.c)
        cam_new.dir = to_local_vec(cam.dir)
        cam_new.up = to_local_vec(cam.up)
        cam_new.right = to_local_vec(cam.right)
        cam_new.theta_xy = np.arctan2(cam_new.dir[1], cam_new.dir[0])
        return cam_new

    def unpack_cylinder(cyl):
        if len(cyl) == 5:
            x, y, z, r_cyl, h = cyl
        elif len(cyl) == 4:
            x, y, r_cyl, h = cyl
            z = 0.0
        else:
            raise ValueError(f"Unexpected cylinder format: {cyl}")
        return x, y, z, r_cyl, h

    cam1_t = transform_camera(cam1)
    cam2_t = transform_camera(cam2)

    cylinders_t = []
    for cyl in cylinders:
        x, y, z, r_cyl, h = unpack_cylinder(cyl)
        base_t = to_local_point([x, y, z])
        cylinders_t.append((base_t[0], base_t[1], base_t[2], r_cyl, h))

    return cam1_t, cam2_t, cylinders_t
