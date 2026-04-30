import numpy as np
import random

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
        self.right = np.array([
            -np.sin(theta_xy),
             np.cos(theta_xy),
             0.0
        ])

        self.fov = np.deg2rad(fov_deg)


# -----------------------
# Skapa kamerapar
# -----------------------
def create_camera_pair(cylinders, bounds, seed=None, baseline_range=(1.0, 3.0), angle_jitter_deg=45, pad=2.0):
    if seed is not None:
        random.seed(seed)

    xmin, xmax, ymin, ymax = bounds

    # scenens centrum
    center_xy = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2
    ])

    # -----------------------
    # Slumpa mittpunkt för paret
    # -----------------------
    pair_center = np.array([
        random.uniform(xmin - pad, xmax + pad),
        random.uniform(ymin - pad, ymax + pad),
        np.mean([h for (_, _, _, h) in cylinders]) / 2
    ])

    # -----------------------
    # Slumpa baseline (XY-plan)
    # -----------------------
    baseline = random.uniform(*baseline_range)
    angle = random.uniform(0, 2*np.pi)

    offset = np.array([
        np.cos(angle) * baseline,
        np.sin(angle) * baseline,
        0.0
    ])

    cam1_pos = pair_center - 0.5 * offset
    cam2_pos = pair_center + 0.5 * offset

    # -----------------------
    # Riktning mot scen + jitter
    # -----------------------
    def sample_theta(cam_pos):
        base_angle = np.arctan2(
            center_xy[1] - cam_pos[1],
            center_xy[0] - cam_pos[0]
        )

        jitter = np.deg2rad(random.uniform(-angle_jitter_deg, angle_jitter_deg))
        return base_angle + jitter

    theta1 = sample_theta(cam1_pos)
    theta2 = sample_theta(cam2_pos)

    # -----------------------
    # Skapa kameror
    # -----------------------
    cam1 = Camera3D(cam1_pos, theta1)
    cam2 = Camera3D(cam2_pos, theta2)

    return cam1, cam2


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
    x, y, r, h = cylinder

    # riktning vinkelrät mot kamerans riktning
    perp = np.array([-cam.dir[1], cam.dir[0], 0])
    perp = perp / np.linalg.norm(perp)

    # två sidopunkter
    left  = np.array([x, y, 0]) + perp * r
    right = np.array([x, y, 0]) - perp * r

    # topp/botten
    bottom_l = left
    top_l    = left + np.array([0, 0, h])

    bottom_r = right
    top_r    = right + np.array([0, 0, h])

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

    d = (d1 + d2 + d3 + d4) / 4

    return u_min, u_max, v_min, v_max, d


def in_fov(cam, point):
    p = point - cam.c
    p_norm = np.linalg.norm(p)

    if p_norm < 1e-6:
        return False

    cos_angle = np.dot(p, cam.dir) / p_norm
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    return angle < cam.fov / 2


# -----------------------
# Beräkna visibility
# -----------------------
def compute_visibility(cam, cylinders):
    projections = []

    for i, cyl in enumerate(cylinders):
        x, y, r, h = cyl

        center = np.array([x, y, h/2])

        if not in_fov(cam, center):
            continue

        res = project_cylinder(cam, cyl)
        if res is None:
            continue

        u_min, u_max, v_min, v_max, d = res
        projections.append((i, u_min, u_max, v_min, v_max, d))

    return projections


# -----------------------
# Convenience
# -----------------------
def compute_camera_pair(cylinders, bounds, seed=None):
    cam1, cam2 = create_camera_pair(cylinders, bounds, seed)

    proj1 = compute_visibility(cam1, cylinders)
    proj2 = compute_visibility(cam2, cylinders)

    return cam1, cam2, proj1, proj2

