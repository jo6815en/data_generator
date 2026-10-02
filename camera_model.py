import numpy as np
import random
from copy import deepcopy

# -----------------------
# Kamera-klass (3D)
# -----------------------
class Camera3D:
    def __init__(self, position, theta_xy, pitch=0.0, fov_deg=90):
        self.c = np.array(position, dtype=float)
        self.theta_xy = theta_xy
        self.pitch = pitch

        # Horisontell forward/right-bas
        forward_xy = np.array([
            np.cos(theta_xy),
            np.sin(theta_xy),
            0.0,
        ])

        self.right = np.array([
            np.sin(theta_xy),
            -np.cos(theta_xy),
            0.0,
        ])

        # Pitch > 0 = kameran tittar uppåt
        self.dir = (
            np.cos(pitch) * forward_xy
            + np.sin(pitch) * np.array([0.0, 0.0, 1.0])
        )
        self.dir /= np.linalg.norm(self.dir)

        # Up måste också roteras så att kamerabasen förblir ortonormal
        self.up = np.cross(self.right, self.dir)
        self.up /= np.linalg.norm(self.up)

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
    pitch_min_deg=-10.0,
    pitch_max_deg=10.0,
    pad=2.0,
    min_cam_cyl_dist=0.05,
    max_cam_cyl_dist=7.0,
    min_visible=2,
    max_tries=5000,
    camera_height=1.7,
    near_tree_prob=0.4,
    near_tree_min_dist=0.05,
    near_tree_max_dist=0.5,
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

    pair_center_z = camera_height

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

    near_tree_idx = None        
    near_tree_mode = random.random() < near_tree_prob

    for _ in range(max_tries):

        if near_tree_mode:
            # Välj en cylinder som kamerorna ska hamna nära
            near_tree_idx = random.randrange(len(cyls))
            tx, ty, tz, tr, th = cyls[near_tree_idx]

            phi = random.uniform(0, 2 * np.pi)
            surface_dist = random.uniform(
                near_tree_min_dist,
                near_tree_max_dist,)

            # Pair center nära cylinderns yta
            dist_from_center = tr + surface_dist

            pair_center = np.array([
                tx + dist_from_center * np.cos(phi),
                ty + dist_from_center * np.sin(phi),
                pair_center_z,
            ])

            # Baseline tangentiellt runt cylindern
            tangent = np.array([
                -np.sin(phi),
                np.cos(phi),])

            offset = np.array([
                tangent[0] * camera_distance,
                tangent[1] * camera_distance,
                0.0,])

        else:
            # Din vanliga fria sampling
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

        if near_tree_mode:
            # Gemensam yaw för hela kameraparet:
            # pair center tittar mot foreground-cylindern
            theta_pair = np.arctan2(
                ty - pair_center[1],
                tx - pair_center[0],
            )

            # Ett gemensamt litet jitter
            theta_pair += np.deg2rad(
                random.uniform(-angle_jitter_deg, angle_jitter_deg)
            )

            theta1 = theta_pair
            theta2 = theta_pair

        else:
            # Vanliga kameror
            theta1 = sample_theta(cam1_pos)
            theta2 = sample_theta(cam2_pos)

        pitch = np.deg2rad(
            random.uniform(pitch_min_deg, pitch_max_deg)
        )

        cam1 = Camera3D(
            cam1_pos,
            theta1,
            pitch=pitch,
        )

        cam2 = Camera3D(
            cam2_pos,
            theta2,
            pitch=pitch,
        )

        proj1 = compute_visibility(cam1, cylinders)
        proj2 = compute_visibility(cam2, cylinders)

        if len(proj1) < min_visible:
            continue
        if len(proj2) < min_visible:
            continue

        return cam1, cam2, near_tree_idx
    
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

    center_xy = np.array([x, y], dtype=float)
    rel_xy = center_xy - cam.c[:2]
    d = np.linalg.norm(rel_xy)

    # Kamera inuti cylindern
    if d <= r + 1e-6:
        return None

    # --------------------------------------------------
    # Horisontell utbredning
    # --------------------------------------------------

    # Absolut bearing till cylindercentrum
    bearing = np.arctan2(rel_xy[1], rel_xy[0])

    # Bearing relativt kamerans yaw
    beta = np.arctan2(
        np.sin(bearing - cam.theta_xy),
        np.cos(bearing - cam.theta_xy),
    )

    # Exakt angular half-width för en cirkel
    alpha = np.arcsin(np.clip(r / d, 0.0, 1.0))

    angle_left = beta - alpha
    angle_right = beta + alpha

    # För pinhole-projektionen gäller u = tan(angle)
    #
    # Om kanten passerar ±90° blir tan instabil.
    # Sätt då mycket stort värde; renderern klipper senare mot FOV.
    eps = 1e-4

    angle_left = np.clip(
        angle_left,
        -np.pi / 2 + eps,
        np.pi / 2 - eps,
    )
    angle_right = np.clip(
        angle_right,
        -np.pi / 2 + eps,
        np.pi / 2 - eps,
    )

    u_min = np.tan(angle_left)
    u_max = np.tan(angle_right)

    if u_min > u_max:
        u_min, u_max = u_max, u_min

    # --------------------------------------------------
    # Vertikal utbredning
    # --------------------------------------------------
    #
    # Använd cylindercentrumets riktning för bottom/top.
    # Detta är stabilare i närfältet än att kräva att alla
    # fyra tangent-hörnpunkter ligger framför kameran.
    # --------------------------------------------------

    bottom = np.array([x, y, z], dtype=float)
    top = np.array([x, y, z + h], dtype=float)

    res_bottom = project_point(cam, bottom)
    res_top = project_point(cam, top)

    # Om centrumlinjen ligger helt bakom kameran är
    # cylindern inte användbar.
    if res_bottom is None and res_top is None:
        return None

    vs = []

    if res_bottom is not None:
        vs.append(res_bottom[0][1])

    if res_top is not None:
        vs.append(res_top[0][1])

    # Om bara ena änden är framför kameran låter vi cylindern
    # fortsätta långt utanför bilden åt andra hållet.
    LARGE_V = 1e4

    if res_bottom is None:
        v_min = -LARGE_V
        v_max = max(vs)
    elif res_top is None:
        v_min = min(vs)
        v_max = LARGE_V
    else:
        v_min = min(vs)
        v_max = max(vs)

    # Vision-depth förblir radialt XY-avstånd
    d_radial = d

    return u_min, u_max, v_min, v_max, d_radial


# -----------------------
# Synlighet / projection per kamera
# -----------------------
def compute_visibility(cam, cylinders, fov_u=1.0, fov_v=1.0):
    projections = []

    for i, cyl in enumerate(cylinders):
        res = project_cylinder(cam, cyl)

        if res is None:
            continue

        u_min, u_max, v_min, v_max, d = res

        # Cylindern är synlig om dess projicerade rektangel
        # överlappar bildens FOV.
        if (
            u_max < -fov_u
            or u_min > fov_u
            or v_max < -fov_v
            or v_min > fov_v
        ):
            continue

        projections.append(
            (i, u_min, u_max, v_min, v_max, d)
        )

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
        cam_new.dir /= np.linalg.norm(cam_new.dir)
        cam_new.up /= np.linalg.norm(cam_new.up)
        cam_new.right /= np.linalg.norm(cam_new.right)
        
        cam_new.theta_xy = np.arctan2(
            cam_new.dir[1],
            cam_new.dir[0],
        )
        horizontal_norm = np.linalg.norm(cam_new.dir[:2])
        
        cam_new.pitch = np.arctan2(
            cam_new.dir[2],
            horizontal_norm,
        )

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
