import numpy as np
import random

# -----------------------
# Kamera-klass (3D)
# -----------------------
class Camera3D:
    def __init__(self, position):
        self.c = np.array(position)

        # rektifierad: tittar alltid längs +Z
        self.dir = np.array([0.0, 0.0, 1.0])

        # kamerakoordinatsystem (standard)
        self.right = np.array([1.0, 0.0, 0.0])
        self.up    = np.array([0.0, 1.0, 0.0])


# -----------------------
# Skapa rektifierat kamerapar
# -----------------------
def look_at_direction(cam_pos, target_xy):
    dir_xy = target_xy - cam_pos[:2]
    dir_xy = dir_xy / np.linalg.norm(dir_xy)

    # lägg till Z-komponent (så vi tittar framåt också)
    dir_3d = np.array([dir_xy[0], dir_xy[1], 1.0])
    return dir_3d / np.linalg.norm(dir_3d)


def create_camera_pair(cylinders, bounds, seed=None, baseline=2.0):
    if seed is not None:
        random.seed(seed)

    xmin, xmax, ymin, ymax = bounds

    # beräkna scenens centrum i XY
    xs = [x for (x, _, _, _) in cylinders]
    ys = [y for (_, y, _, _) in cylinders]

    center_xy = np.array([np.mean(xs), np.mean(ys)])

    # placera kameror utanför scenen (vänster sida)
    cam1_pos = np.array([
        xmin - 2.0,                           # utanför
        (ymin + ymax) / 2,                    # mitten i y
        2.0                                   # höjd
    ])

    cam2_pos = cam1_pos + np.array([baseline, 0.0, 0.0])

    cam1 = Camera3D(cam1_pos)
    cam2 = Camera3D(cam2_pos)

    cam1.dir = look_at_direction(cam1_pos, center_xy)
    cam2.dir = look_at_direction(cam2_pos, center_xy)

    return cam1, cam2


# -----------------------
# Projektion (pinhole)
# -----------------------
def project_point(cam, point):
    p = point - cam.c

    # bakom kamera
    if p[2] <= 0:
        return None

    # pinhole f=1
    u = p[0] / p[2]
    v = p[1] / p[2]

    return np.array([u, v])


# -----------------------
# Projektion av cylinder
# -----------------------
def project_cylinder(cam, cylinder):
    x, y, r, h = cylinder

    # ta mitten av cylindern
    center = np.array([x, y, h / 2])

    proj = project_point(cam, center)
    if proj is None:
        return None

    # approx skala (perspektiv)
    scale = 1.0 / center[2]
    r_proj = r * scale

    return proj, r_proj


# -----------------------
# Beräkna "visibility"
# (nu enklare: ingen 1D ocklusion ännu)
# -----------------------
def compute_visibility(cam, cylinders):
    projections = []

    for i, cyl in enumerate(cylinders):
        res = project_cylinder(cam, cyl)

        if res is None:
            continue

        (u, v), r_proj = res

        # approx depth
        center = np.array([cyl[0], cyl[1], cyl[3]/2])
        d = np.linalg.norm(center - cam.c)

        projections.append((i, u, v, r_proj, d))

    return projections


# -----------------------
# Convenience
# -----------------------
def compute_camera_pair(cylinders, bounds, seed=None):
    cam1, cam2 = create_camera_pair(cylinders, bounds, seed)

    proj1 = compute_visibility(cam1, cylinders)
    proj2 = compute_visibility(cam2, cylinders)

    return cam1, cam2, proj1, proj2