import numpy as np
from matplotlib.colors import to_rgb
from pathlib import Path
import random
from PIL import Image

BG_DIR = Path("backgrounds")
BG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_random_background(image_size):
    files = [p for p in BG_DIR.iterdir() if p.suffix.lower() in BG_EXTS]
    if not files:
        raise FileNotFoundError(f"Inga bakgrundsbilder hittades i {BG_DIR}")

    bg_path = random.choice(files)
    bg = Image.open(bg_path).convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
    return np.array(bg, dtype=np.uint8)

def _color_to_rgb255(color):
    rgb = np.array(to_rgb(color), dtype=float)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def _clip_rect_to_fov(u_min, u_max, v_min, v_max, fov_u, fov_v):
    u0, u1 = sorted((u_min, u_max))
    v0, v1 = sorted((v_min, v_max))

    if u1 < -fov_u or u0 > fov_u or v1 < -fov_v or v0 > fov_v:
        return None

    u0 = max(u0, -fov_u)
    u1 = min(u1, fov_u)
    v0 = max(v0, -fov_v)
    v1 = min(v1, fov_v)

    if u0 >= u1 or v0 >= v1:
        return None

    return u0, u1, v0, v1


def _rect_to_pixel_bounds(u0, u1, v0, v1, width, height, fov_u, fov_v):
    # u: vänster → höger
    c0 = int(round((u0 + fov_u) / (2 * fov_u) * (width - 1)))
    c1 = int(round((u1 + fov_u) / (2 * fov_u) * (width - 1)))

    # v: upp → ned (bildkoordinater)
    r0 = int(round((fov_v - v1) / (2 * fov_v) * (height - 1)))
    r1 = int(round((fov_v - v0) / (2 * fov_v) * (height - 1)))

    c0, c1 = sorted((c0, c1))
    r0, r1 = sorted((r0, r1))

    c0 = max(0, min(width - 1, c0))
    c1 = max(0, min(width - 1, c1))
    r0 = max(0, min(height - 1, r0))
    r1 = max(0, min(height - 1, r1))

    if c0 > c1 or r0 > r1:
        return None

    return r0, r1, c0, c1

def random_bark_color():
    palettes = [
        (90, 75, 60),
        (110, 95, 75),
        (75, 70, 65),
        (125, 110, 90),
        (65, 60, 55),
    ]

    base = np.array(random.choice(palettes), dtype=float)
    jitter = np.random.normal(0, 12, 3)
    return np.clip(base + jitter, 20, 200).astype(np.uint8)

def make_bark_texture(height, width, base_color, seed):
    rng = np.random.default_rng(seed)
    base = np.asarray(base_color, dtype=np.float32)

    pixel_noise = rng.normal(0, 12, (height, width, 1))
    column_noise = rng.normal(0, 15, (1, width, 1))
    column_noise = np.repeat(column_noise, height, axis=0)

    texture = base[None, None, :] + pixel_noise + column_noise

    # Cylindrical shading: brighter near the center, darker at the edges
    x = np.linspace(-1.0, 1.0, width)
    shading = 0.60 + 0.40 * np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
    texture *= shading[None, :, None]

    return np.clip(texture, 0, 255).astype(np.uint8)


def make_ground_background(
    image_size, cam, fov_u=1.0, fov_v=1.0,
    seed=None, ground_z=0.0,
):
    width, height = image_size
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Scene appearance
    # --------------------------------------------------
    sky_styles = [
        ([120, 160, 200], [190, 210, 220]),
        ([135, 140, 145], [185, 190, 190]),
        ([175, 185, 190], [215, 220, 215]),
        ([175, 165, 145], [220, 205, 175]),
        ([105, 125, 140], [165, 175, 180]),
    ]

    top, bottom = sky_styles[rng.integers(len(sky_styles))]
    sky_top = np.asarray(top, dtype=np.float32) + rng.normal(0, 10, 3)
    sky_bottom = np.asarray(bottom, dtype=np.float32) + rng.normal(0, 8, 3)

    ground_base = np.array([
        rng.uniform(45, 100),
        rng.uniform(65, 130),
        rng.uniform(30, 80),
    ], dtype=np.float32)

    phases = rng.uniform(0, 2 * np.pi, 3)
    angles = rng.uniform(0, 2 * np.pi, 3)
    scales = np.array([
        rng.uniform(0.4, 0.9),
        rng.uniform(1.2, 2.5),
        rng.uniform(3.0, 6.0),
    ])
    contrast = rng.uniform(15, 35)

    haze_color = (
        0.65 * sky_bottom
        + 0.35 * np.array([150, 155, 135], dtype=np.float32)
    )

    # --------------------------------------------------
    # Pixel coordinates
    # --------------------------------------------------
    u = np.linspace(-fov_u, fov_u, width, dtype=np.float32)
    v = np.linspace(fov_v, -fov_v, height, dtype=np.float32)

    U, V = np.meshgrid(u, v)

    # --------------------------------------------------
    # Rays for every pixel
    # [H,W,3]
    # --------------------------------------------------
    rays = (
        cam.dir[None, None, :]
        + U[..., None] * cam.right[None, None, :]
        + V[..., None] * cam.up[None, None, :]
    )

    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # --------------------------------------------------
    # Sky
    # --------------------------------------------------
    sky_t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]

    sky = (
        (1 - sky_t) * sky_top[None, None, :]
        + sky_t * sky_bottom[None, None, :]
    )

    sky = np.broadcast_to(
        sky,
        (height, width, 3),
    ).copy()

    img = sky.copy()

    # Only downward rays hit the ground.
    ground_mask = rays[..., 2] < -1e-8

    if not np.any(ground_mask):
        return np.clip(img, 0, 255).astype(np.uint8)

    # --------------------------------------------------
    # Ray / ground-plane intersections
    # --------------------------------------------------
    ray_z = rays[..., 2]

    t = np.zeros_like(ray_z)
    t[ground_mask] = (
        (ground_z - cam.c[2])
        / ray_z[ground_mask]
    )

    ground_mask &= t > 0

    # World-space coordinates for every pixel.
    X = cam.c[0] + t * rays[..., 0]
    Y = cam.c[1] + t * rays[..., 1]

    # --------------------------------------------------
    # Procedural world-space texture
    # --------------------------------------------------
    q0 = np.cos(angles[0]) * X + np.sin(angles[0]) * Y
    q1 = np.cos(angles[1]) * X + np.sin(angles[1]) * Y
    q2 = np.cos(angles[2]) * X + np.sin(angles[2]) * Y

    texture = (
        0.35 * np.sin(q0 / scales[0] + phases[0])
        + 0.30 * np.sin(q1 / scales[1] + phases[1])
        + 0.20 * np.sin(q2 / scales[2] + phases[2])
        + 0.15
        * np.sin(X / (scales[1] * 1.7) + phases[0])
        * np.cos(Y / (scales[1] * 1.3) + phases[1])
    )

    texture += 0.12 * (
        np.sin(7.13 * X + 3.71 * Y + phases[0])
        * np.sin(2.31 * X - 5.17 * Y + phases[1])
    )

    # --------------------------------------------------
    # Ground colour
    # --------------------------------------------------
    texture_rgb = texture[..., None] * np.array(
        [1.5, 1.9, 1.2],
        dtype=np.float32,
    )

    ground = (
        ground_base[None, None, :]
        + contrast * texture_rgb
    )

    # --------------------------------------------------
    # Atmospheric haze
    # --------------------------------------------------
    distance = np.sqrt(
        (X - cam.c[0]) ** 2
        + (Y - cam.c[1]) ** 2
    )

    haze = 0.65 * (
        1.0 - np.exp(-distance / 18.0)
    )

    ground = (
        (1.0 - haze[..., None]) * ground
        + haze[..., None] * haze_color[None, None, :]
    )

    # Copy ground only to pixels whose rays hit z=0.
    img[ground_mask] = ground[ground_mask]

    return np.clip(img, 0, 255).astype(np.uint8)

# -----------------------
# MAIN RENDER
# -----------------------
def render_camera_image(
    projections,
    colors,
    image_size=(256, 256),
    fov_u=1.0,
    fov_v=1.0,
    background=None,
    appearances=None,
    cam=None,
    background_seed=None,
):
    width, height = image_size

    if cam is not None:
        img = make_ground_background(
            image_size,
            cam,
            fov_u=fov_u,
            fov_v=fov_v,
            seed=background_seed,
            ground_z=0.0,
        )
    elif background is None:
        img = _load_random_background(image_size)
    else:
        img = np.full(
            (height, width, 3),
            background,
            dtype=np.uint8,
        )

    projections = sorted(projections, key=lambda x: x[5], reverse=True)

    for (i, u_min, u_max, v_min, v_max, d) in projections:
        clipped = _clip_rect_to_fov(u_min, u_max, v_min, v_max, fov_u, fov_v)
        if clipped is None:
            continue

        u0, u1, v0, v1 = clipped
        bounds = _rect_to_pixel_bounds(u0, u1, v0, v1, width, height, fov_u, fov_v)
        if bounds is None:
            continue

        r0, r1, c0, c1 = bounds
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        base_color = appearances[i]["color"]
        seed = appearances[i]["seed"]

        texture = make_bark_texture(h, w, base_color, seed)
        img[r0:r1 + 1, c0:c1 + 1] = texture

    return img


# -----------------------
# DEBUG RENDER
# -----------------------
def render_camera_image_debug(
    projections,
    colors,
    image_size=(256, 256),
    fov_u=1.0,
    fov_v=1.0,
    background=(255, 255, 255),
    alpha=0.35,
):
    """
    Debug-render med transparens och kanter.
    """

    width, height = image_size
    img = np.full((height, width, 3), background, dtype=np.float32)

    projections = sorted(projections, key=lambda x: x[5], reverse=True)

    for (i, u_min, u_max, v_min, v_max, d) in projections:
        clipped = _clip_rect_to_fov(u_min, u_max, v_min, v_max, fov_u, fov_v)
        if clipped is None:
            continue

        u0, u1, v0, v1 = clipped
        bounds = _rect_to_pixel_bounds(u0, u1, v0, v1, width, height, fov_u, fov_v)
        if bounds is None:
            continue

        r0, r1, c0, c1 = bounds
        rgb = _color_to_rgb255(colors[i]).astype(np.float32)

        # Fyll
        img[r0:r1 + 1, c0:c1 + 1] = (
            (1.0 - alpha) * img[r0:r1 + 1, c0:c1 + 1] + alpha * rgb
        )

        # Kanter
        img[r0:r0 + 1, c0:c1 + 1] = rgb
        img[r1:r1 + 1, c0:c1 + 1] = rgb
        img[r0:r1 + 1, c0:c0 + 1] = rgb
        img[r0:r1 + 1, c1:c1 + 1] = rgb

    return np.clip(img, 0, 255).astype(np.uint8)