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


# -----------------------
# MAIN RENDER
# -----------------------
def render_camera_image(
    projections,
    colors,
    image_size=(256, 256),
    fov_u=1.0,
    fov_v=1.0,
    background=None,   # None => slumpad bakgrund från backgrounds/
):
    width, height = image_size

    if background is None:
        img = _load_random_background(image_size)
    else:
        img = np.full((height, width, 3), background, dtype=np.uint8)

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
        img[r0:r1 + 1, c0:c1 + 1] = _color_to_rgb255(colors[i])

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