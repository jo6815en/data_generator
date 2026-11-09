"""Visibility labels matching the supplied rectangle renderer exactly.

Place beside render.py and vision.py. Cylinder IDs remain list indices.
Visibility is measured at the original render resolution, before augmentation.
"""
import numpy as np

from render import _clip_rect_to_fov, _rect_to_pixel_bounds
from vision import build_vision_pair


def visible_projections(projections, num_cylinders, image_size=(128, 128),
                        fov_u=1.0, fov_v=1.0, min_pixels=1,
                        min_fraction=0.0):
    """Return filtered projections, per-ID statistics and an HxW ID map.

    fraction = final visible pixels / clipped projected rectangle pixels.
    Background ID is -1. A fully hidden cylinder never passes the filter.
    Both thresholds must pass; default keeps every cylinder with >=1 pixel.
    Uses the same far-to-near ordering and inclusive bounds as render.py.
    """
    if min_pixels < 1 or not 0 <= min_fraction <= 1:
        raise ValueError('min_pixels >= 1 and 0 <= min_fraction <= 1 required')
    width, height = image_size
    if width < 2 or height < 2 or fov_u <= 0 or fov_v <= 0:
        raise ValueError('Invalid image size or FOV')
    projections = list(projections)
    ids = [int(p[0]) for p in projections]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= num_cylinders for i in ids):
        raise ValueError('Expected one projection per valid cylinder ID')
    id_map = np.full((height, width), -1, dtype=np.int32)
    projected = np.zeros(num_cylinders, dtype=np.int64)
    for cid, u0, u1, v0, v1, depth in sorted(
            projections, key=lambda p: p[5], reverse=True):
        clipped = _clip_rect_to_fov(u0, u1, v0, v1, fov_u, fov_v)
        if clipped is None:
            continue
        bounds = _rect_to_pixel_bounds(
            *clipped, width, height, fov_u, fov_v)
        if bounds is None:
            continue
        r0, r1, c0, c1 = bounds
        projected[cid] = (r1 - r0 + 1) * (c1 - c0 + 1)
        id_map[r0:r1 + 1, c0:c1 + 1] = cid
    visible = np.bincount(id_map[id_map >= 0], minlength=num_cylinders)
    fraction = np.divide(visible, projected, out=np.zeros(num_cylinders),
                         where=projected > 0)
    keep = (visible >= min_pixels) & (fraction >= min_fraction)
    stats = [dict(cylinder_id=i, projected_pixels=int(projected[i]),
                  visible_pixels=int(visible[i]), visible_fraction=float(fraction[i]),
                  keep=bool(keep[i])) for i in range(num_cylinders)]
    # Filter after all occluders have been rasterized. Rejected objects still
    # occlude objects behind them in the actual image.
    return [p for p in projections if keep[p[0]]], stats, id_map


def build_visible_vision_pair(proj1, proj2, cylinders, num_bins=128,
                              fov_degrees=90.0, cam1=None, cam2=None,
                              image_size=(128, 128), fov_u=1.0, fov_v=1.0,
                              min_pixels=1, min_fraction=0.0):
    """Rebuild both matrices from visible candidates, preserving original IDs.

    Projections, cylinders and cameras must share the same coordinate frame.
    Returned metadata includes all IDs, even those without a centre in the FOV
    or excluded from the matrix by a same-bin collision.
    """
    opts = dict(image_size=image_size, fov_u=fov_u, fov_v=fov_v,
                min_pixels=min_pixels, min_fraction=min_fraction)
    p1, stats1, _ = visible_projections(proj1, len(cylinders), **opts)
    p2, stats2, _ = visible_projections(proj2, len(cylinders), **opts)
    mat1, mat2 = build_vision_pair(
        p1, p2, cylinders, num_bins=num_bins, fov_degrees=fov_degrees,
        cam1=cam1, cam2=cam2)
    return mat1, mat2, dict(
        visibility1=stats1, visibility2=stats2,
        settings=dict(image_size=list(image_size), fov_u=fov_u, fov_v=fov_v,
                      num_bins=num_bins, fov_degrees=fov_degrees,
                      min_pixels=min_pixels, min_fraction=min_fraction,
                      method='renderer_rectangle_id_map_v1'))
