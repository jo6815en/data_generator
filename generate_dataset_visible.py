"""Generate a small dataset with visible-cylinder occupancy and stable GT IDs.

Requires the original cylinder_generator.py, camera_model.py, render.py,
vision.py and the supplied visibility_labels.py in the same directory.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from cylinder_generator import generate_cylinders
from camera_model import create_camera_pair, compute_projections, transform_scene_to_cam1
from render import render_camera_image, random_bark_color
from vision import build_vision_pair
from visibility_labels import visible_projections


# Same geometry as the notebook, with a configurable camera baseline.
IMAGE_SIZE = (128, 128)  # width, height
FOV_U = FOV_V = 1.0
NUM_BINS = 128
VISION_FOV_DEGREES = 90.0
CYLINDER_SETTINGS = dict(num=20, xmin=-12, xmax=12, ymin=-8, ymax=8,
                         r_min=0.25, r_max=0.8, h=15.0)
CAMERA_SETTINGS = dict(
    camera_distance=1.0, camera_height=2.2, angle_jitter_deg=5,
    pitch_min_deg=-8.0, pitch_max_deg=8.0,
    min_cam_cyl_dist=0.02, max_cam_cyl_dist=8.0,
    near_tree_prob=0.35, near_tree_min_dist=0.05,
    near_tree_max_dist=0.50, min_visible=2,
)


def camera_to_dict(cam):
    # Preserve the original training convention: forward, left, up.
    R = np.stack([cam.dir, -cam.right, cam.up], axis=0)
    return dict(
        position=cam.c.tolist(), R=R.tolist(), t=(-R @ cam.c).tolist(),
        K=np.eye(3).tolist(), theta_xy=float(cam.theta_xy),
        pitch=float(cam.pitch), fov_degrees=float(np.rad2deg(cam.fov)),
        right=cam.right.tolist(), forward=cam.dir.tolist(), up=cam.up.tolist(),
    )


def cylinders_to_dict(cylinders):
    return [dict(id=i, x=float(x), y=float(y), z=float(z), r=float(r), h=float(h))
            for i, (x, y, z, r, h) in enumerate(cylinders)]


def generate_scene(scene_id, output, min_pairs, max_pairs,
                   min_pixels, min_fraction, seed, camera_distance=1.0):
    # Independent per-scene seeds, saved so this scene can be reproduced.
    random.seed(seed)
    np.random.seed(seed)
    camera_settings = dict(CAMERA_SETTINGS, camera_distance=camera_distance)
    world_cylinders, bounds = generate_cylinders(**CYLINDER_SETTINGS)
    appearances = [dict(color=random_bark_color().tolist(),
                        seed=random.randint(0, 1_000_000)) for _ in world_cylinders]
    background_seed = random.randint(0, 1_000_000)
    scene_dir = output / f"scene_{scene_id:03d}"
    # Refuse to overwrite an earlier experiment.
    scene_dir.mkdir(parents=True, exist_ok=False)
    (scene_dir / "images").mkdir()
    scene = dict(
        schema_version=3, scene_id=scene_id, seed=seed,
        cylinders_frame="world", camera_frame="pair_cam1_yaw_normalized",
        world_camera_frame="world", cylinder_id_rule="index in cylinders list",
        cylinders=cylinders_to_dict(world_cylinders),
        bounds=list(bounds), appearances=appearances, background_seed=background_seed,
        settings=dict(
            image_size=list(IMAGE_SIZE), fov_u=FOV_U, fov_v=FOV_V,
            num_bins=NUM_BINS, vision_fov_degrees=VISION_FOV_DEGREES,
            min_visible_pixels=min_pixels, min_visible_fraction=min_fraction,
            visibility_method="renderer_rectangle_id_map_v1",
            visible_fraction_denominator="clipped projected rectangle pixels",
            vision_columns=["occupancy", "radius", "radial_xy_depth", "cylinder_id"],
            bin_rule="floor of uniform relative XY bearing bins; lower bound inclusive",
            projection_columns=["cylinder_id", "u_min", "u_max", "v_min", "v_max", "radial_xy_depth"],
            camera_R_rows=["forward", "left", "up"],
            image_projection="u=dot(P-C,right)/dot(P-C,forward); v=dot(P-C,up)/dot(P-C,forward)",
            K_convention="unit-focal normalized coordinates, not pixel intrinsics",
            cylinder_settings=CYLINDER_SETTINGS, camera_settings=camera_settings,
            min_pairs=min_pairs, max_pairs=max_pairs,
        ),
        pairs=[],
    )
    for pair_id in range(np.random.randint(min_pairs, max_pairs + 1)):
        w1, w2, near_tree_idx = create_camera_pair(
            world_cylinders, bounds, **camera_settings)
        wp1, wp2 = compute_projections(w1, w2, world_cylinders)
        c1, c2, cylinders = transform_scene_to_cam1(w1, w2, world_cylinders)
        p1, p2 = compute_projections(c1, c2, cylinders)
        # Raw labels retained for comparison; training uses filtered vision1/2.
        raw1, raw2 = build_vision_pair(p1, p2, cylinders, num_bins=NUM_BINS,
            fov_degrees=VISION_FOV_DEGREES, cam1=c1, cam2=c2)
        entry = dict(
            pair_id=pair_id, near_tree_idx=near_tree_idx,
            camera1=camera_to_dict(c1), camera2=camera_to_dict(c2),
            world_camera1=camera_to_dict(w1), world_camera2=camera_to_dict(w2),
            cylinders_pair=cylinders_to_dict(cylinders),
            cylinders_pair_frame="pair_cam1_yaw_normalized",
            vision1_all=raw1.tolist(), vision2_all=raw2.tolist(),
        )
        filtered = []
        for view, wc, wp, projections in ((1, w1, wp1, p1), (2, w2, wp2, p2)):
            # Visibility is measured from precisely the world projections used
            # by the RGB renderer. Labels use the matching normalized geometry.
            kept, stats, _ = visible_projections(
                wp, len(world_cylinders), image_size=IMAGE_SIZE,
                fov_u=FOV_U, fov_v=FOV_V,
                min_pixels=min_pixels, min_fraction=min_fraction)
            ids = {p[0] for p in kept}
            filtered.append([p for p in projections if p[0] in ids])
            base = f"pair_{pair_id:03d}_cam{view}"
            image_path = f"images/{base}.png"
            image = render_camera_image(
                wp, {}, image_size=IMAGE_SIZE, fov_u=FOV_U, fov_v=FOV_V,
                appearances=appearances, cam=wc, background_seed=background_seed)
            Image.fromarray(image).save(scene_dir / image_path)
            entry.update({f"image{view}": image_path,
                          f"visibility{view}": stats,
                          f"world_projections{view}": [list(p) for p in wp]})
        mat1, mat2 = build_vision_pair(
            *filtered, cylinders, num_bins=NUM_BINS,
            fov_degrees=VISION_FOV_DEGREES, cam1=c1, cam2=c2)
        entry.update(vision1=mat1.tolist(), vision2=mat2.tolist())
        scene["pairs"].append(entry)
        print(f"scene {scene_id} pair {pair_id}: positive bins "
              f"{int(raw1[:, 0].sum())}->{int(mat1[:, 0].sum())}, "
              f"{int(raw2[:, 0].sum())}->{int(mat2[:, 0].sum())}")
    (scene_dir / "labels.json").write_text(json.dumps(scene, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dataset_visible_test")
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-visible-pixels", type=int, default=1)
    parser.add_argument("--min-visible-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-distance", type=float, default=1.0,
                        help="Distance between the two cameras in scene units (meters)")
    args = parser.parse_args()
    if args.num_scenes < 1 or not 1 <= args.min_pairs <= args.max_pairs:
        parser.error("Require positive scene count and 1 <= min-pairs <= max-pairs")
    if not np.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("--camera-distance must be positive and finite")
    output = Path(args.output_dir)
    for scene_id in range(args.num_scenes):
        generate_scene(scene_id, output, args.min_pairs, args.max_pairs,
                       args.min_visible_pixels, args.min_visible_fraction,
                       args.seed + scene_id, args.camera_distance)


if __name__ == "__main__":
    main()
