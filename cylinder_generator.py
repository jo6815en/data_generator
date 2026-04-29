import numpy as np
import random

# -----------------------
# Overlap (i XY-plan)
# -----------------------
def is_overlapping(x, y, r, cylinders):
    for (cx, cy, cr, _) in cylinders:
        if np.hypot(x - cx, y - cy) < (r + cr):
            return True
    return False


# -----------------------
# Generera cylindrar
# -----------------------
def generate_cylinders(
    num=5,
    xmin=-2, xmax=6,
    ymin=-3, ymax=3,
    r_min=0.3, r_max=0.8,
    h=50,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    cylinders = []

    while len(cylinders) < num:
        r = random.uniform(r_min, r_max)
        x = random.uniform(xmin + r, xmax - r)
        y = random.uniform(ymin + r, ymax - r)

        if not is_overlapping(x, y, r, cylinders):
            cylinders.append((x, y, r, h))

    return cylinders, (xmin, xmax, ymin, ymax)