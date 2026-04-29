import numpy as np
import random

# -----------------------
# Overlap-check
# -----------------------
def is_overlapping(x, y, r, circles):
    for (cx, cy, cr) in circles:
        if np.hypot(x - cx, y - cy) < (r + cr):
            return True
    return False


# -----------------------
# Generera cirklar
# -----------------------
def generate_circles(
    num_circles=5,
    xmin=-2, xmax=6,
    ymin=-3, ymax=3,
    r_min=0.3, r_max=0.8,
    seed=None
):
    if seed is not None:
        random.seed(seed)
    
    circles = []
    
    while len(circles) < num_circles:
        r = random.uniform(r_min, r_max)
        x = random.uniform(xmin + r, xmax - r)
        y = random.uniform(ymin + r, ymax - r)
        
        if not is_overlapping(x, y, r, circles):
            circles.append((x, y, r))
    
    return circles, (xmin, xmax, ymin, ymax)