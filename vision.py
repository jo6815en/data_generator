import numpy as np

# -----------------------
# Vision matrix parametrar
# -----------------------
FOV_MIN = -np.pi / 2
FOV_MAX = np.pi / 2


# -----------------------
# Bygg vision matrix
# -----------------------
def build_vision_matrix(vis, projections, circles, num_bins=32):
    """
    vis: [(idx, umin, umax)]
    projections: [(idx, umin, umax, d)]
    circles: [(x, y, r)]
    """

    bin_size = (FOV_MAX - FOV_MIN) / num_bins

    # [occupancy, radius, depth]
    mat = np.zeros((num_bins, 3))

    # lookup för depth
    depth_dict = {idx: d for (idx, _, _, d) in projections}

    for idx, umin, umax in vis:
        R = circles[idx][2]
        d = depth_dict[idx]

        # klipp till FOV
        umin_clipped = max(umin, FOV_MIN)
        umax_clipped = min(umax, FOV_MAX)

        if umin_clipped >= umax_clipped:
            continue

        # bin-index
        bin_start = int((umin_clipped - FOV_MIN) / bin_size)
        bin_end   = int((umax_clipped - FOV_MIN) / bin_size)

        for b in range(bin_start, bin_end + 1):
            if b < 0 or b >= num_bins:
                continue

            # välj närmaste objekt
            if mat[b, 0] == 0 or d < mat[b, 2]:
                mat[b, 0] = 1
                mat[b, 1] = R
                mat[b, 2] = d

    return mat


# -----------------------
# Convenience: båda kameror
# -----------------------
def build_vision_pair(proj1, vis1, proj2, vis2, circles, num_bins=32):
    mat1 = build_vision_matrix(vis1, proj1, circles, num_bins)
    mat2 = build_vision_matrix(vis2, proj2, circles, num_bins)

    return mat1, mat2