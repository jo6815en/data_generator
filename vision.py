import numpy as np

# -----------------------
# Bygg vision matrix
# -----------------------
def build_vision_matrix(projections, cylinders,
                        num_bins=32, u_min=-1.0, u_max=1.0):
    """
    projections: [(idx, u_min, u_max, v_min, v_max, d)]
    cylinders: [(x, y, r, h)]
    """

    bin_size = (u_max - u_min) / num_bins

    # [occupancy, radius, depth]
    mat = np.zeros((num_bins, 3))

    for (idx, u0, u1, v_min, v_max, d) in projections:

        r = cylinders[idx][3]

        # klipp till FOV
        u0_clip = max(u0, u_min)
        u1_clip = min(u1, u_max)

        if u0_clip >= u1_clip:
            continue

        # bin-index
        b_start = int((u0_clip - u_min) / bin_size)
        b_end   = int((u1_clip - u_min) / bin_size)

        for b in range(b_start, b_end + 1):
            if b < 0 or b >= num_bins:
                continue

            # välj närmaste objekt (occlusion)
            if mat[b, 0] == 0 or d < mat[b, 2]:
                mat[b, 0] = 1      # occupancy
                mat[b, 1] = r      # radius
                mat[b, 2] = d      # depth

    return mat


# -----------------------
# Convenience
# -----------------------
def build_vision_pair(proj1, proj2, cylinders,
                      num_bins=32, u_min=-1.0, u_max=1.0):

    mat1 = build_vision_matrix(proj1, cylinders, num_bins, u_min, u_max)
    mat2 = build_vision_matrix(proj2, cylinders, num_bins, u_min, u_max)

    return mat1, mat2