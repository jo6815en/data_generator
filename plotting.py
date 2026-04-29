import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Rita kamera
# -----------------------
def draw_camera(ax, cam, color):
    c = cam.c
    d = cam.dir[:2]
    d = d / np.linalg.norm(d)

    arrow_len = 1.5  # 🔥 justera denna

    ax.plot(c[0], c[1], 'o', color=color)
    ax.arrow(
        c[0], c[1],
        d[0] * arrow_len,
        d[1] * arrow_len,
        head_width=0.2,
        color=color
    )


# -----------------------
# Plotta ett kamerapar
# -----------------------
def plot_camera_pair(
    axs,
    circles,
    cam1,
    cam2,
    proj1_all,
    vis1,
    proj2_all,
    vis2,
    bounds,
    colors
):
    xmin, xmax, ymin, ymax = bounds

    # --- Värld ---
    ax = axs[0]
    for i, (x, y, r) in enumerate(circles):
        circ = plt.Circle((x, y), r, fill=False, color=colors[i])
        ax.add_patch(circ)
        ax.text(x, y, str(i), fontsize=10, ha='center', va='center')

    draw_camera(ax, cam1, 'red')
    draw_camera(ax, cam2, 'blue')

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Värld")

    # --- Kamera 1 ---
    ax = axs[1]
    for idx, umin, umax, d in proj1_all:
        ax.plot([umin, umax], [0, 0], color='gray', linewidth=1)

    for idx, umin, umax in vis1:
        ax.plot([umin, umax], [0, 0], linewidth=4, color=colors[idx])
        ax.text((umin+umax)/2, 0.1, str(idx), ha='center', fontsize=9)

    ax.set_ylim(-1, 1)
    ax.set_xlim(-np.pi/2, np.pi/2)
    ax.set_title("Cam1")

    # --- Kamera 2 ---
    ax = axs[2]
    for idx, umin, umax, d in proj2_all:
        ax.plot([umin, umax], [0, 0], color='gray', linewidth=1)

    for idx, umin, umax in vis2:
        ax.plot([umin, umax], [0, 0], linewidth=4, color=colors[idx])
        ax.text((umin+umax)/2, 0.1, str(idx), ha='center', fontsize=9)

    ax.set_ylim(-1, 1)
    ax.set_xlim(-np.pi/2, np.pi/2)
    ax.set_title("Cam2")


    import numpy as np

def draw_fov_cone(ax, cam, fov_u, length=5.0, color='black'):
    p = cam.c[:2]
    d = cam.dir[:2]
    d = d / np.linalg.norm(d)

    # FOV-vinkel (från u)
    angle = np.arctan(fov_u)

    # rotationsmatriser
    def rot(vec, a):
        return np.array([
            np.cos(a)*vec[0] - np.sin(a)*vec[1],
            np.sin(a)*vec[0] + np.cos(a)*vec[1]
        ])

    left = rot(d, angle)
    right = rot(d, -angle)

    # rita cone-linjer
    ax.plot([p[0], p[0] + left[0]*length],
            [p[1], p[1] + left[1]*length],
            linestyle='--', color=color)

    ax.plot([p[0], p[0] + right[0]*length],
            [p[1], p[1] + right[1]*length],
            linestyle='--', color=color)

    # fyll cone (valfritt men snyggt)
    cone_x = [p[0],
              p[0] + left[0]*length,
              p[0] + right[0]*length]

    cone_y = [p[1],
              p[1] + left[1]*length,
              p[1] + right[1]*length]

    ax.fill(cone_x, cone_y, color=color, alpha=0.1)

    print("FOV:", -fov_u, fov_u)

    