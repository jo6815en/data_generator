import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# Rita kamera
# -----------------------
def draw_camera(ax, cam, color, arrow_len=1.5, marker_size=45, label=None):
    c = np.asarray(cam.c, dtype=float)
    d = np.asarray(cam.dir[:2], dtype=float)

    n = np.linalg.norm(d)
    if n < 1e-12:
        return
    d = d / n

    ax.scatter(c[0], c[1], s=marker_size, color=color, zorder=5)
    ax.annotate(
        "",
        xy=(c[0] + d[0] * arrow_len, c[1] + d[1] * arrow_len),
        xytext=(c[0], c[1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=18),
        zorder=6,
    )

    if label is not None:
        ax.text(c[0], c[1], label, color=color, fontsize=10, ha="left", va="bottom")


# -----------------------
# Rita FOV-cone
# -----------------------
def draw_fov_cone(ax, cam, fov_u, length=5.0, color="black"):
    p = cam.c[:2]
    d = cam.dir[:2]
    d = d / np.linalg.norm(d)

    angle = np.arctan(fov_u)

    def rot(vec, a):
        return np.array(
            [
                np.cos(a) * vec[0] - np.sin(a) * vec[1],
                np.sin(a) * vec[0] + np.cos(a) * vec[1],
            ]
        )

    left = rot(d, angle)
    right = rot(d, -angle)

    ax.plot(
        [p[0], p[0] + left[0] * length],
        [p[1], p[1] + left[1] * length],
        linestyle="--",
        color=color,
    )
    ax.plot(
        [p[0], p[0] + right[0] * length],
        [p[1], p[1] + right[1] * length],
        linestyle="--",
        color=color,
    )

    cone_x = [p[0], p[0] + left[0] * length, p[0] + right[0] * length]
    cone_y = [p[1], p[1] + left[1] * length, p[1] + right[1] * length]
    ax.fill(cone_x, cone_y, color=color, alpha=0.1)


# -----------------------
# Top view
# -----------------------
def plot_top_view_scene(ax, cylinders, cam1, cam2, colors=None, pad=0.5, title="Top view"):
    """
    Plottar cylindrar och två kameror i top view (xy-planet).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axeln att rita på.
    cylinders : list[tuple]
        Lista med cylindrar i något av formaten:
        - (x, y, r, h)
        - (x, y, z, r, h)
        Endast x och y används i top view.
    cam1, cam2 : Camera3D
        Kamerorna som ska ritas.
    colors : list or None
        Valfria färger för cylindrarna.
    pad : float
        Marginal runt all data.
    title : str
        Titel för plotten.
    """

    if colors is None:
        colors = [f"C{i % 10}" for i in range(len(cylinders))]

    all_x = []
    all_y = []

    for cyl in cylinders:
        if len(cyl) == 5:
            x, y, z, r, h = cyl
        else:
            x, y, r, h = cyl
        all_x.append(x)
        all_y.append(y)

    all_x += [cam1.c[0], cam2.c[0]]
    all_y += [cam1.c[1], cam2.c[1]]

    xmin, xmax = min(all_x) - pad, max(all_x) + pad
    ymin, ymax = min(all_y) - pad, max(all_y) + pad

    for i, cyl in enumerate(cylinders):
        if len(cyl) == 5:
            x, y, z, r, h = cyl
        else:
            x, y, r, h = cyl

        circ = plt.Circle((x, y), r, fill=False, color=colors[i])
        ax.add_patch(circ)
        ax.text(x, y, str(i), ha="center", va="center")

    draw_camera(ax, cam1, "red")
    draw_camera(ax, cam2, "blue")

    draw_fov_cone(ax, cam1, fov_u=np.tan(cam1.fov / 2), color="red")
    draw_fov_cone(ax, cam2, fov_u=np.tan(cam2.fov / 2), color="blue")

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)