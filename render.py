

def render_camera_image(cam, projections, cylinders,
                        colors,
                        image_size=(256, 256),
                        fov_u=0.5, fov_v=1.0):

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(3, 3), dpi=image_size[0] // 3)

    projections = sorted(projections, key=lambda x: x[5], reverse=True)

    for (i, u_min, u_max, v1, v2, d) in projections:
        r = cylinders[i][2]

        # --- KLIPP I U (horisontellt FOV) ---
        if u_max < -fov_u or u_min > fov_u:
            continue

        # --- NORMALISERA V (höjd fyller hela bilden) ---
        v_center = (v1 + v2) / 2
        half_height = abs(v2 - v1) / 2

        if half_height < 1e-6:
            continue

        scale = fov_v / half_height

        v1_scaled = (v1 - v_center) * scale
        v2_scaled = (v2 - v_center) * scale

        # säkerställ ordning
        vmin = min(v1_scaled, v2_scaled)
        vmax = max(v1_scaled, v2_scaled)

        # klipp till FOV
        vmin = max(vmin, -fov_v)
        vmax = min(vmax,  fov_v)

        if vmin >= vmax:
            continue

        # --- RITA ---
        ax.fill_betweenx([vmin, vmax],
                         u_min, u_max,
                         color=colors[i])

    # --- FIX VIEW ---
    ax.set_xlim(fov_u, -fov_u)   # spegling fix
    ax.set_ylim(-fov_v, fov_v)

    ax.set_aspect('equal')
    ax.axis('off')

    # --- RENDER TILL NUMPY ---
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[..., :3]

    plt.close(fig)

    return image