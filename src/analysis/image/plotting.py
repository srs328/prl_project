"""3D visualization for lesion and rim analysis.

Functions:
    plot_lesion_rim_3d: 3D scatter plot of rim voxels, convex hull, and enclosing sphere.
"""

from __future__ import annotations

import numpy as np


def plot_lesion_rim_3d(
    lesion_data: dict,
    label_source: str = "infer",
    alpha_wireframe: float = 0.15,
    alpha_lesion: float = 0.05,
    alpha_hull: float = 0.4,
    alpha_rim: float = 0.3,
):
    """3D scatter plot of rim voxels with convex hull and enclosing sphere.

    Args:
        lesion_data: Dict from analyze_prl_case containing rim masks, hulls, voxel sizes.
        label_source: "infer" or "truth" — which label source to plot.
        alpha_wireframe: Transparency of sphere wireframe.
        alpha_lesion: Transparency of lesion voxel scatter.
        alpha_hull: Transparency of convex hull edges.
        alpha_rim: Transparency of rim voxel scatter.
    """
    import matplotlib.pyplot as plt

    klab = label_source
    rim = lesion_data[f"rim_{klab}"]
    voxel_size = lesion_data[f"voxel_size_{klab}"]
    coords = np.argwhere(rim) * np.array(voxel_size)

    hull = lesion_data[f"rim_hull_{klab}"]

    vertices = coords[hull.vertices]
    center = vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Rim voxels
    ax.scatter(*coords.T, c="red", alpha=alpha_rim, s=10, label="Rim voxels")

    # Convex hull faces
    for simplex in hull.simplices:
        pts = coords[simplex]
        tri = np.vstack([pts, pts[0]])
        ax.plot(*tri.T, "b-", alpha=alpha_hull, linewidth=0.5)

    # Enclosing sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="green", alpha=alpha_wireframe, linewidth=0.3)

    # Lesion voxels for context
    lesion_key = f"lesion_{klab}"
    if lesion_key in lesion_data:
        lesion_coords = np.argwhere(lesion_data[lesion_key] == 1) * np.array(voxel_size)
        ax.scatter(
            *lesion_coords.T, c="grey", alpha=alpha_lesion, s=2, label="Lesion"
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.set_title(f"Hull vol={hull.volume:.1f} mm\u00b3, sphere r={radius:.2f} mm")
    plt.tight_layout()
    plt.show()
