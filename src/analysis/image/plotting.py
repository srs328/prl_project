"""3D visualization for lesion and rim analysis.

Functions:
    plot_lesion_rim_3d: 3D scatter plot of rim voxels, convex hull, and enclosing sphere.
"""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def generate_random_colors_names(N, seed=42) -> list[str]:
  """
  Generates a list of N random color names from the CSS4 specification.
  """
  rng = np.random.default_rng(seed)
  css4_colors = list(mcolors.CSS4_COLORS.keys())
  # Randomly choose N colors from the list
  colors = [rng.choice(css4_colors) for _ in range(N)]
  return colors


def plot_lesion_rim_3d_orig(
    lesion_data: dict,
    label_source: str = "infer",
    rim: dict | None = None,
    hull: dict | None = None,
    wireframe: dict | None = None,
    lesion: dict | None = None,
    points: list[dict] | None = None,
):
    """3D scatter plot of rim voxels with convex hull and enclosing sphere.

    Each component is controlled by passing a dict of matplotlib kwargs (or an empty
    dict for defaults). Pass None to skip that component entirely.

    Args:
        lesion_data: Dict from analyze_prl_case containing rim masks, hulls, voxel sizes.
        label_source: "infer" or "truth" — which label source to plot.
        rim: Kwargs for rim voxel scatter (default: ``{"c": "red", "alpha": 0.3, "s": 10}``).
        hull: Kwargs for convex hull edges (default: ``{"alpha": 0.4, "linewidth": 0.5}``).
        wireframe: Kwargs for enclosing sphere wireframe (default: ``{"color": "green", "alpha": 0.15, "linewidth": 0.3}``).
        lesion: Kwargs for lesion voxel scatter (default: ``{"c": "grey", "alpha": 0.05, "s": 2}``).
        points: List of individual points to plot. Each entry is a dict with a required
            ``"coords"`` key (x, y, z) and any matplotlib scatter kwargs
            (``c``, ``s``, ``marker``, ``label``, ``alpha``, etc.).
            Example: ``[{"coords": (10, 20, 15), "c": "blue", "s": 80, "marker": "*"}]``
    """

    def _kw(defaults, override):
        return defaults | override if override is not None else None

    rim_kwargs = _kw({"c": "red", "alpha": 0.3, "s": 10}, rim)
    hull_kwargs = _kw({"alpha": 0.4, "linewidth": 0.5}, hull)
    wireframe_kwargs = _kw({"color": "green", "alpha": 0.15, "linewidth": 0.3}, wireframe)
    lesion_kwargs = _kw({"c": "grey", "alpha": 0.05, "s": 2}, lesion)

    # Default: show all components if none are explicitly specified
    if rim is None and hull is None and wireframe is None and lesion is None:
        rim_kwargs = {"c": "red", "alpha": 0.3, "s": 10}
        hull_kwargs = {"alpha": 0.4, "linewidth": 0.5}
        wireframe_kwargs = {"color": "green", "alpha": 0.15, "linewidth": 0.3}
        lesion_kwargs = {"c": "grey", "alpha": 0.05, "s": 2}

    klab = label_source
    rim_mask = lesion_data[f"rim_{klab}"]
    voxel_size = lesion_data[f"voxel_size_{klab}"]
    coords = np.argwhere(rim_mask) * np.array(voxel_size)

    conv_hull = lesion_data[f"rim_hull_{klab}"]

    vertices = coords[conv_hull.vertices]
    center = vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if rim_kwargs is not None:
        ax.scatter(*coords.T, label="Rim voxels", **rim_kwargs)

    if hull_kwargs is not None:
        for simplex in conv_hull.simplices:
            pts = coords[simplex]
            tri = np.vstack([pts, pts[0]])
            ax.plot(*tri.T, "b-", **hull_kwargs)

    if wireframe_kwargs is not None:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, **wireframe_kwargs)

    if lesion_kwargs is not None:
        lesion_key = f"lesion_{klab}"
        if lesion_key in lesion_data:
            lesion_coords = np.argwhere(lesion_data[lesion_key] == 1) * np.array(voxel_size)
            ax.scatter(*lesion_coords.T, label="Lesion", **lesion_kwargs)

    if points:
        for pt in points:
            pt = dict(pt)
            x, y, z = pt.pop("coords")
            pt.setdefault("s", 60)
            pt.setdefault("marker", "o")
            ax.scatter([x], [y], [z], **pt)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.set_title(f"Hull vol={conv_hull.volume:.1f} mm\u00b3, sphere r={radius:.2f} mm")
    # plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_3d(images: list[np.ndarray], colors=None, slice_index=None, random_seed=42, **kwargs):
    """
    images: list of images to plot
    """
    if slice_index is None:
        slice_index = np.s_[:,:,:]
        
    if colors is None:
        colors = generate_random_colors_names(len(images), seed=random_seed)
    elif len(colors) < len(images):
        colors = colors + generate_random_colors_names(len(images) - len(colors))
    
    if "alphas" not in kwargs:
        alphas = [1]*len(images)
    else:
        alphas = kwargs['alphas']
        
    if "sizes" not in kwargs:
        sizes = [5]*len(images)
    else:
        sizes = kwargs['sizes']
    if len(sizes) < len(images):
        sizes = sizes + [5]*(len(images) - len(sizes))
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    for i, mask in enumerate(images):
        mask = mask[slice_index]
        coords = np.argwhere(mask).T
        ax.scatter(*coords, c=colors[i], alpha=alphas[i], s=sizes[i])
     
    ax.set_xlim([0, images[0].shape[0]])
    ax.set_ylim([0, images[0].shape[1]])
    ax.set_zlim([0, images[0].shape[2]])   
    ax.set_aspect('equal')
    ax.view_init(elev=-90, roll=90, azim=0)
    return fig, ax
    
