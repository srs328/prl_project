"""Pure geometry functions for 3D mask analysis.

Functions:
    get_convex_hull: ConvexHull of binary mask voxels.
    rim_convex_hull_volume: Convex hull volume in mm^3.
    rim_enclosing_sphere_radius: Radius of smallest enclosing sphere in mm.
    compute_pca_features: PCA-based shape descriptors (sphericity, planarity, linearity).
    compute_radial_metrics: Radial distance statistics from mask centroid.
    
Some considerations:
- How expensive is the computation of convex hull? Should I be careful about only computing it once and making
    sure that it is simply passed to other functions?
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy import ndimage


def get_convex_hull(
    mask: np.ndarray, voxel_sizes: tuple[float, ...] | None = None
) -> ConvexHull | None:
    """Convex hull of binary mask voxels.

    Args:
        mask: 3D boolean/binary array.
        voxel_sizes: If provided, scale voxel coords to mm before computing hull.

    Returns:
        ConvexHull object, or None if fewer than 4 non-coplanar points.
    """
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return None

    if voxel_sizes is not None:
        coords = coords * np.array(voxel_sizes)
    try:
        return ConvexHull(coords)
    except Exception:
        return None


def rim_convex_hull_volume(
    rim_mask: np.ndarray, voxel_sizes: tuple[float, ...]
) -> float | None:
    """Convex hull volume of rim voxels in mm^3.

    Returns None if fewer than 4 non-coplanar rim voxels.
    """
    coords = np.argwhere(rim_mask)
    if len(coords) < 4:
        return None
    coords_mm = coords * np.array(voxel_sizes)
    try:
        hull = ConvexHull(coords_mm)
        return float(hull.volume)
    except Exception:
        return None


def rim_enclosing_sphere_radius(
    rim_mask: np.ndarray, voxel_sizes: tuple[float, ...]
) -> float | None:
    """Radius (mm) of the smallest sphere enclosing all rim voxels.

    Uses convex hull vertices + centroid approach (guaranteed enclosing,
    not mathematically minimal but close for typical rim shapes).
    Returns None if no rim voxels.
    """
    coords = np.argwhere(rim_mask)
    if len(coords) == 0:
        return None
    coords_mm = coords * np.array(voxel_sizes)
    if len(coords) < 4:
        center = coords_mm.mean(axis=0)
        return float(np.max(np.linalg.norm(coords_mm - center, axis=1)))
    try:
        hull = ConvexHull(coords_mm)
        vertices = coords_mm[hull.vertices]
    except Exception:
        vertices = coords_mm
    center = vertices.mean(axis=0)
    return float(np.max(np.linalg.norm(vertices - center, axis=1)))


def compute_pca_features(rim_mask: np.ndarray) -> dict[str, float]:
    """PCA-based shape descriptors for a binary rim mask.

    Fits PCA on voxel coordinates and derives shape ratios from the
    three singular values (largest to smallest: λ1, λ2, λ3):

    - **pca_size**: sum of singular values (overall spread).
    - **pca_sphericity**: λ3/λ1 — 1.0 for a perfect sphere, 0 for flat/linear.
    - **pca_planarity**: (λ2−λ3)/λ1 — high when the shape is disc-like.
    - **pca_linearity**: (λ1−λ2)/λ1 — high when the shape is elongated.

    Returns NaN values if fewer than 3 rim voxels.
    """
    from sklearn.decomposition import PCA
    coords = np.argwhere(rim_mask > 0)
    if len(coords) < 3:
        return {
            "pca_size": np.nan,
            "pca_sphericity": np.nan,
            "pca_planarity": np.nan,
            "pca_linearity": np.nan,
        }

    pca = PCA(n_components=3)
    pca.fit(coords)

    comps = sorted(pca.singular_values_, reverse=True)

    features = {}
    features["pca_size"] = np.sum(comps)
    features["pca_sphericity"] = comps[2] / comps[0] if comps[0] > 0 else np.nan
    features["pca_planarity"] = (comps[1] - comps[2]) / comps[0] if comps[0] > 0 else np.nan
    features["pca_linearity"] = (comps[0] - comps[1]) / comps[0] if comps[0] > 0 else np.nan
    return features


def compute_radial_metrics(rim_mask: np.ndarray) -> dict[str, float]:
    """Radial distance statistics of rim voxels from the mask centroid.

    Computes the Euclidean distance of each rim voxel to the center of
    mass and returns summary statistics:

    - **mean_radius**, **std_radius**, **min_radius**, **max_radius**
    - **radial_cv**: coefficient of variation (std/mean) — low values
      indicate a uniform shell, high values indicate irregular spread.

    Distances are in voxel units. Returns NaN values if no rim voxels.
    """

    coords = np.argwhere(rim_mask > 0)

    if len(coords) == 0:
        return {
            "mean_radius": np.nan,
            "std_radius": np.nan,
            "min_radius": np.nan,
            "max_radius": np.nan,
            "radial_cv": np.nan,
        }

    centroid = ndimage.center_of_mass(rim_mask)
    dists = np.linalg.norm(coords - centroid, axis=1)

    features = {}
    features["mean_radius"] = dists.mean()
    features["std_radius"] = dists.std()
    features["min_radius"] = dists.min()
    features["max_radius"] = dists.max()
    features["radial_cv"] = (
        features["std_radius"] / features["mean_radius"]
        if features["mean_radius"] > 0 else np.nan
    )
    return features


def lesion_pmap_features(rim_mask: np.ndarray, lesion_pmap: np.ndarray) -> dict:
    """Extract features based on the lesion probability map for classification
    This is a function I could add more features to as I think of them. First thoughts:
    - Some cumulative measure of probability within the iron rim (e.g. within the convex hull)
    - Probability at the centroid of the rim_mask
    """
    centroid = ndimage.center_of_mass(rim_mask)
    
    hull = get_convex_hull(rim_mask)
    features = {}
    return features