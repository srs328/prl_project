"""Image-level analysis for PRL lesion diagnostics.

Submodules:
    geometry — convex hull, enclosing sphere
    lesion_analysis — compatibility imports for classification.lesion_analysis
    plotting — 3D visualization
"""

from analysis.image.geometry import (
    get_convex_hull,
    rim_convex_hull_volume,
    rim_enclosing_sphere_radius,
)

_LESION_EXPORTS = (
    "parse_bounding_boxes",
    "crop_from_volume",
    "get_center_lesion",
    "get_center_label",
    "get_lesion_rim",
    "count_rim_for_lesion",
    "analyze_prl_case",
)

__all__ = [
    "get_convex_hull",
    "rim_convex_hull_volume",
    "rim_enclosing_sphere_radius",
    *_LESION_EXPORTS,
]


def __getattr__(name):
    if name in _LESION_EXPORTS:
        from classification import lesion_analysis

        return getattr(lesion_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
