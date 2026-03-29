"""Image-level analysis for PRL lesion diagnostics.

Submodules:
    geometry — convex hull, enclosing sphere
    lesion_analysis — rim extraction, PRL case analysis
    plotting — 3D visualization
"""

from analysis.image.geometry import (
    get_convex_hull,
    rim_convex_hull_volume,
    rim_enclosing_sphere_radius,
)
from analysis.image.lesion_analysis import (
    parse_bounding_boxes,
    crop_from_volume,
    get_center_lesion,
    get_center_label,
    get_lesion_rim,
    count_rim_for_lesion,
    analyze_prl_case,
)
