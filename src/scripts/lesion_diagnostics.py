"""Compatibility imports for notebooks that used the diagnostics script."""

from analysis.image.geometry import rim_convex_hull_volume
from classification.lesion_analysis import (
    analyze_prl_case,
    count_predicted_prls,
    count_rim_for_lesion as _count_rim_for_lesion,
    crop_from_volume as _crop_from_volume,
    get_center_label,
    get_center_lesion as _get_center_lesion,
    get_convex_hull,
    get_lesion_rim as _get_lesion_rim,
    parse_bounding_boxes as _parse_bounding_boxes,
    rim_enclosing_sphere_radius,
)


def plot_lesion_rim_3d(
    prl_case: dict,
    label_source: str = "infer",
    alpha_wireframe: float = 0.15,
    alpha_lesion: float = 0.05,
    alpha_hull: float = 0.4,
    alpha_rim: float = 0.3,
):
    """Display the original diagnostics plot using the shared plotting code."""
    import matplotlib.pyplot as plt

    from analysis.image.plotting import plot_lesion_rim_3d_orig

    fig, _ = plot_lesion_rim_3d_orig(
        prl_case,
        label_source,
        wireframe={"alpha": alpha_wireframe},
        lesion={"alpha": alpha_lesion},
        hull={"alpha": alpha_hull},
        rim={"alpha": alpha_rim},
    )
    fig.tight_layout()
    plt.show()
