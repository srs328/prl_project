# README

Read [this](https://chatgpt.com/share/6b08b727-8c94-83ea-a6c2-470a52a68628)

```python
def split_lst_ai_lesions(
    prob_path: str,
    mask_path: str | None = None,
    prob_thresh: float = 0.5,
    sigma_mm: float = 2.0,
    min_center_voxels: int = 3,
    l1_tol: float = -1e-4,
    return_intermediates: bool = False,
):
    """
    Split confluent lesions using Hessian-based center detection
    and marker-controlled watershed segmentation.

    Gaussian smoothing is applied to accommodate saturated
    lesion probability maps.
    """

    prob_img = nib.load(prob_path)
    prob = prob_img.get_fdata(dtype=np.float32)
    zooms = prob_img.header.get_zooms()[:3]

    if prob.ndim != 3:
        raise ValueError("Expected a 3D probability map.")

    # Define original lesion tissue
    if mask_path is not None:
        mask_img = nib.load(mask_path)

        if mask_img.shape != prob_img.shape:
            raise ValueError("Mask and probability map shapes differ.")

        if not np.allclose(mask_img.affine, prob_img.affine):
            raise ValueError("Mask and probability map affines differ.")

        mask = mask_img.get_fdata() > 0

    else:
        mask = prob >= prob_thresh

    struct_26 = np.ones((3, 3, 3), dtype=bool)

    # Convert physical smoothing scale to voxel units
    sigma_vox = tuple(sigma_mm / z for z in zooms)

    # Smooth probability map for watershed and fallback seeds
    smoothed_prob = ndi.gaussian_filter(
        prob,
        sigma=sigma_vox,
        mode="nearest",
    )

    # Compute Hessian of Gaussian-smoothed probability map
    H_elems = hessian_matrix(
        prob,
        sigma=sigma_vox,
        mode="nearest",
        use_gaussian_derivatives=True,
    )

    # Convert Hessian derivatives to physical units
    # Assumes orthogonal image axes
    sx, sy, sz = zooms

    H_elems = [
        H_elems[0] / sx**2,
        H_elems[1] / (sx * sy),
        H_elems[2] / (sx * sz),
        H_elems[3] / sy**2,
        H_elems[4] / (sy * sz),
        H_elems[5] / sz**2,
    ]

    # Eigenvalues sorted in descending numerical order
    eigvals = hessian_matrix_eigvals(H_elems)
    l1 = eigvals[0]

    del H_elems

    # Detect negative definite Hessian regions
    centers_mask = mask & (l1 < l1_tol)

    # Identify distinct lesion centers
    center_labels, _ = ndi.label(
        centers_mask,
        structure=struct_26,
    )

    # Remove small candidate centers
    if min_center_voxels > 1:
        counts = np.bincount(center_labels.ravel())

        valid_ids = counts >= min_center_voxels
        valid_ids[0] = False

        filtered_centers = valid_ids[center_labels]

        center_labels, _ = ndi.label(
            filtered_centers,
            structure=struct_26,
        )

    # Identify original connected lesions
    orig_labels, num_orig = ndi.label(
        mask,
        structure=struct_26,
    )

    # Determine which original lesions contain a center
    seeded_components = np.unique(
        orig_labels[center_labels > 0]
    )

    unseeded_components = np.setdiff1d(
        np.arange(1, num_orig + 1),
        seeded_components,
    )

    next_id = int(center_labels.max()) + 1

    # Add fallback centers where Hessian detection failed
    for comp_id in unseeded_components:
        comp_mask = orig_labels == comp_id

        peak_idx = np.argmax(
            np.where(comp_mask, smoothed_prob, -np.inf)
        )

        peak_coord = np.unravel_index(peak_idx, prob.shape)

        center_labels[peak_coord] = next_id
        next_id += 1

    # Partition lesion tissue using marker-controlled watershed
    split_labels = watershed(
        -smoothed_prob,
        markers=center_labels,
        mask=mask,
        connectivity=struct_26,
    ).astype(np.int32)

    # Verify preservation of original lesion tissue
    if not np.array_equal(split_labels > 0, mask):
        raise RuntimeError(
            "Watershed did not preserve the original lesion mask."
        )

    # Create integer-valued NIfTI label image
    out_img = nib.Nifti1Image(
        split_labels,
        prob_img.affine,
        prob_img.header.copy(),
    )

    out_img.set_data_dtype(np.int32)

    if return_intermediates:
        intermediates = {
            "prob": prob,
            "smoothed_prob": smoothed_prob,
            "mask": mask,
            "eigvals": eigvals,
            "l1": l1,
            "centers_mask": centers_mask,
            "center_labels": center_labels,
            "orig_labels": orig_labels,
            "unseeded_components": unseeded_components,
        }

        return split_labels, out_img, intermediates

    return split_labels, out_img
```