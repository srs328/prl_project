# Lesion Analysis

Regarding `notebooks/lesion_diagnostics2.ipynb`

Q1: generate_binary_structure(3,1) vs (3,3) — which for shells?
generate_binary_structure(rank, connectivity):

connectivity=1 → 6-connectivity: only face-adjacent neighbors (3D cross, 7 True voxels in 3x3x3). Dilation/erosion only acts along the 3 cardinal axes.
connectivity=2 → 18-connectivity: face + edge neighbors (19 True voxels).
connectivity=3 → 26-connectivity: full 3x3x3 cube, all 26 neighbors (27 True voxels).
For shell preservation in stage 2: connectivity=1 is the obviously better choice.

Binary closing = dilation then erosion. With connectivity=1, dilation only spreads along cardinal axes — diagonal corners of the shell's interior don't get filled in, so thin shells stay thin. With connectivity=3, diagonal filling is aggressive: even 2 iterations on a 1-2 voxel thick shell can collapse it into a blob.

So your code choice is right — use (3,3) for stage 1 (aggressive merge), (3,1) for stage 2 (shell-preserving).

Q2: Smart way to grab non-trivial rim units
The fixed > 10 threshold is fine as a floor but doesn't scale. A more principled approach — fraction of the largest component:


max_size = np.sum(labeled == sorted_comps[0])
components = [c for c in sorted_comps if np.sum(labeled == c) > max_size * 0.05]
5-10% of the largest component works well because:

A real secondary rim from a confluent lesion (your case) will be a substantial fraction of the dominant rim
Noise fragments are typically <1% of the largest component
Adapts automatically when the dominant rim is large vs. small
You could also floor it: max(10, max_size * 0.05) to handle edge cases with very small rims.

Q3: Two-stage assignment — how to assign rim_close2 voxels to stage-1 components
The cleanest way is a nearest-neighbor / Voronoi assignment using distance_transform_edt:


# Stage 1: aggressive closure to identify components
rim_close1 = ndimage.binary_closing(rim, iterations=3, structure=ndimage.generate_binary_structure(3, 3))
labeled_coarse, n = ndimage.label(rim_close1, structure=ndimage.generate_binary_structure(3, 3))
# filter to non-trivial components (as above)...

# Stage 2: conservative closure to preserve shell shape
rim_close2 = ndimage.binary_closing(rim, iterations=2, structure=ndimage.generate_binary_structure(3, 1))

# Assign rim_close2 voxels to nearest stage-1 component
_, indices = ndimage.distance_transform_edt(labeled_coarse == 0, return_indices=True)
nearest_label = labeled_coarse[tuple(indices)]  # Voronoi expansion of components
rim_close2_labeled = nearest_label * rim_close2  # restrict to conservatively-closed mask
distance_transform_edt(..., return_indices=True) gives the coordinates of the nearest non-zero voxel for every background voxel — exactly what you need to do watershed-style assignment without implementing it yourself. The result rim_close2_labeled is a labeled version of your conservative closure where each voxel belongs to the nearest stage-1 component.

Summary for your two-stage design
Stage	Structure	Iterations	Purpose
1	(3,3) 26-conn	3–4	Aggressively merge fragments → find component units
2	(3,1) 6-conn	1–2	Close small gaps while preserving shell topology
Then use the distance_transform_edt Voronoi trick to assign stage-2 voxels to stage-1 component IDs.

>Instead of the distance transform what If I just get the labeled component image from stage 1 (labeled), zero out the components that did not pass the filter (labeled[~np.isin(labeled, components)] = 0), and then  multiply the stage 2 closure with the labeled image and mask to get the component I want np.argwhere(rim_close2*labeled == desired_component).
>
>Will that work, or are there holes in my logic

Yes, it mostly works. The logic is sound for the typical case. Let me walk through where it holds and where there's a subtle gap.

Why it works: Since stage 1 uses a larger structuring element ((3,3)) than stage 2 ((3,1)), rim_close1 is generally a superset of rim_close2. This means labeled (from stage 1) should cover all voxels in rim_close2, so the product rim_close2 * labeled assigns every conservative-closure voxel to a component ID.

The one real hole: Voxels in rim_close2 that fall in the gap between two non-trivial stage-1 components. Both stage-1 components were aggressive enough to be distinct in labeled, but a rim_close2 voxel sitting in that gap would have labeled == 0 there (it belongs to neither component's territory). That voxel gets silently dropped.

With the distance transform approach, those gap voxels would be assigned to whichever component is closer.
With your approach, they're just missing.
Whether this matters: Probably not for your use case. Those gap voxels are the most ambiguous ones (sitting between two rim units), and dropping them is a defensible choice. If you're extracting features per rim unit, the missing voxels are a tiny fraction at the boundary.

One thing to watch: labeled[~np.isin(labeled, components)] = 0 mutates labeled in place. If you call this twice (e.g., iterating over cases), or reference labeled again later, it's already modified. Consider working on a copy: labeled_filtered = labeled.copy(); labeled_filtered[...] = 0.

Bottom line: your approach is correct and simpler than the distance transform for this use case. The only missing voxels are the ambiguous boundary ones, which you likely don't care about.
