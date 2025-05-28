# Notes

Some green subjects who supposedly have at least 1 PRL don't have PRl files (check errors.log in test_train0)

## PRL Notes

- Why does 1044 have a PRL label that elongates with the lesion?


---

Email from Chris about the PRL labels (5/27/2025)

> Hey so I’ve uploaded some PRL labels to the server and attaching a sheet with the assessments (same password).
>
> Each subject should now have a folder “lesion.t3m20” which are lesion labels created from the older LST program — not ideal.
>
> Most of the files in this folder can be ignored.  The relevant ones are “lesion_mask.t3m20.nii.gz” which is a lesion mask (lesion=1).  “Lesion_index_t3m20.nii.gz”  is the same file, but with each cluster given a unique label using FSL’s “cluster” function (C3D can also do this with -conn).   There are some “rim”, “core” etc. which can be ignored.  (Also these were made using erosion/dilation instead of SDT so shouldn’t be used anyway.)  The “centerlesion_analysis” folder is basically the “lesion_mask.t3m20” and “lesion_pmap.t3m20” (<-probability map from 0 to 1) fed into a UPenn pipeline which attempts to subsegment the confluent lesions.  It’s a step in the right direction but this remains an important challenge/problem.  If you want to read the paper of how this is done, I’m attaching it.  Another interesting project would be to use the T1 map to inform confluent lesion sub-segmentation.  I think it’s much easier to identify lesions based on their T1 signature rather than the FLAIR, which is too variable within each lesion.  Maybe we can try to recruit someone interested in doing a few confluent lesion segmentations at some point for their own DL project.
>
>Some patients have an “mlesion_index.t30m20.nii.gz”.  This is a file in which PRL have been manually segmented from confluent lesions, or cleaned up in some other way.  When available, this file should always be used.
>
>If they have this mlesion file, there should also be an accompanying “prl_labels_definite” and “prl_labels_probable” files.  These are PRL labels that have been extracted from the [mlesion] file.  You can binarize and combine the “definite” and ”probable” files together to create a PRL training mask along with either the lesion_mask file or an updated LST-AI lesion segmentation for non-PRL lesions.
>
>The other attachment is an Excel file with PRL labels. It’s a bit of a mess so here is how to interpret it:
>
>Col D:
>Green = longitudinal scans >2 years apart available: these were prioritized for manual segmentations (“mlesion_index.t30m20.nii.gz”)
>Yellow = +/- longitudinal scans <2 years: no manual segmentations done.  May or may not need manual segmentation.
>
>Column L, P, T (every 4th column thereafter):  these are the labels corresponding to potential PRL.  The confidence rating appears in the column directly thereafter.  GREEN means that there should be a manual segmentation (“mlesion_index.t30m20.nii.gz”) of the PRLs performed.  Probably we should exclude any lesions labeled as “possible” as these come with a risk of false positives.
>
>To make matters worse, sometimes lesions were manually segmented from a different lesion segmentation program, either SAMSEG or the UPenn subsegmentation program. This is sometimes and inconsistently noted in column “I”. 
>
>To start the training, I’d suggest we hand-select patients with clean PRL examples as well as the ones with manual segmentation.  For example it looks like ms1007 has a single “definite” PRL.  But it is yellow, meaning that it hasn’t been manually segmented and needs to be reviewed to make sure the PRL is cleanly defined.  (I just looked, and it needs some minor work.)
>
>Our best bet is probably to look at the greens, but even then every subject will likely need QC.  Probably we should sit down and do some PRL teaching before doing too much further.

Excel file is `PRL_labels.xlxs`
