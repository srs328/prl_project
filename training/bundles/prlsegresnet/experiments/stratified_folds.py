"""Standalone experiment: per-subject stratified fold assignment.

Goal: assign subjects to folds such that all lesions from one subject
stay in the same fold, while distributing PRL cases as evenly as possible.

This is separate from the bundle pipeline — it's for experimentation
in notebooks before committing to a fold strategy.

Usage:

    from prlsegresnet.experiments.stratified_folds import (
        make_dummy_subjects,
        stratified_subject_fold_assignment,
        print_fold_summary,
    )

    subjects = make_dummy_subjects(50)
    assignments = stratified_subject_fold_assignment(subjects, n_folds=5)
    print_fold_summary(assignments)
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import floor

import numpy as np


def make_dummy_subjects(n_subjects: int = 50, seed: int = 42) -> list[dict]:
    """Generate dummy subject data with realistic distributions.

    PRL count: 0-8, strong right skew (median ~2, most subjects have 0-3).
    Lesion count: matching real data percentiles:
        0% -> 3, 10% -> 7, 25% -> 20, 50% -> 27, 75% -> 52, 90% -> 85, 100% -> 106

    Returns:
        List of dicts with subject_id, n_lesions, n_prls.
    """
    rng = np.random.default_rng(seed)

    # Lesion count: use log-normal fitted to match the given percentiles.
    # The percentiles suggest a right-skewed distribution with median ~27.
    # log(27) ~ 3.3, and we need std to get IQR from ~20 to ~52.
    lesion_counts = rng.lognormal(mean=3.3, sigma=0.6, size=n_subjects)
    lesion_counts = np.clip(lesion_counts, 3, 120).astype(int)

    # PRL count: strong right skew, median ~2, IQR [1,2], max ~8
    # Use a geometric-like distribution shifted by 0.
    # P(k) proportional to decay^k for k=0,1,...,8
    prl_weights = np.array([0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01])
    prl_weights /= prl_weights.sum()
    prl_counts = rng.choice(range(9), size=n_subjects, p=prl_weights)

    # Ensure n_prls <= n_lesions
    prl_counts = np.minimum(prl_counts, lesion_counts)

    subjects = []
    for i in range(n_subjects):
        subjects.append({
            "subject_id": f"sub{1000 + i}",
            "n_lesions": int(lesion_counts[i]),
            "n_prls": int(prl_counts[i]),
        })

    return subjects


def stratified_subject_fold_assignment(
    subjects: list[dict],
    n_folds: int = 5,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Assign subjects to folds with per-subject stratification.

    All lesions from one subject go to the same fold. Subjects are stratified
    by PRL count to distribute PRL cases evenly across folds and testing.

    Strategy:
    1. Bucket subjects by PRL count (0, 1, 2, 3+).
    2. Within each bucket, shuffle and split off test_fraction for testing.
    3. Remaining subjects are assigned to folds round-robin within each bucket.

    Args:
        subjects: List of dicts with subject_id, n_lesions, n_prls.
        n_folds: Number of training folds.
        test_fraction: Fraction of subjects for testing.
        seed: Random seed.

    Returns:
        Dict with:
            training: list of {subject_id, fold, n_lesions, n_prls}
            testing: list of {subject_id, n_lesions, n_prls}
            fold_summary: dict of per-fold statistics
    """
    rng = random.Random(seed)

    # Bucket subjects by PRL count
    buckets: dict[int, list[dict]] = defaultdict(list)
    for subj in subjects:
        bucket_key = min(subj["n_prls"], 3)  # 0, 1, 2, 3+
        buckets[bucket_key].append(subj)

    training = []
    testing = []

    for bucket_key in sorted(buckets.keys()):
        bucket = list(buckets[bucket_key])
        rng.shuffle(bucket)

        n_test = max(1, floor(len(bucket) * test_fraction)) if len(bucket) > 1 else 0
        test_subjects = bucket[:n_test]
        train_subjects = bucket[n_test:]

        for subj in test_subjects:
            testing.append({
                "subject_id": subj["subject_id"],
                "n_lesions": subj["n_lesions"],
                "n_prls": subj["n_prls"],
            })

        for i, subj in enumerate(train_subjects):
            fold = i % n_folds
            training.append({
                "subject_id": subj["subject_id"],
                "fold": fold,
                "n_lesions": subj["n_lesions"],
                "n_prls": subj["n_prls"],
            })

    # Compute summary statistics
    fold_summary = {}
    for fold_num in range(n_folds):
        fold_subjs = [t for t in training if t["fold"] == fold_num]
        fold_summary[fold_num] = {
            "n_subjects": len(fold_subjs),
            "n_lesions": sum(s["n_lesions"] for s in fold_subjs),
            "n_prls": sum(s["n_prls"] for s in fold_subjs),
        }
    test_summary = {
        "n_subjects": len(testing),
        "n_lesions": sum(s["n_lesions"] for s in testing),
        "n_prls": sum(s["n_prls"] for s in testing),
    }
    fold_summary["testing"] = test_summary

    return {
        "training": training,
        "testing": testing,
        "fold_summary": fold_summary,
    }


def print_fold_summary(assignments: dict) -> None:
    """Print a readable summary of fold assignments."""
    summary = assignments["fold_summary"]

    print(f"{'Split':<10} {'Subjects':>10} {'Lesions':>10} {'PRLs':>10}")
    print("-" * 42)

    for key in sorted(k for k in summary if k != "testing"):
        s = summary[key]
        print(f"Fold {key:<5} {s['n_subjects']:>10} {s['n_lesions']:>10} {s['n_prls']:>10}")

    s = summary["testing"]
    print(f"{'Testing':<10} {s['n_subjects']:>10} {s['n_lesions']:>10} {s['n_prls']:>10}")

    # Check balance
    fold_prls = [summary[k]["n_prls"] for k in sorted(k for k in summary if k != "testing")]
    if fold_prls:
        print(f"\nPRL range across folds: {min(fold_prls)} - {max(fold_prls)} "
              f"(mean={np.mean(fold_prls):.1f}, std={np.std(fold_prls):.1f})")

    # Verify no subject appears in multiple folds
    all_ids = [t["subject_id"] for t in assignments["training"]]
    all_ids += [t["subject_id"] for t in assignments["testing"]]
    assert len(all_ids) == len(set(all_ids)), "Subject appears in multiple splits!"
    print("No subject duplication across splits.")
