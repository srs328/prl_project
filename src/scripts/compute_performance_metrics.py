#!/usr/bin/env python3
"""
Compute comprehensive performance metrics (TP, FP, TN, FN, sensitivity, specificity, etc.)
from inference results against ground truth labels.

Treats label value 1 (lesion) as negative and label value 2 (rim) as positive.

Usage:
    python compute_performance_metrics.py /path/to/run2
    python compute_performance_metrics.py /path/to/run2 --test-only --print-results
    python compute_performance_metrics.py /path/to/run2 --output-csv performance_metrics.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
import pandas as pd
from my_python_utils import save_json

from helpers.paths import load_config

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed")


def get_test_inference(test_data, dataroot, inf_root, suffix) -> list[dict]:
    for scan in test_data:
        scan.update({k: Path(v) if isinstance(v, str) and ".nii.gz" in v else v for k, v in scan.items()})
        scan['inference'] = inf_root / Path(scan['label']).relative_to(dataroot).with_name(f"flair.phase{suffix}_ensemble.nii.gz")
        if "prl" in Path(scan['label']).name:
            scan['case_type'] = "PRL"
        else:
            scan['case_type'] = "Lesion"
    return test_data


def get_validation_inference(train_data, dataroot, val_root, suffix):
    validation_data = defaultdict(list)
    for scan in train_data:
        scan.update({k: Path(v) if isinstance(v, str) and ".nii.gz" in v else v for k, v in scan.items()})
        scan['inference'] = val_root / f"fold{scan['fold']}" / scan['label'].relative_to(dataroot).with_name(f"flair.phase{suffix}.nii.gz")
        if "prl" in scan['label'].name:
            scan['case_type'] = "PRL"
        else:
            scan['case_type'] = "Lesion"
        validation_data[f"fold{scan['fold']}"].append(scan)
    return validation_data


def get_confusion_matrix(lab_path: Path, inf_path: Path) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix from label and inference images.

    Treats:
        - Label 1 (lesion) = negative
        - Label 2 (rim) = positive

    Returns:
        TP, FP, TN, FN
    """
    if not HAS_NIBABEL:
        return 0, 0, 0, 0

    try:
        lab = nib.load(lab_path).get_fdata()
        inf = nib.load(inf_path).get_fdata()

        TP = np.sum((lab == 2) & (inf == 2))
        FP = np.sum((lab == 1) & (inf == 2))
        TN = np.sum((lab == 1) & (inf == 1))
        FN = np.sum((lab == 2) & (inf == 1))

        return int(TP), int(FP), int(TN), int(FN)
    except Exception as e:
        print(f"Warning: Failed to compute metrics for {lab_path}: {e}")
        return 0, 0, 0, 0


def compute_derived_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Compute all derived metrics from confusion matrix.

    Returns dict with:
        - Sensitivity (recall, TPR): TP / (TP + FN)
        - Specificity, TNR: TN / (TN + FP)
        - FPR: FP / (FP + TN) = 1 - Specificity
        - FNR: FN / (TP + FN) = 1 - Sensitivity
        - Precision: TP / (TP + FP)
        - Negative Predictive Value: TN / (TN + FN)
        - Accuracy: (TP + TN) / (TP + FP + TN + FN)
        - F1 Score: 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    """
    metrics = {}

    # Sensitivity = Recall = TPR (true positive rate)
    if (tp + fn) > 0:
        metrics['sensitivity'] = tp / (tp + fn)
    else:
        metrics['sensitivity'] = np.nan

    # Specificity = TNR (true negative rate)
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        metrics['specificity'] = np.nan

    # FPR = 1 - Specificity
    if (tn + fp) > 0:
        metrics['fpr'] = fp / (tn + fp)
    else:
        metrics['fpr'] = np.nan

    # FNR = 1 - Sensitivity
    if (tp + fn) > 0:
        metrics['fnr'] = fn / (tp + fn)
    else:
        metrics['fnr'] = np.nan

    # Precision = PPV (positive predictive value)
    if (tp + fp) > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = np.nan

    # NPV (negative predictive value)
    if (tn + fn) > 0:
        metrics['npv'] = tn / (tn + fn)
    else:
        metrics['npv'] = np.nan

    # Accuracy
    total = tp + fp + tn + fn
    if total > 0:
        metrics['accuracy'] = (tp + tn) / total
    else:
        metrics['accuracy'] = np.nan

    # F1 Score
    precision = metrics.get('precision', np.nan)
    recall = metrics.get('sensitivity', np.nan)
    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        metrics['f1'] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics['f1'] = np.nan

    return metrics


def analyze_dataset(data, split: str = "testing") -> Dict:
    """
    Analyze a dataset split (testing, training, validation).

    Returns dict with per-case and aggregated metrics.
    """
    

    results = {
        'split': split,
        'cases': [],
        'aggregated': {},
    }

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    valid_cases = 0

    for item in data:
        lab_path = Path(item['label'])
        inf_path = Path(item['inference'])

        # Compute confusion matrix
        tp, fp, tn, fn = get_confusion_matrix(lab_path, inf_path)


        # Compute metrics for this case
        metrics = {
            'subid': item.get('subid'),
            'lesion_index': item.get('lesion_index'),
            'split': split,
            'case_type': item.get('case_type'),
            # 'fold': item.get('fold', "NA"),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }
        metrics.update(compute_derived_metrics(tp, fp, tn, fn))
        
        if (tp + fp + tn + fn) == 0:
            print(lab_path, inf_path)
            metrics.update({"Notes": "Something's wrong"})
        else:
            metrics.update({"Notes": ""})
            # Accumulate for aggregate metrics
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            valid_cases += 1
            
        results['cases'].append(metrics)

    if valid_cases == 0:
        print(f"Warning: No valid cases found for {split}")
        return results
    else:
        print(f"{valid_cases} valid cases of {len(data)}")

    # Compute aggregated metrics from summed confusion matrix
    agg_metrics = compute_derived_metrics(total_tp, total_fp, total_tn, total_fn)
    agg_metrics.update({
        'tp': total_tp,
        'fp': total_fp,
        'tn': total_tn,
        'fn': total_fn,
        'case_count': valid_cases,
    })
    results['aggregated'] = agg_metrics

    return results


def print_results(results: Dict):
    """Print formatted results."""
    split = results['split']
    agg = results['aggregated']
    cases = results['cases']

    print("\n" + "="*100)
    print(f"PERFORMANCE METRICS: {split.upper()}")
    print("="*100)

    if not agg:
        print(f"No valid cases found for {split}")
        return

    print(f"\nAGGREGATED METRICS (summed across {agg['case_count']} cases)")
    print("-"*100)
    print(f"  Confusion Matrix:")
    print(f"    TP: {agg['tp']:>8}  |  FP: {agg['fp']:>8}")
    print(f"    FN: {agg['fn']:>8}  |  TN: {agg['tn']:>8}")
    print(f"\n  Metrics:")
    print(f"    Sensitivity (Recall):  {agg.get('sensitivity', np.nan):.4f}  (TP / (TP+FN))")
    print(f"    Specificity:           {agg.get('specificity', np.nan):.4f}  (TN / (TN+FP))")
    print(f"    FPR:                   {agg.get('fpr', np.nan):.4f}  (1 - Specificity)")
    print(f"    FNR:                   {agg.get('fnr', np.nan):.4f}  (1 - Sensitivity)")
    print(f"    Precision (PPV):       {agg.get('precision', np.nan):.4f}  (TP / (TP+FP))")
    print(f"    NPV:                   {agg.get('npv', np.nan):.4f}  (TN / (TN+FN))")
    print(f"    Accuracy:              {agg.get('accuracy', np.nan):.4f}")
    print(f"    F1 Score:              {agg.get('f1', np.nan):.4f}")

    # Per-case statistics
    if cases:
        print(f"\n\nPER-CASE STATISTICS")
        print("-"*100)

        metrics_to_show = ['sensitivity', 'specificity', 'fpr', 'fnr', 'precision', 'accuracy', 'f1']

        for metric_name in metrics_to_show:
            values = [case.get(metric_name, np.nan) for case in cases]
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                print(f"\n  {metric_name.upper()}:")
                print(f"    Mean:   {np.mean(valid_values):.4f}")
                print(f"    Std:    {np.std(valid_values):.4f}")
                print(f"    Min:    {np.min(valid_values):.4f}")
                print(f"    Max:    {np.max(valid_values):.4f}")

        # Show cases sorted by accuracy (best to worst)
        print(f"\n\nPER-CASE BREAKDOWN (sorted by accuracy)")
        print("-"*120)
        print(f"{'Case':<40} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'Acc':<10} {'F1':<10} {'Type':<10}")
        print("-"*120)

        sorted_cases = sorted(cases, key=lambda c: c.get('accuracy', -1), reverse=True)
        for case in sorted_cases[:30]:  # Show top 30
            case_name = f"Sub{case['subid']}-L{case['lesion_index']}"
            if 'fold' in case and case['fold'] is not None:
                case_name += f"-F{case['fold']}"

            case_type = case['case_type']

            # Format metrics, showing N/A for NaN
            def fmt(val):
                return f"{val:.4f}" if not np.isnan(val) else "N/A"

            sens = fmt(case.get('sensitivity', np.nan))
            spec = fmt(case.get('specificity', np.nan))
            prec = fmt(case.get('precision', np.nan))
            acc = fmt(case.get('accuracy', np.nan))
            f1 = fmt(case.get('f1', np.nan))

            print(f"{case_name:<40} {sens:>10} {spec:>10} {prec:>10} {acc:>10} {f1:>10} {case_type:<10}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute performance metrics from inference results"
    )
    parser.add_argument("train_dir", type=Path, help="Path to training home")
    parser.add_argument("--test-only", action="store_true", help="Only analyze test set")
    parser.add_argument("--output-csv", type=Path, help="Save results to CSV")
    parser.add_argument("--save-path", type=Path, help="Save image and label paths for all cases")
    parser.add_argument("--print-results", action="store_true", help="Whether to print results to console")

    args = parser.parse_args()

    if not args.train_dir.exists():
        print(f"Error: Run directory not found: {args.train_dir}")
        return 1
    
    train_dir = Path(args.train_dir)
    label_config = load_config(train_dir / "label_config.json")

    expand_xy, expand_z = label_config['expand_xy'], label_config['expand_z']
    expand_suffix = f"_xy{expand_xy}_z{expand_z}"
    with open(train_dir / f"datalist_xy{expand_xy}_z{expand_z}.json", 'r') as f:
        datalist = json.load(f)

    dataroot = label_config['dataroot']
    inf_root = train_dir / "ensemble_output"
    val_root = train_dir / "fold_predictions"

    # Analyze splits
    splits = ['testing']
    if not args.test_only:
        splits.extend(['training', 'validation'])

    all_results = {}
    

    inf_data = get_test_inference(datalist['testing'], dataroot, inf_root, expand_suffix)
    val_data = get_validation_inference(datalist['training'], dataroot, val_root, expand_suffix)
    
    inf_results = analyze_dataset(inf_data, split="testing")
    if inf_results.get('aggregated'):
        all_results['testing'] = inf_results
        if args.print_results:
            print_results(inf_results)
    
    for fold, fold_data in val_data.items():
        split=f"validation {fold}"
        results = analyze_dataset(fold_data, split=split)
        if results.get('aggregated'):
            all_results[split] = results
            if args.print_results:
                print_results(results)

    # Save to CSV if requested
    if args.output_csv:
        all_cases = []
        for split, results in all_results.items():
            for case in results['cases']:
                case['split'] = split
                all_cases.append(case)

        if all_cases:
            df = pd.DataFrame(all_cases)
            df.to_csv(args.output_csv, index=False)
            print(f"\n\nResults saved to: {args.output_csv}")
    
    if args.save_path:
        save_path = args.save_path
        if not save_path.is_absolute():
            save_path = train_dir / save_path
    else:
        save_path = train_dir / f"datalist_{expand_suffix}_final.json"
    save_data = {"testing": inf_data, "validation": val_data}
    save_json(save_data, save_path)

    return 0


if __name__ == "__main__":
    exit(main())
