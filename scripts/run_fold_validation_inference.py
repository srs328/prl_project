#!/usr/bin/env python3
"""
Run inference on validation data for each fold and save organized results.

This module can be:
1. Used standalone to generate validation predictions for a completed training run
2. Integrated into the training pipeline to generate predictions after each fold trains

Saves predictions organized as:
    run_dir/fold_predictions/fold{N}/
        └── sub{subid}-{sesid}/
            └── {lesion_index}/
                └── flair.phase_{suffix}_pred.nii.gz

Usage:
    # Standalone: run on completed training run
    python run_fold_validation_inference.py \
        /path/to/run2 \
        /path/to/datalist_xy20_z2.json \
        --data-root /path/to/dataroot

    # For specific fold only
    python run_fold_validation_inference.py \
        /path/to/run2 \
        /path/to/datalist_xy20_z2.json \
        --data-root /path/to/dataroot \
        --fold 4

    # With custom output directory
    python run_fold_validation_inference.py \
        /path/to/run2 \
        /path/to/datalist_xy20_z2.json \
        --data-root /path/to/dataroot \
        --output-dir /custom/output/path
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from collections import defaultdict

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed")

HAS_MONAI = False
try:
    import monai
    from monai.networks.nets import SegResNetDS
    HAS_MONAI = True
except Exception as e:
    print(f"Warning: Could not import MONAI properly: {e}")


class FoldValidationInference:
    """Run inference on validation data for a specific fold."""

    def __init__(self, run_dir: Path, fold_num: int, datalist_path: Path,
                 dataroot: Path, output_dir: Optional[Path] = None,
                 device: str = "cuda"):
        """
        Initialize validation inference for a fold.

        Args:
            run_dir: Path to training run directory (e.g., run2)
            fold_num: Fold number (0-4 for 5-fold CV)
            datalist_path: Path to datalist JSON file
            dataroot: Path to data root directory
            output_dir: Output directory for predictions (default: run_dir/fold_predictions)
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.run_dir = Path(run_dir)
        self.fold_num = fold_num
        self.dataroot = Path(dataroot)
        self.device = device

        # Load datalist
        with open(datalist_path, 'r') as f:
            self.datalist = json.load(f)

        # Extract expansion parameters from datalist filename
        # e.g., datalist_xy20_z2.json -> xy20_z2
        self.suffix = "_" + datalist_path.stem.split("_", 1)[1]

        # Setup output directory
        if output_dir is None:
            self.output_dir = self.run_dir / "fold_predictions" / f"fold{fold_num}"
        else:
            self.output_dir = Path(output_dir)

        # Get model path for this fold
        self.model_path = self.run_dir / f"segresnet_{fold_num}" / "model" / "model.pt"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Get validation cases (cases labeled with this fold number)
        self.validation_cases = self._get_validation_cases()

        print(f"Fold {fold_num} validation inference:")
        print(f"  Model: {self.model_path}")
        print(f"  Cases: {len(self.validation_cases)}")
        print(f"  Output: {self.output_dir}")

    def _get_validation_cases(self) -> List[Dict]:
        """Get all training cases labeled with this fold number."""
        validation_cases = []

        for item in self.datalist.get('training', []):
            if item.get('fold') == self.fold_num:
                validation_cases.append(item)

        return validation_cases

    def load_model(self) -> torch.nn.Module:
        """Load the trained model for this fold."""
        if not HAS_MONAI:
            raise ImportError("MONAI is required to load models")

        print(f"Loading model from {self.model_path}...")

        # Load checkpoint (MONAI Auto3DSeg format)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Extract components
        state_dict = checkpoint.get('state_dict', checkpoint)
        config = checkpoint.get('config', {})

        # Get network config
        network_config = config.get('network', {})
        in_channels = network_config.get('in_channels', 2)
        out_channels = network_config.get('out_channels', 3)
        init_filters = network_config.get('init_filters', 32)
        blocks_down = network_config.get('blocks_down', [1, 2, 4])
        dsdepth = network_config.get('dsdepth', 4)
        norm = network_config.get('norm', 'INSTANCE')

        print(f"  Model: SegResNetDS")
        print(f"  In channels: {in_channels}, Out channels: {out_channels}")

        # Create wrapper model (matching the segmenter architecture)
        # The state dict has encoder/up_layers structure, so we need a wrapper
        class SegmentationModel(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network

            def forward(self, x):
                return self.network(x)

        # Try to import SegResNetDS
        try:
            from monai.networks.nets import SegResNetDS
        except ImportError:
            from monai.networks.nets import SegResNet as SegResNetDS
            print("  Warning: Using SegResNet instead of SegResNetDS")

        # Create the core network
        network = SegResNetDS(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=blocks_down,
            norm=norm,
            dsdepth=dsdepth,
        )

        # Wrap it
        model = SegmentationModel(network)

        # Load state dict - might have wrapper keys
        try:
            model.load_state_dict(state_dict, strict=False)
            print("  Loaded with strict=False (some keys may be missing)")
        except RuntimeError as e:
            print(f"  Failed to load state dict: {e}")
            print("  Trying to extract network-only state dict...")

            # Extract network-only keys (remove wrapper prefix if any)
            network_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes
                if k.startswith('network.'):
                    new_k = k[8:]  # Remove "network." prefix
                else:
                    new_k = k

                network_state_dict[new_k] = v

            # Try loading network directly
            try:
                network.load_state_dict(network_state_dict, strict=False)
                print("  Loaded network state dict with strict=False")
            except Exception as e2:
                print(f"  Warning: Could not fully load state dict: {e2}")
                # Continue anyway - might work for inference with uninitialized weights

        model.to(self.device)
        model.eval()

        return model

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load NIfTI image and return data and affine."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required to load images")

        img = nib.load(image_path)
        data = img.get_fdata()

        return data, img

    def preprocess_image(self, image_data: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Preprocess image for inference.

        Applies:
        - Normalization (mean/std)
        - Ensure 3D and add batch/channel dims
        - Pad to divisible by 4 (required for SegResNetDS)
        - Convert to tensor

        Returns:
            Tuple of (tensor, original_size) for unpadding later
        """
        # Normalize (simple mean/std normalization)
        data = image_data.astype(np.float32)
        data = (data - data.mean()) / (data.std() + 1e-8)

        # Store original size
        original_size = data.shape

        # Add channel dimension
        if data.ndim == 3:
            data = data[np.newaxis, :]  # Add channel dim -> (C, D, H, W)

        # Pad to be divisible by 4 (required for SegResNetDS with dsdepth=4)
        # Current shape: (C, D, H, W)
        divisor = 4
        padded_data = np.zeros((data.shape[0],) + tuple(
            ((s + divisor - 1) // divisor) * divisor for s in data.shape[1:]
        ), dtype=np.float32)
        padded_data[:, :data.shape[1], :data.shape[2], :data.shape[3]] = data

        # Add batch dimension
        padded_data = padded_data[np.newaxis, ...]  # (B, C, D, H, W)

        tensor = torch.tensor(padded_data, device=self.device, dtype=torch.float32)

        return tensor, original_size

    def run_inference(self, image_tensor: torch.Tensor, model: torch.nn.Module,
                     original_size: Tuple[int, int, int]) -> np.ndarray:
        """Run inference on preprocessed image and unpad result.

        Args:
            image_tensor: Padded input tensor (B, C, D, H, W)
            model: Model to use for inference
            original_size: Original size before padding (D, H, W)

        Returns:
            Prediction array with original size
        """
        with torch.no_grad():
            # Forward pass
            output = model(image_tensor)

            # Convert to probability (softmax) and get argmax
            # output shape: (batch, channels, d, h, w) or list of multi-scale outputs for DS
            if isinstance(output, list):
                output = output[0]  # Use the full-resolution output for deep supervision

            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0].astype(np.uint8)

            # Unpad to original size
            d, h, w = original_size
            pred = pred[:d, :h, :w]

            return pred

    def process_case(self, case: Dict, model: torch.nn.Module) -> bool:
        """
        Run inference for a single case and save result.

        Returns:
            True if successful, False otherwise
        """
        try:
            image_path = Path(case['image'])

            # Skip if image doesn't exist
            if not image_path.exists():
                print(f"  ⚠ Image not found: {image_path}")
                return False

            # Load image
            image_data, img_nifti = self.load_image(image_path)

            # Preprocess
            image_tensor = self.preprocess_image(image_data)

            # Run inference
            prediction = self.run_inference(image_tensor, model)

            # Save prediction with same affine as input
            output_path = self.output_dir / image_path.relative_to(self.dataroot)
            output_path = output_path.with_stem(output_path.stem + "_pred")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create NIfTI image with same affine as input
            pred_img = nib.Nifti1Image(prediction, affine=img_nifti.affine)
            nib.save(pred_img, output_path)

            return True

        except Exception as e:
            print(f"  ✗ Error processing {case.get('subid')}-L{case.get('lesion_index')}: {e}")
            return False

    def run(self) -> Dict:
        """Run inference on all validation cases for this fold.

        Returns:
            Dict with statistics about the run
        """
        print(f"\n{'='*80}")
        print(f"Running validation inference for Fold {self.fold_num}")
        print(f"{'='*80}\n")

        # Load model
        model = self.load_model()

        # Process each case
        successful = 0
        failed = 0

        print(f"Processing {len(self.validation_cases)} validation cases...")

        for i, case in enumerate(self.validation_cases, 1):
            subid = case.get('subid')
            lesion_idx = case.get('lesion_index')

            print(f"  [{i}/{len(self.validation_cases)}] Sub{subid}-L{lesion_idx}...", end=" ")

            if self.process_case(case, model):
                print("✓")
                successful += 1
            else:
                print("✗")
                failed += 1

        # Print summary
        print(f"\n{'─'*80}")
        print(f"Fold {self.fold_num} complete: {successful}/{len(self.validation_cases)} successful")
        if failed > 0:
            print(f"  Failed: {failed}")
        print(f"{'─'*80}\n")

        return {
            'fold': self.fold_num,
            'total': len(self.validation_cases),
            'successful': successful,
            'failed': failed,
            'output_dir': str(self.output_dir),
        }


def run_all_folds(run_dir: Path, datalist_path: Path, dataroot: Path,
                  output_dir: Optional[Path] = None, folds: Optional[List[int]] = None,
                  device: str = "cuda") -> Dict:
    """
    Run validation inference for all folds.

    Args:
        run_dir: Path to training run directory
        datalist_path: Path to datalist JSON
        dataroot: Path to data root
        output_dir: Output directory (default: run_dir/fold_predictions)
        folds: List of fold numbers to process (default: all folds)
        device: Device for inference
    """
    # Determine number of folds from datalist if not specified
    if folds is None:
        with open(datalist_path, 'r') as f:
            datalist = json.load(f)

        fold_nums = set(item.get('fold') for item in datalist.get('training', []))
        folds = sorted(fold_nums)

    results = {}

    for fold_num in folds:
        try:
            inferencer = FoldValidationInference(
                run_dir, fold_num, datalist_path, dataroot, output_dir, device
            )
            result = inferencer.run()
            results[fold_num] = result
        except Exception as e:
            print(f"\n✗ Error processing Fold {fold_num}: {e}")
            results[fold_num] = {
                'fold': fold_num,
                'error': str(e),
            }

    # Print overall summary
    print(f"\n{'='*80}")
    print("VALIDATION INFERENCE SUMMARY")
    print(f"{'='*80}\n")

    total_successful = 0
    total_cases = 0

    for fold_num in sorted(results.keys()):
        result = results[fold_num]
        if 'error' in result:
            print(f"Fold {fold_num}: ERROR - {result['error']}")
        else:
            total_successful += result['successful']
            total_cases += result['total']
            success_rate = 100 * result['successful'] / result['total']
            print(f"Fold {fold_num}: {result['successful']}/{result['total']} ({success_rate:.1f}%)")

    if total_cases > 0:
        overall_rate = 100 * total_successful / total_cases
        print(f"\nOverall: {total_successful}/{total_cases} ({overall_rate:.1f}%)")

    print(f"\nPredictions saved to: {run_dir}/fold_predictions/")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run validation inference on fold data and save predictions"
    )
    parser.add_argument("run_dir", type=Path, help="Path to training run directory")
    parser.add_argument("datalist", type=Path, help="Path to datalist JSON file")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to data root")
    parser.add_argument("--fold", type=int, default=None,
                       help="Specific fold to process (default: all folds)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Custom output directory for predictions")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for inference (cuda or cpu)")

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        return 1

    if not args.datalist.exists():
        print(f"Error: Datalist not found: {args.datalist}")
        return 1

    if not args.data_root.exists():
        print(f"Error: Data root not found: {args.data_root}")
        return 1

    # Run inference
    if args.fold is not None:
        # Single fold
        folds = [args.fold]
    else:
        # All folds
        folds = None

    results = run_all_folds(
        args.run_dir,
        args.datalist,
        args.data_root,
        args.output_dir,
        folds,
        args.device,
    )

    # Return error if any fold failed
    if any('error' in r for r in results.values()):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
