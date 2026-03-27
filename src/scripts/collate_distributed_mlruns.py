import shutil
import time
import random
from pathlib import Path
import yaml

def collate_mlruns(run_dir: str | Path):
    """
    Collate distributed per-fold mlruns into a single top-level mlruns directory.
    Safe to run multiple times (skips if top-level mlruns already exists).
    """
    run_dir = Path(run_dir)
    dst_mlruns = run_dir / "mlruns"
    
    if dst_mlruns.exists():
        print(f"Top-level mlruns already exists at {dst_mlruns}, skipping.")
        return

    # Collect all per-fold mlruns
    fold_mlruns = sorted(run_dir.glob("segresnet_*/model/mlruns"))
    if not fold_mlruns:
        raise FileNotFoundError(f"No per-fold mlruns found under {run_dir}")

    # New experiment: pick a fresh ID and use the run directory name
    new_exp_id = str(random.randint(10**17, 10**18 - 1))
    exp_name = run_dir.name  # e.g. "run1" or "run2"
    new_exp_dir = dst_mlruns / new_exp_id
    new_exp_dir.mkdir(parents=True)

    # Create top-level default experiment (id=0)
    default_exp_dir = dst_mlruns / "0"
    default_exp_dir.mkdir()
    (default_exp_dir / "meta.yaml").write_text(
        "artifact_location: mlflow-artifacts:/0\n"
        "creation_time: 0\nexperiment_id: '0'\nlast_update_time: 0\n"
        "lifecycle_stage: active\nname: Default\n"
    )
    (dst_mlruns / ".trash").mkdir()

    # Write new experiment meta.yaml
    now_ms = int(time.time() * 1000)
    exp_meta = {
        "artifact_location": str(new_exp_dir),
        "creation_time": now_ms,
        "experiment_id": new_exp_id,
        "last_update_time": now_ms,
        "lifecycle_stage": "active",
        "name": exp_name,
    }
    with open(new_exp_dir / "meta.yaml", "w") as f:
        yaml.dump(exp_meta, f, default_flow_style=False)

    # Copy each fold's runs into the new experiment
    for fold_mlrun in fold_mlruns:
        fold_idx = fold_mlrun.parts[-3]  # "segresnet_0"
        # Each fold's mlruns has exactly one experiment dir (numeric ID)
        src_exp_dirs = [d for d in fold_mlrun.iterdir()
                        if d.is_dir() and d.name not in ("0", ".trash")]
        for src_exp_dir in src_exp_dirs:
            for run_dir_src in src_exp_dir.iterdir():
                if not run_dir_src.is_dir():
                    continue
                run_id = run_dir_src.name
                dst_run_dir = new_exp_dir / run_id
                shutil.copytree(run_dir_src, dst_run_dir)

                # Rewrite run meta.yaml with new experiment_id and artifact_uri
                run_meta_path = dst_run_dir / "meta.yaml"
                with open(run_meta_path) as f:
                    run_meta = yaml.safe_load(f)
                run_meta["experiment_id"] = new_exp_id
                run_meta["artifact_uri"] = str(dst_run_dir / "artifacts")
                # Tag with fold source if not already in run_name
                if fold_idx not in run_meta.get("run_name", ""):
                    run_meta["run_name"] = f"{fold_idx} - {run_meta.get('run_name', run_id)}"
                with open(run_meta_path, "w") as f:
                    yaml.dump(run_meta, f, default_flow_style=False)

    print(f"Created {dst_mlruns}")
    print(f"  Experiment '{exp_name}' (id={new_exp_id})")
    print(f"  Collated {len(fold_mlruns)} fold(s)")


