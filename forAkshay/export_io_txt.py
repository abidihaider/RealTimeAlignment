#!/usr/bin/env python3
"""
Export 10 input tensors and matching ONNX predictions to two plain-text files.


Run with the project environment, e.g.:

  source /gpfs/mnt/gpfs01/usfcc/pusharma/Haider_LDRD/RealTimeAlignment/setEnv.sh
  python export_io_txt.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _repo_paths() -> tuple[Path, Path, Path]:
    """
    Defaults follow the repo layout under forAkshay/. If REALTIME_ALIGNMENT_ROOT
    is set (e.g. after sourcing setEnv.sh), use that tree:
      source /gpfs/mnt/gpfs01/usfcc/pusharma/Haider_LDRD/RealTimeAlignment/setEnv.sh
    """
    env_root = os.environ.get("REALTIME_ALIGNMENT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        here = root / "forAkshay"
    else:
        here = Path(__file__).resolve().parent
        root = here.parent
    npz = here / "data" / "calibration_samples.npz"
    # Match checks_stepwise_simple_full_compare.ipynb default full-model path.
    onnx_path = root / "onnx_no-residual" / "onnx_files_narrow" / "mlp_fp32.onnx"
    return here, npz, onnx_path


def run_onnx(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    """
    Run inference. Exports often fix the batch dimension to 1; then we run
    one forward per row and stack outputs.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]

    batch_decl = in_meta.shape[0] if len(in_meta.shape) > 0 else None
    fixed_batch_one = batch_decl == 1

    if n == 1:
        y = sess.run(None, {in_name: x})[0]
        return np.asarray(y, dtype=np.float64)

    if fixed_batch_one:
        parts = [sess.run(None, {in_name: x[i : i + 1]})[0] for i in range(n)]
        return np.asarray(np.concatenate(parts, axis=0), dtype=np.float64)

    try:
        y = sess.run(None, {in_name: x})[0]
        return np.asarray(y, dtype=np.float64)
    except Exception:
        parts = [sess.run(None, {in_name: x[i : i + 1]})[0] for i in range(n)]
        return np.asarray(np.concatenate(parts, axis=0), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to calibration_samples.npz (default: forAkshay/data/calibration_samples.npz)",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to model_fp32.onnx (default: onnx_no-residual/onnx_files_narrow/mlp_fp32.onnx next to forAkshay)",
    )
    parser.add_argument(
        "-n",
        "--num-entries",
        type=int,
        default=10,
        help="Number of rows to export (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as this script)",
    )
    args = parser.parse_args()

    here, default_npz, default_onnx = _repo_paths()
    npz_path = args.npz or default_npz
    onnx_path = args.onnx or default_onnx
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    with np.load(npz_path) as z:
        inputs = np.asarray(z["inputs"], dtype=np.float32)

    n = min(args.num_entries, inputs.shape[0])
    inputs = inputs[:n]

    preds = run_onnx(onnx_path, inputs)

    flat_in = inputs.reshape(n, -1)

    inputs_txt = out_dir / "inputs_10.txt"
    preds_txt = out_dir / "predictions_10.txt"

    with inputs_txt.open("w") as f:
        f.write(
            f"# n={n}  shape_per_row=(50,6)  flattened_columns={flat_in.shape[1]}  dtype=float32\n"
        )
        np.savetxt(f, flat_in, fmt="%.8e")

    with preds_txt.open("w") as f:
        f.write(f"# n={n}  columns={preds.shape[1]}  (MLP out_features)\n")
        np.savetxt(f, preds, fmt="%.8e")

    print(f"Wrote {inputs_txt}  rows x cols: {flat_in.shape}")
    print(f"Wrote {preds_txt}  rows x cols: {preds.shape}")


if __name__ == "__main__":
    main()
