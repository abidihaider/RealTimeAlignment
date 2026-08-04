"""
Run inference with the trained MLP, then plot local x/y residuals
before and after alignment correction.

Usage examples
--------------
# Use pre-existing dataset pointed to by DATAROOT:
  python plot_residuals.py --config config_narrow.yaml

# Generate data on-the-fly (no DATAROOT needed):
  python plot_residuals.py --config config_narrow.yaml --generate-data 500

# Evolution mode — repeat 10 independent experiments of 200 events each and
# plot how residual mean and sigma change across experiments:
  python plot_residuals.py --config config_narrow.yaml --repeat 10 --generate-data 200

Misalignment types injected during generation
---------------------------------------------
Three types can be applied independently to each detector per event
(up to max_num_misalignments types chosen at random):

  center_shift  — translates the detector center in 3D (isotropic Gaussian,
                  σ = scale × mean_sensor_half-range mm).
  normal_shift  — tilts the detector by rotating its normal to a new direction
                  (Rayleigh-distributed angle, σ ≈ scale rad).
  axes_rotation — rolls the local x/y axes within the detector plane by a
                  Gaussian angle (σ = scale rad); normal and center unchanged.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

from rtal.datasets.dataset import ROMDataset
from rtal.utils import Checkpointer, get_data_root
from rtal.models.mlp_no_residual import MLP
from rtal.geometry.line import reconstruct, get_center_basis
from rtal.data.generate import generate_one

PITCH = 0.1

# Human-readable descriptions of each misalignment type that can be injected.
# center_shift  — pure translation of the detector center.
# normal_shift  — tilts the detector by rotating its normal (+ local axes) to a new direction.
# axes_rotation — in-plane roll of the local x/y axes; normal and center are unchanged.
_MISALIGNMENT_TYPES = {
    'center_shift': (
        'Translation — shifts detector center in 3D by an isotropic Gaussian '
        '(σ = scale × mean_sensor_half-range mm).'
    ),
    'normal_shift': (
        'Tilt — rotates the detector normal into a new direction sampled from a cone '
        'around the original normal (Rayleigh-distributed tilt angle, σ ≈ scale rad).'
    ),
    'axes_rotation': (
        'Roll — rotates local x/y axes within the detector plane by a Gaussian angle '
        '(σ = scale rad); normal and center are left intact.'
    ),
}

# Default generation config — 3 detectors stacked along y, particles from origin
_GEN_CONFIG = {
    'dataset': {
        'random_seed': 42,
        'num_particles': 200,
        'rounded': False,
    },
    'detectors': [
        {'center_start': [0, 10, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
        {'center_start': [0, 20, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
        {'center_start': [0, 30, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
    ],
    'misalignments': {
        'max_num_misalignments': 3,
        'center_shift': {'scale': 0.01},
        'normal_shift': {'scale': 0.0873},
        'axes_rotation': {'scale': 0.0873},
    },
    'particles': {
        'vertex_mean':    [0, 0, 0],
        'vertex_std':     [0.1, 0.1, 0.1],
        'direction_mean': [0, 1, 0],
        'direction_std':  [0.1, 0.1, 0.1],
    },
}


def collect_residuals(model, dataloader, device, num_events):
    """
    Run inference over the dataloader and collect residuals per detector.

    Local residuals (mm): difference in local (x, y) bin coordinates × pitch.
    Global residuals (mm): difference in 3D global coordinates (x, y, z).

    "Before" uses the start detector geometry with the misaligned readout.
    "After"  uses the predicted corrected detector geometry.
    "True"   reference is detector_start geometry with readout_start.

    Returns:
        local_before:  ndarray (N_hits, n_detectors, 2)
        local_after:   ndarray (N_hits, n_detectors, 2)
        global_before: ndarray (N_hits, n_detectors, 3)
        global_after:  ndarray (N_hits, n_detectors, 3)
    """
    local_before_list  = []
    local_after_list   = []
    global_before_list = []
    global_after_list  = []

    n_processed = 0
    model.eval()
    with torch.no_grad():
        for event in tqdm(dataloader, desc='inference'):

            readout = event['readout_curr_cont'].to(device)
            # (B, n_dets, n_particles, 2) -> (B, n_particles, n_dets, 2)
            readout = torch.transpose(readout, 1, 2)

            readout_start = event['readout_start_cont'].to(device)
            readout_start = torch.transpose(readout_start, 1, 2)

            detector_start = event['detector_start'].to(device)   # (B, 3, 9)

            # forward pass
            misalignment_pred = model.inference(readout.flatten(-2, -1), randperm=False)
            misalignment_pred = misalignment_pred.reshape(-1, 3, 9)
            detector_pred = detector_start + misalignment_pred     # (B, 3, 9)

            # --- local residuals (x, y) ---
            # before: raw offset in local frame (mm)
            local_res_before = (readout - readout_start) * PITCH   # (B, P, D, 2)

            # after: reconstruct with predicted geometry, project back to start frame
            pts_corrected = reconstruct(detector_pred, readout)    # (B, P, D, 3)
            center, basis = get_center_basis(detector_start)       # (B, D, 3), (B, D, 2, 3)
            displacement  = pts_corrected - center.unsqueeze(1)    # (B, P, D, 3)
            local_xy      = torch.einsum('bpdr,bdsr->bpds',
                                         displacement, basis)       # (B, P, D, 2) mm
            local_res_after = local_xy - readout_start * PITCH     # (B, P, D, 2)

            # --- global residuals (x, y, z) ---
            pts_true  = reconstruct(detector_start, readout_start) # (B, P, D, 3)
            pts_naive = reconstruct(detector_start, readout)       # (B, P, D, 3)

            global_res_before = pts_naive     - pts_true           # (B, P, D, 3)
            global_res_after  = pts_corrected - pts_true           # (B, P, D, 3)

            # --- local R residual: R(pos) - R(true_pos) ---
            R_true_local   = torch.norm(readout_start * PITCH, dim=-1)  # (B, P, D)
            R_before_local = torch.norm(readout       * PITCH, dim=-1)
            R_after_local  = torch.norm(local_xy,              dim=-1)
            local_res_before = torch.cat(
                [local_res_before, (R_before_local - R_true_local).unsqueeze(-1)], dim=-1)  # (B,P,D,3)
            local_res_after  = torch.cat(
                [local_res_after,  (R_after_local  - R_true_local).unsqueeze(-1)], dim=-1)

            # --- global R residual: transverse R = sqrt(x^2 + z^2) ---
            R_true_global   = torch.norm(pts_true[..., [0, 2]],      dim=-1)  # (B, P, D)
            R_before_global = torch.norm(pts_naive[..., [0, 2]],     dim=-1)
            R_after_global  = torch.norm(pts_corrected[..., [0, 2]], dim=-1)
            global_res_before = torch.cat(
                [global_res_before, (R_before_global - R_true_global).unsqueeze(-1)], dim=-1)  # (B,P,D,4)
            global_res_after  = torch.cat(
                [global_res_after,  (R_after_global  - R_true_global).unsqueeze(-1)], dim=-1)

            # accumulate: merge batch and particle dims
            B, P, D, _ = local_res_before.shape
            local_before_list.append( local_res_before.reshape( B*P, D, 3).cpu().numpy())
            local_after_list.append(  local_res_after.reshape(  B*P, D, 3).cpu().numpy())
            global_before_list.append(global_res_before.reshape(B*P, D, 4).cpu().numpy())
            global_after_list.append( global_res_after.reshape( B*P, D, 4).cpu().numpy())

            n_processed += B
            if num_events is not None and n_processed >= num_events:
                break

    local_before  = np.concatenate(local_before_list,  axis=0)
    local_after   = np.concatenate(local_after_list,   axis=0)
    global_before = np.concatenate(global_before_list, axis=0)
    global_after  = np.concatenate(global_after_list,  axis=0)
    return local_before, local_after, global_before, global_after


def _save_residual_plot(before, after, title, xlabel, colour, out_path):
    """Save a single square histogram (before vs after) to out_path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=13)

    lo = min(before.min(), after.min())
    hi = max(before.max(), after.max())
    bins = np.linspace(lo, hi, 80)

    ax.hist(before, bins=bins, alpha=0.5, color='grey',
            label=f'before  μ={before.mean():.3f}  σ={before.std():.3f}')
    ax.hist(after,  bins=bins, alpha=0.6, color=colour,
            label=f'after   μ={after.mean():.3f}  σ={after.std():.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'saved {out_path}')


def plot_detector(det_idx, before, after, output_dir):
    """Save local x, y, and R residuals as separate square PNGs."""
    specs = [
        ('local_x', 'Local X residual (mm)', 'steelblue',    0),
        ('local_y', 'Local Y residual (mm)', 'darkorange',   1),
        ('local_r', 'Local R residual (mm)', 'mediumpurple', 2),
    ]
    for fname_tag, xlabel, colour, coord in specs:
        _save_residual_plot(
            before[:, coord], after[:, coord],
            title    = f'Detector {det_idx} — {xlabel.split(" ")[0]} {xlabel.split(" ")[1]}',
            xlabel   = xlabel,
            colour   = colour,
            out_path = Path(output_dir) / f'det{det_idx}_{fname_tag}.png',
        )


def plot_detector_global(det_idx, before, after, output_dir):
    """Save global x, y, z, and R residuals as separate square PNGs."""
    specs = [
        ('global_x', 'Global X residual (mm)', 'steelblue',    0),
        ('global_y', 'Global Y residual (mm)', 'seagreen',     1),
        ('global_z', 'Global Z residual (mm)', 'darkorange',   2),
        ('global_r', 'Global R residual (mm)', 'mediumpurple', 3),
    ]
    for fname_tag, xlabel, colour, coord in specs:
        _save_residual_plot(
            before[:, coord], after[:, coord],
            title    = f'Detector {det_idx} — {xlabel.split(" ")[0]} {xlabel.split(" ")[1]}',
            xlabel   = xlabel,
            colour   = colour,
            out_path = Path(output_dir) / f'det{det_idx}_{fname_tag}.png',
        )


def generate_data(data_dir, num_samples, config):
    """
    Generate num_samples npz files into data_dir/test/ using config.
    Skips generation if the directory already has enough files.
    """
    out = Path(data_dir) / 'test'
    existing = list(out.glob('*.npz')) if out.exists() else []
    if len(existing) >= num_samples:
        print(f'Found {len(existing)} existing samples in {out}, skipping generation.')
        return str(Path(data_dir))

    out.mkdir(parents=True, exist_ok=True)
    print(f'Generating {num_samples} samples in {out} ...')
    for i in tqdm(range(num_samples), desc='generating'):
        # use a unique seed per sample so each has a different misalignment
        cfg = {**config, 'dataset': {**config['dataset'], 'random_seed': i}}
        generate_one(cfg, out, f'sample_{i}')

    return str(Path(data_dir))


def generate_data_for_repeat(data_dir, num_samples, config, repeat_idx):
    """Generate num_samples events for one repeat, seeded to avoid collisions across repeats.

    Each repeat gets its own sub-directory so repeated runs with the same M skip
    already-generated batches.
    """
    out = Path(data_dir) / f'repeat_{repeat_idx:04d}' / 'test'
    existing = list(out.glob('*.npz')) if out.exists() else []
    if len(existing) >= num_samples:
        return str(Path(data_dir) / f'repeat_{repeat_idx:04d}')
    out.mkdir(parents=True, exist_ok=True)
    seed_offset = repeat_idx * 10_000
    for i in tqdm(range(num_samples), desc=f'  repeat {repeat_idx} data', leave=False):
        cfg = {**config, 'dataset': {**config['dataset'], 'random_seed': seed_offset + i}}
        generate_one(cfg, out, f'sample_{i}')
    return str(Path(data_dir) / f'repeat_{repeat_idx:04d}')


def compute_batch_stats(local_before, local_after, global_before, global_after):
    """Compute per-detector mean and sigma for each residual coordinate.

    Returns a dict mapping label → (mean_before, std_before, mean_after, std_after),
    where each value is a 1-D array of length n_detectors.
    """
    specs = [
        ('local_x',  local_before,  local_after,  0),
        ('local_y',  local_before,  local_after,  1),
        ('local_r',  local_before,  local_after,  2),
        ('global_x', global_before, global_after, 0),
        ('global_y', global_before, global_after, 1),
        ('global_z', global_before, global_after, 2),
        ('global_r', global_before, global_after, 3),
    ]
    stats = {}
    for label, b_arr, a_arr, idx in specs:
        b = b_arr[:, :, idx]   # (N_hits, n_dets)
        a = a_arr[:, :, idx]
        stats[label] = (b.mean(axis=0), b.std(axis=0), a.mean(axis=0), a.std(axis=0))
    return stats


def _misalignment_summary(gen_config):
    """Short string describing the injection config, embedded in evolution plot titles."""
    mis = gen_config.get('misalignments', {})
    max_n = mis.get('max_num_misalignments', '?')
    parts = []
    if 'center_shift' in mis:
        s = mis['center_shift'].get('scale', '?')
        parts.append(f'center-shift (σ={s}×range)')
    if 'normal_shift' in mis:
        s = mis['normal_shift'].get('scale', '?')
        parts.append(f'normal-tilt (σ≈{s} rad)')
    if 'axes_rotation' in mis:
        s = mis['axes_rotation'].get('scale', '?')
        parts.append(f'axes-rotation (σ={s} rad)')
    header = f'max {max_n} types/detector'
    return f'{header}  |  ' + '  |  '.join(parts) if parts else header


def _print_misalignment_config(gen_config):
    """Print a human-readable summary of what misalignment types are being injected."""
    mis = gen_config.get('misalignments', {})
    print('\nMisalignment injection configuration:')
    print(f"  Max types per detector per event: {mis.get('max_num_misalignments', '?')}")
    for key, desc in _MISALIGNMENT_TYPES.items():
        if key in mis:
            scale = mis[key].get('scale', '?')
            print(f'  [{key}]  scale={scale}')
            print(f'    {desc}')
    print()


def plot_evolution(all_stats, output_dir, n_dets, gen_config):
    """Save evolution plots: mean and sigma vs. experiment index, for each detector.

    Produces one PNG per detector per scope (local, global), each with
    two rows (mean, sigma) and one column per coordinate.
    """
    n_repeats = len(all_stats)
    repeats   = np.arange(1, n_repeats + 1)
    subtitle  = _misalignment_summary(gen_config)

    coord_groups = {
        'local': [
            ('local_x',  'Local X',  'steelblue'),
            ('local_y',  'Local Y',  'darkorange'),
            ('local_r',  'Local R',  'mediumpurple'),
        ],
        'global': [
            ('global_x', 'Global X', 'steelblue'),
            ('global_y', 'Global Y', 'seagreen'),
            ('global_z', 'Global Z', 'darkorange'),
            ('global_r', 'Global R', 'mediumpurple'),
        ],
    }

    for det_idx in range(n_dets):
        for scope, specs in coord_groups.items():
            n_coords = len(specs)
            fig, axes = plt.subplots(2, n_coords,
                                     figsize=(4 * n_coords, 7),
                                     squeeze=False)
            fig.suptitle(
                f'Detector {det_idx} — {scope} residuals over {n_repeats} independent experiments\n'
                f'{subtitle}',
                fontsize=10,
            )

            for c_idx, (key, coord_label, colour) in enumerate(specs):
                means_b = np.array([s[key][0][det_idx] for s in all_stats])
                stds_b  = np.array([s[key][1][det_idx] for s in all_stats])
                means_a = np.array([s[key][2][det_idx] for s in all_stats])
                stds_a  = np.array([s[key][3][det_idx] for s in all_stats])

                # row 0: mean
                ax = axes[0, c_idx]
                ax.plot(repeats, means_b, 'o--', color='grey',  alpha=0.7,
                        markersize=4, label='before')
                ax.plot(repeats, means_a, 'o-',  color=colour, alpha=0.9,
                        markersize=4, label='after')
                ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
                ax.set_title(coord_label, fontsize=9)
                ax.set_ylabel('Mean residual (mm)', fontsize=8)
                ax.set_xlabel('Experiment #', fontsize=8)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

                # row 1: sigma
                ax = axes[1, c_idx]
                ax.plot(repeats, stds_b, 'o--', color='grey',  alpha=0.7,
                        markersize=4, label='before')
                ax.plot(repeats, stds_a, 'o-',  color=colour, alpha=0.9,
                        markersize=4, label='after')
                ax.set_title(coord_label, fontsize=9)
                ax.set_ylabel('Sigma (mm)', fontsize=8)
                ax.set_xlabel('Experiment #', fontsize=8)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

            fig.tight_layout()
            out_path = Path(output_dir) / f'evolution_det{det_idx}_{scope}.png'
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f'saved {out_path}')


def get_args():
    parser = argparse.ArgumentParser(description='Plot local x/y residuals before and after alignment correction')
    parser.add_argument('--config',
                        type    = str,
                        default = 'config_narrow.yaml',
                        help    = 'path to config yaml | default: config_narrow.yaml')
    parser.add_argument('--device',
                        type    = str,
                        default = 'cpu',
                        choices = ('cuda', 'cpu'),
                        help    = 'inference device | default: cpu')
    parser.add_argument('--gpu-id',
                        type    = int,
                        default = 0,
                        help    = 'GPU index (only used when --device cuda) | default: 0')
    parser.add_argument('--output',
                        type    = str,
                        default = 'plots',
                        help    = 'output directory for plots | default: plots/')
    parser.add_argument('--split',
                        type    = str,
                        default = 'test',
                        choices = ('train', 'test'),
                        help    = 'dataset split to load | default: test')
    parser.add_argument('--num-events',
                        type    = int,
                        default = None,
                        help    = 'max number of events to process | default: all')
    parser.add_argument('--generate-data',
                        type    = int,
                        default = None,
                        metavar = 'N',
                        help    = ('generate N samples on-the-fly instead of '
                                   'reading from DATAROOT | e.g. --generate-data 500'))
    parser.add_argument('--data-dir',
                        type    = str,
                        default = './generated_data',
                        help    = ('directory to write/read generated samples '
                                   '(only used with --generate-data) | default: ./generated_data'))
    parser.add_argument('--repeat',
                        type    = int,
                        default = None,
                        metavar = 'M',
                        help    = ('run M independent experiments with fresh random misalignments '
                                   'and plot how mean and sigma evolve across experiments. '
                                   'Use --generate-data N to set events per experiment. '
                                   'e.g. --repeat 10 --generate-data 500'))
    return parser.parse_args()


def main():
    args = get_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_id)

    with open(args.config, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    # build model and load checkpoint
    # resolve checkpoint_path relative to the config file, not cwd
    config_dir = Path(args.config).resolve().parent
    checkpoint_path = config_dir / config['checkpointing']['checkpoint_path']
    model = MLP(**config['model']).to(args.device)
    checkpointer = Checkpointer(model, checkpoint_path=checkpoint_path)
    checkpointer.load(device=args.device)

    num_particles = config['data']['num_particles']
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # --- Evolution mode: M independent experiments ---
    if args.repeat is not None:
        n_events = args.generate_data if args.generate_data is not None else 200
        print(f'Running {args.repeat} independent experiments, {n_events} events each ...')
        _print_misalignment_config(_GEN_CONFIG)

        all_stats = []
        n_dets    = None
        for r in tqdm(range(args.repeat), desc='experiments'):
            data_root  = generate_data_for_repeat(args.data_dir, n_events, _GEN_CONFIG, r)
            dataset    = ROMDataset(data_root, split='test', num_particles=num_particles)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

            lb, la, gb, ga = collect_residuals(model, dataloader, args.device, None)
            all_stats.append(compute_batch_stats(lb, la, gb, ga))
            if n_dets is None:
                n_dets = lb.shape[1]

        plot_evolution(all_stats, args.output, n_dets, _GEN_CONFIG)
        print(f'\nDone. Evolution plots saved to {args.output}/')
        return

    # --- Single-run mode (original behaviour) ---
    if args.generate_data is not None:
        data_root = generate_data(args.data_dir, args.generate_data, _GEN_CONFIG)
    else:
        data_root = get_data_root()

    dataset    = ROMDataset(data_root, split=args.split, num_particles=num_particles)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    local_before, local_after, global_before, global_after = collect_residuals(
        model, dataloader, args.device, args.num_events
    )

    n_detectors = local_before.shape[1]
    for det_idx in range(n_detectors):
        plot_detector(det_idx,
                      local_before[:, det_idx, :],
                      local_after[:, det_idx, :],
                      args.output)
        plot_detector_global(det_idx,
                             global_before[:, det_idx, :],
                             global_after[:, det_idx, :],
                             args.output)

    print(f'\nDone. Plots saved to {args.output}/')


if __name__ == '__main__':
    main()
