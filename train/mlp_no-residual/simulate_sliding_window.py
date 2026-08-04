"""
Simulate real-time alignment with a sliding window of tracks.

The detector misalignment evolves over time following a chosen profile.  At each
time step one new particle track is recorded.  The NN uses the W most-recent
tracks (the sliding window) to predict the alignment correction, and we measure
(a) the residual of the newest track before and after that correction, and
(b) how well the predicted alignment *parameters* follow the injected ones.

Misalignment state per detector — 6 parameters
-----------------------------------------------
  dx, dy, dz  (mm)   — translation of the detector centre
  nu          (-)    — projection of the detector normal on the reference u axis
  nv          (-)    — projection of the detector normal on the reference v axis
  rho         (rad)  — in-plane rotation (roll) of the local x/y axes

`nu`/`nv`/`rho` are exactly the parameterisation implemented by
`rtal.geometry.misalign.Misalign`, whose default reference frame
(u_ref = [-1, 0, 0], v_ref = [0, 0, 1], n_ref = [0, 1, 0]) coincides with the
starting orientation of every detector below.  Using that class for both the
forward map (parameters -> geometry) and the inverse map (geometry ->
parameters) means the recovered parameters are exact, not an approximation.

For small angles `nu` ~ tilt about the local y axis and `nv` ~ tilt about the
local x axis, both in radians.  The constraint nu^2 + nv^2 < 1 must hold.

These parameters map onto the three misalignment types used to generate the
training data (see `rtal/data/detector.py`):

    dx, dy, dz  <->  center_shift    (training sigma = 0.05 mm per axis)
    nu, nv      <->  normal_shift    (training sigma ~ 0.0873 rad = 5 deg)
    rho         <->  axes_rotation   (training sigma = 0.0873 rad = 5 deg)

Injection profiles
------------------
  walk    random walk, per-step std --drift-center / --drift-angle (default)
  ramp    linear sweep from 0 to the amplitude over the whole run
  sine    oscillation of the given amplitude and --period
  static  constant offset equal to the amplitude

Combine a profile with --params to isolate a single misalignment type, e.g.
`--profile ramp --params dx` drives dx alone while the other five parameters
stay exactly zero.  `--scan` runs one such isolated study per parameter and
writes a side-by-side comparison.

Sliding window
--------------
At time t the NN input is the stack of readout_curr vectors for tracks
[t-W+1, ..., t], treated as a single event with W "particles".  The NN
predicts one misalignment correction for the whole window.  The residual
of track t (the newest) under this correction is the real-time resolution.

Usage
-----
  # combined random walk (original behaviour)
  python simulate_sliding_window.py \\
      --config train/mlp_no-residual/config_narrow.yaml \\
      --steps 1000 --window 50 \\
      --output train/mlp_no-residual/plots/sliding_window

  # single misalignment type, linear ramp
  python simulate_sliding_window.py --config config_narrow.yaml \\
      --steps 500 --window 50 --profile ramp --params dx --output plots/dx_ramp

  # sweep all six parameters one at a time and compare
  python simulate_sliding_window.py --config config_narrow.yaml \\
      --steps 300 --window 50 --profile ramp --scan --output plots/scan

Architectures
-------------
Both the original 27-output model (rtal.models.mlp_no_residual.MLP) and the
physical-parameter model (rtal.models.mlp_physical.PhysicalMLP) are supported;
--arch defaults to inferring which one from the config. Point --config at the
relevant training config and everything downstream is identical, so the two can
be compared plot for plot:

  python simulate_sliding_window.py \\
      --config ../mlp_physical/config.yaml \\
      --steps 300 --window 50 --profile ramp --scan --output plots/scan_physical

For the physical model the predicted parameters come straight from the network,
so the tracking plots involve no inversion at all.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import torch

from rtal.utils import Checkpointer
from rtal.models.mlp_no_residual import MLP
from rtal.models.mlp_physical import PhysicalMLP, params_to_detector
from rtal.geometry.line import reconstruct, get_center_basis
from rtal.geometry.misalign import Misalign
from rtal.geometry import weak_modes
from rtal.data.detector import Detector
from rtal.data.particle import Particles

PITCH = 0.1

_DETECTORS = [
    {'center_start': [0, 10, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
    {'center_start': [0, 20, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
    {'center_start': [0, 30, 0], 'local_x_start': [-1, 0, 0], 'local_y_start': [0, 0, 1]},
]

_PARTICLE_CONFIG = {
    'vertex_mean':    np.array([0.0, 0.0, 0.0]),
    'vertex_std':     np.array([0.1, 0.1, 0.1]),
    'direction_mean': np.array([0.0, 1.0, 0.0]),
    'direction_std':  np.array([0.1, 0.1, 0.1]),
}

# The six misalignment parameters, in the order used everywhere in this module.
_PARAM_NAMES  = ['dx', 'dy', 'dz', 'nu', 'nv', 'rho']
_PARAM_UNITS  = ['mm', 'mm', 'mm', '-',  '-',  'rad']
_PARAM_COLORS = ['steelblue', 'seagreen', 'darkorange',
                 'firebrick', 'mediumpurple', 'saddlebrown']
_TRANSLATION_PARAMS = (0, 1, 2)   # dx, dy, dz  — driven by the "centre" scales
_ANGLE_PARAMS       = (3, 4, 5)   # nu, nv, rho — driven by the "angle" scales

# The nine raw detector parameters the network actually predicts.
_RAW9_LABELS = ['d cx', 'd cy', 'd cz',
                'd lx_x', 'd lx_y', 'd lx_z',
                'd ly_x', 'd ly_y', 'd ly_z']

_PROFILES = ('walk', 'ramp', 'sine', 'static')

# Forward/inverse orientation map.  float64 so the round-trip check is tight.
_MISALIGNER = Misalign(dtype=torch.float64)


def load_dataset_config(path):
    """
    Adopt the detector geometry and beam from a data-generation config.

    The tracks fed to the model here must be drawn from the same distribution
    it was trained on — the vertex and direction spreads set how much of each
    sensor is illuminated and at what incidence angle, which is precisely what
    determines how observable a tilt is. Evaluating a spread-vertex model with
    the default collimated beam would be measuring it out of distribution.
    """
    global _DETECTORS, _PARTICLE_CONFIG   # pylint: disable=global-statement

    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    _DETECTORS = config['detectors']

    particles = config['particles']
    _PARTICLE_CONFIG = {
        'vertex_mean':    np.asarray(particles['vertex_mean'],    dtype=np.float64),
        'vertex_std':     np.asarray(particles['vertex_std'],     dtype=np.float64),
        'direction_mean': np.asarray(particles['direction_mean'], dtype=np.float64),
        'direction_std':  np.asarray(particles['direction_std'],  dtype=np.float64),
    }

    _check_reference_frame()

    print(f'dataset config: {path}')
    print(f'  detectors    : {len(_DETECTORS)}')
    print(f'  vertex_std   : {_PARTICLE_CONFIG["vertex_std"]}')
    print(f'  direction_std: {_PARTICLE_CONFIG["direction_std"]}')


def _check_reference_frame():
    """
    `Misalign` parameterises the orientation relative to a single fixed reference
    frame.  That is only a valid description of our detectors if every detector
    starts in exactly that frame.
    """
    u_ref = _MISALIGNER.u_ref.numpy()
    v_ref = _MISALIGNER.v_ref.numpy()
    for i, dc in enumerate(_DETECTORS):
        lx = np.array(dc['local_x_start'], dtype=np.float64); lx /= np.linalg.norm(lx)
        ly = np.array(dc['local_y_start'], dtype=np.float64); ly /= np.linalg.norm(ly)
        assert np.allclose(lx, u_ref, atol=1e-8) and np.allclose(ly, v_ref, atol=1e-8), (
            f'detector {i} does not start in the Misalign reference frame '
            f'(u_ref={u_ref}, v_ref={v_ref}) — the nu/nv/rho parameterisation '
            f'would not be comparable across detectors'
        )


_check_reference_frame()


# ---------------------------------------------------------------------------
# Geometry <-> parameters
# ---------------------------------------------------------------------------

def params_to_geometry(trajectory):
    """
    Map misalignment parameters to detector geometry.

    trajectory : (..., n_dets, 6) — [dx, dy, dz, nu, nv, rho]

    Returns centers, local_x, local_y — each (..., n_dets, 3), float64.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    starts = np.stack([np.asarray(d['center_start'], dtype=np.float64)
                       for d in _DETECTORS])                       # (n_dets, 3)

    centers = starts + traj[..., :3]

    state = torch.as_tensor(traj, dtype=torch.float64)
    local_x, local_y = _MISALIGNER.misalign(state[..., 3], state[..., 4], state[..., 5])

    return centers, local_x.numpy(), local_y.numpy()


def geometry_to_params(det9, det9_start):
    """
    Inverse of `params_to_geometry`, acting on the 9-parameter detector vectors
    [center(3), local_x(3), local_y(3)] used throughout the codebase.

    det9, det9_start : (..., 9) array-like (numpy or torch)

    Returns (..., 6) numpy float64 — [dx, dy, dz, nu, nv, rho].
    """
    det   = torch.as_tensor(np.asarray(_to_numpy(det9)),       dtype=torch.float64)
    start = torch.as_tensor(np.asarray(_to_numpy(det9_start)), dtype=torch.float64)

    nu, nv, rho = _MISALIGNER.get_misalignment(det[..., 3:6], det[..., 6:9])
    dcenter = det[..., :3] - start[..., :3]

    return torch.cat([dcenter,
                      nu.unsqueeze(-1),
                      nv.unsqueeze(-1),
                      rho.unsqueeze(-1)], dim=-1).numpy()


def frame_health(det9):
    """
    How far the predicted local axes are from an orthonormal pair.

    The network is free to emit any 9 numbers, so its local_x / local_y need not
    be unit length or perpendicular.  Large deviations make the recovered `rho`
    (and to a lesser extent nu/nv) less meaningful, so we track them.

    Returns (..., 3): [|cos(lx, ly)|, ||lx|| - 1, ||ly|| - 1].
    """
    det = np.asarray(_to_numpy(det9), dtype=np.float64)
    lx, ly = det[..., 3:6], det[..., 6:9]

    nx = np.linalg.norm(lx, axis=-1)
    ny = np.linalg.norm(ly, axis=-1)
    cos = np.abs(np.sum(lx * ly, axis=-1) / np.clip(nx * ny, 1e-12, None))

    return np.stack([cos, nx - 1.0, ny - 1.0], axis=-1)


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

def detect_arch(model_config):
    """
    Infer the architecture from the model block of a config.

    PhysicalMLP takes num_detectors and emits 6 parameters per detector; the
    original MLP takes out_features and emits the 9 raw numbers.
    """
    if 'num_detectors' in model_config:
        return 'physical'
    if 'out_features' in model_config:
        return 'raw9'
    raise ValueError(
        'cannot tell which architecture this config describes — expected either '
        '"num_detectors" (physical) or "out_features" (raw9) under model:'
    )


class Predictor:
    """
    Uniform interface over the two architectures.

    Both are asked the same question — given a window of readouts and the
    nominal geometry, what is the corrected geometry? — so every plot
    downstream is architecture-agnostic.

    The physical model returns its six parameters directly, so for it the
    tracking plots need no inversion at all; for the raw-9 model they are
    recovered with geometry_to_params.
    """

    def __init__(self, model, arch):
        self.model = model
        self.arch  = arch

    def __call__(self, model_input, detector_start):
        """
        model_input    : (1, W, n_dets*2)
        detector_start : (1, n_dets, 9)

        Returns detector_pred (1, n_dets, 9) and, when the architecture
        provides them, params (n_dets, 6) as numpy — otherwise None.
        """
        if self.arch == 'physical':
            params = self.model.inference(model_input, randperm=False)   # (1, n_dets, 6)
            detector_pred = params_to_detector(params, detector_start,
                                               self.model.misalign_layer)
            return detector_pred, _to_numpy(params).squeeze(0).astype(np.float64)

        n_dets = detector_start.shape[1]
        misalignment = self.model.inference(model_input, randperm=False)  # (1, n_dets*9)
        misalignment = misalignment.reshape(1, n_dets, 9)
        return detector_start + misalignment, None


# ---------------------------------------------------------------------------
# Misalignment trajectory
# ---------------------------------------------------------------------------

def detector_positions():
    """Plane coordinate along the beam, for the weak-mode projection."""
    centers = np.stack([np.asarray(d['center_start'], dtype=np.float64)
                        for d in _DETECTORS])
    return centers[:, int(np.argmax(centers.std(axis=0)))]


def observable_weights(p_idx):
    """
    Per-detector pattern for an injection the detector can actually see.

    Applying the same value to every plane — the default for every profile — is
    a *blind* mode: straight tracks cannot see a global translation, shear,
    rotation or roll, so projecting a uniform pattern leaves exactly nothing.
    The pattern has to be built observable in the first place.

    For the translations, whose constant and linear-in-L modes are both blind,
    the only survivor with three planes is the second difference (+1, -2, +1).
    For the rotations, only the common mode is blind, so the simplest survivor
    is the linear-in-L pattern (+1, 0, -1).

    Normalised to unit peak, so `--amp-*` still means the largest per-detector
    excursion.
    """
    positions = detector_positions()
    family = weak_modes.blind_family(_PARAM_NAMES[p_idx])
    basis  = weak_modes.blind_basis(positions, family)          # (n_dets, rank)

    # For the rotations the observable subspace is 2-dimensional, so "any
    # orthogonal vector" is under-determined — and an arbitrary SVD basis
    # vector can put a zero on a detector, leaving it untested. Start from an
    # alternating template instead: projected, it becomes the second difference
    # (+1, -2, +1), which is observable for *both* families and gives every
    # detector a non-zero excursion.
    template  = (-1.0) ** np.arange(len(positions))
    projector = np.eye(len(positions)) - basis @ basis.T
    weights   = projector @ template

    peak = np.abs(weights).max()
    if peak < 1e-12:
        raise ValueError(f'no observable pattern exists for {_PARAM_NAMES[p_idx]} '
                         f'with {len(positions)} detectors')

    return weights / peak


def build_trajectory(n_steps, n_dets, profile, active_params, active_dets,
                     amps, drifts, period, rng, phase_step=0.0, shape='uniform'):
    """
    Build a misalignment trajectory of shape (n_steps, n_dets, 6).

    Every parameter not listed in `active_params`, and every detector not listed
    in `active_dets`, stays exactly zero — that is what makes single-type
    injection studies clean.

    profile       : one of _PROFILES
    active_params : iterable of parameter indices (0..5)
    active_dets   : iterable of detector indices
    amps          : (6,) amplitude per parameter — endpoint for `ramp`,
                    peak for `sine`, constant for `static`
    drifts        : (6,) per-step std, used by `walk`
    period        : steps per full `sine` period
    phase_step    : `sine` phase offset added per detector index (rad)
    """
    if profile not in _PROFILES:
        raise ValueError(f'unknown profile {profile!r}, expected one of {_PROFILES}')

    traj = np.zeros((n_steps, n_dets, 6))
    t    = np.arange(n_steps)

    for det_idx in active_dets:
        for p_idx in active_params:
            if profile == 'walk':
                steps    = rng.normal(0.0, drifts[p_idx], n_steps)
                steps[0] = 0.0
                values   = np.cumsum(steps)
            elif profile == 'ramp':
                values = amps[p_idx] * t / max(n_steps - 1, 1)
            elif profile == 'sine':
                values = amps[p_idx] * np.sin(2 * np.pi * t / period
                                              + phase_step * det_idx)
            else:  # static
                values = np.full(n_steps, amps[p_idx])

            if shape == 'observable':
                values = values * observable_weights(p_idx)[det_idx]

            traj[:, det_idx, p_idx] = values

    return _clamp_normal(traj)


def _clamp_normal(traj, limit=0.99):
    """
    Enforce nu^2 + nv^2 < 1, which `Misalign.misalign` needs to form the normal
    (it takes sqrt(1 - nu^2 - nv^2)).  Rescales the offending steps rather than
    letting a NaN propagate silently.
    """
    radius = np.hypot(traj[..., 3], traj[..., 4])
    worst  = radius.max()
    if worst <= limit:
        return traj

    scale = limit / worst
    print(f'WARNING: tilt magnitude sqrt(nu^2 + nv^2) reaches {worst:.4f}, which '
          f'exceeds the physical limit of 1. Scaling nu and nv by {scale:.4f}.')
    traj = traj.copy()
    traj[..., 3] *= scale
    traj[..., 4] *= scale
    return traj


def describe_injection(profile, active_params, active_dets, amps, drifts, period):
    """One-line human-readable summary, used in plot titles and the scorecard."""
    names = ', '.join(_PARAM_NAMES[p] for p in active_params) or 'none'
    dets  = ', '.join(str(d) for d in active_dets)

    if profile == 'walk':
        detail = 'per-step std ' + ', '.join(
            f'{_PARAM_NAMES[p]}={drifts[p]:g}' for p in active_params)
    else:
        detail = 'amplitude ' + ', '.join(
            f'{_PARAM_NAMES[p]}={amps[p]:g}' for p in active_params)
        if profile == 'sine':
            detail += f'; period {period}'

    return f'profile={profile}; params={names}; detectors={dets}; {detail}'


# ---------------------------------------------------------------------------
# Single-track generation
# ---------------------------------------------------------------------------

def generate_track(centers, local_x, local_y, rng, max_tries=100):
    """
    Generate one valid particle track for the given per-detector geometry.

    centers, local_x, local_y : (n_dets, 3) — the misaligned ("curr") geometry.

    Returns (readout_curr, readout_start, detector_curr, detector_start)
      readout_*:  float32 (n_dets, 2) in pixel units
      detector_*: float64 (n_dets, 9)
    Raises RuntimeError if no valid particle found within max_tries.
    """
    pc = _PARTICLE_CONFIG
    for _ in range(max_tries):
        vertex    = rng.normal(pc['vertex_mean'],    pc['vertex_std'])
        direction = rng.normal(pc['direction_mean'], pc['direction_std'])
        particles = Particles(vertex=vertex[None], direction=direction[None])

        detectors = []
        for i, dc in enumerate(_DETECTORS):
            cs = np.array(dc['center_start'], dtype=np.float64)
            lx = np.array(dc['local_x_start'], dtype=np.float64); lx /= np.linalg.norm(lx)
            ly = np.array(dc['local_y_start'],  dtype=np.float64); ly /= np.linalg.norm(ly)
            detectors.append(Detector(
                center_start=cs,        local_x_start=lx,       local_y_start=ly,
                center_curr=centers[i], local_x_curr=local_x[i], local_y_curr=local_y[i],
            ))

        rd_s_list, rd_c_list = [], []
        valid = True
        for det in detectors:
            rd_s, m_s = det.get_readout(particles, 'start')
            rd_c, m_c = det.get_readout(particles, 'curr')
            if not (m_s[0] and m_c[0]):
                valid = False
                break
            rd_s_list.append(rd_s[0].astype(np.float32))
            rd_c_list.append(rd_c[0].astype(np.float32))

        if not valid:
            continue

        params = [d.get_parameters() for d in detectors]
        det_s  = np.stack([p['start'] for p in params]).astype(np.float64)  # (n_dets, 9)
        det_c  = np.stack([p['curr']  for p in params]).astype(np.float64)

        return (np.stack(rd_c_list),   # readout_curr  (n_dets, 2)
                np.stack(rd_s_list),   # readout_start (n_dets, 2)
                det_c,                 # detector_curr  (n_dets, 9)
                det_s)                 # detector_start (n_dets, 9)

    raise RuntimeError('Could not generate a valid track — check detector geometry / particle config.')


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(predictor, device, trajectory, window_size, seed, meta):
    """
    Generate tracks along `trajectory` and run sliding-window inference.

    predictor  : Predictor wrapping either architecture
    trajectory : (n_steps, n_dets, 6)

    Returns a dict of per-inference-step arrays (indexed from window_size-1 on).
    """
    num_steps, n_dets, _ = trajectory.shape
    rng = np.random.default_rng(seed)

    centers, local_x, local_y = params_to_geometry(trajectory)

    print(f'Generating {num_steps} tracks ...')
    tracks       = []
    roundtrip_err = 0.0
    for t in tqdm(range(num_steps), desc='tracks'):
        track = generate_track(centers[t], local_x[t], local_y[t], rng)
        tracks.append(track)
        # The forward map (params -> geometry) and the inverse map used for the
        # tracking plots must agree, otherwise "predicted vs true" is measuring
        # our own bookkeeping error rather than the model.
        recovered = geometry_to_params(track[2], track[3])
        roundtrip_err = max(roundtrip_err, np.abs(recovered - trajectory[t]).max())

    print(f'parameter round-trip error (forward then inverse): {roundtrip_err:.3e}')
    assert roundtrip_err < 1e-6, (
        f'parameter round-trip error {roundtrip_err:.3e} is too large — the '
        f'forward and inverse misalignment maps disagree'
    )

    print(f'Sliding-window inference  (W = {window_size}) ...')
    results = {k: [] for k in (
        't',
        'true_center_shift', 'true_orient_shift',
        'pred_center_shift', 'pred_orient_shift',
        'true_params', 'pred_params',
        'true_raw', 'pred_raw',
        'frame_health',
        'local_before', 'local_after',
        'global_before', 'global_after',
    )}

    predictor.model.eval()
    with torch.no_grad():
        for t in tqdm(range(window_size - 1, num_steps), desc='inference'):
            w0 = t - window_size + 1

            # ---- model input: (1, W, n_dets*2) ----
            win_rc = np.stack([tracks[i][0] for i in range(w0, t + 1)])  # (W, n_dets, 2)
            inp    = torch.tensor(win_rc, dtype=torch.float32, device=device)
            inp    = inp.flatten(-2, -1).unsqueeze(0)                     # (1, W, 6)

            # ---- newest track residuals ----
            rc_new, rs_new, det_c_new, det_s_new = tracks[t]
            det_s_t  = torch.tensor(det_s_new, dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_dets, 9)

            det_pred, params_direct = predictor(inp, det_s_t)              # (1, n_dets, 9)

            rc_t = torch.tensor(rc_new, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,n_dets,2)
            rs_t = torch.tensor(rs_new, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            # local before
            lb = (rc_t - rs_t) * PITCH                                              # (1,1,n_dets,2)

            # local after: reconstruct with predicted geometry, project to start frame
            pts_c = reconstruct(det_pred, rc_t)                                     # (1,1,n_dets,3)
            ctr, basis = get_center_basis(det_s_t)                                  # (1,n_dets,3), (1,n_dets,2,3)
            disp  = pts_c - ctr.unsqueeze(1)                                        # (1,1,n_dets,3)
            lxy   = torch.einsum('bpdr,bdsr->bpds', disp, basis)                    # (1,1,n_dets,2)
            la    = lxy - rs_t * PITCH                                              # (1,1,n_dets,2)

            # local R
            R_true  = torch.norm(rs_t * PITCH, dim=-1)
            R_curr  = torch.norm(rc_t * PITCH, dim=-1)
            R_after = torch.norm(lxy,           dim=-1)
            lb = torch.cat([lb, (R_curr  - R_true).unsqueeze(-1)], dim=-1)          # (1,1,n_dets,3)
            la = torch.cat([la, (R_after - R_true).unsqueeze(-1)], dim=-1)

            # global
            pts_true  = reconstruct(det_s_t, rs_t)                                 # (1,1,n_dets,3)
            pts_naive = reconstruct(det_s_t, rc_t)
            gb = pts_naive - pts_true
            ga = pts_c     - pts_true

            # global R
            Rt_g = torch.norm(pts_true[...,  [0, 2]], dim=-1)
            Rn_g = torch.norm(pts_naive[..., [0, 2]], dim=-1)
            Ra_g = torch.norm(pts_c[...,     [0, 2]], dim=-1)
            gb = torch.cat([gb, (Rn_g - Rt_g).unsqueeze(-1)], dim=-1)              # (1,1,n_dets,4)
            ga = torch.cat([ga, (Ra_g - Rt_g).unsqueeze(-1)], dim=-1)

            # squeeze to (n_dets, coords)
            lb = lb.squeeze(0).squeeze(0).cpu().numpy()
            la = la.squeeze(0).squeeze(0).cpu().numpy()
            gb = gb.squeeze(0).squeeze(0).cpu().numpy()
            ga = ga.squeeze(0).squeeze(0).cpu().numpy()

            # ---- misalignment: raw 9 params, physical 6 params, magnitudes ----
            # For both architectures the raw view is the delta between the
            # predicted and the nominal geometry — for the raw-9 model that is
            # exactly what it emitted.
            det_pred_np = _to_numpy(det_pred).squeeze(0).astype(np.float64)
            mis_true    = det_c_new - det_s_new                                    # (n_dets, 9)
            mis_pred_np = det_pred_np - det_s_new

            # The physical model hands back its parameters directly, so no
            # inversion is needed; the raw-9 model needs one.
            pred_params = (params_direct if params_direct is not None
                           else geometry_to_params(det_pred_np, det_s_new))

            results['t'].append(t)
            results['true_raw'].append(mis_true)
            results['pred_raw'].append(mis_pred_np)
            results['true_params'].append(geometry_to_params(det_c_new, det_s_new))
            results['pred_params'].append(pred_params)
            results['frame_health'].append(frame_health(det_pred_np))
            results['true_center_shift'].append(np.linalg.norm(mis_true[:, :3], axis=-1))
            results['true_orient_shift'].append(np.linalg.norm(mis_true[:, 3:], axis=-1))
            results['pred_center_shift'].append(np.linalg.norm(mis_pred_np[:, :3], axis=-1))
            results['pred_orient_shift'].append(np.linalg.norm(mis_pred_np[:, 3:], axis=-1))
            results['local_before'].append(lb)
            results['local_after'].append(la)
            results['global_before'].append(gb)
            results['global_after'].append(ga)

    for k in results:
        results[k] = np.array(results[k])

    results['trajectory'] = trajectory   # (n_steps, n_dets, 6) — full walk, for plotting
    results['meta']       = meta
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _fit_stats(true, pred):
    """
    Agreement between a true and a predicted parameter series.

    slope — gain of the model's response (1 = tracks perfectly, 0 = ignores it)
    r2    — squared Pearson correlation (how much of the variation is followed)
    bias  — mean(pred - true)
    rmse  — root mean square of (pred - true)
    """
    true = np.asarray(true, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()

    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]

    out = {'slope': np.nan, 'r2': np.nan, 'bias': np.nan, 'rmse': np.nan}
    if true.size < 2:
        return out

    out['bias'] = float(np.mean(pred - true))
    out['rmse'] = float(np.sqrt(np.mean((pred - true) ** 2)))

    # A constant truth (an inactive parameter under a static profile) has no
    # slope to fit — leave slope/r2 as NaN rather than inventing a number.
    if np.std(true) > 1e-12 and np.std(pred) > 1e-12:
        out['slope'] = float(np.polyfit(true, pred, 1)[0])
        out['r2']    = float(np.corrcoef(true, pred)[0, 1] ** 2)

    return out


def compute_scorecard(results):
    """
    Per-parameter agreement plus the residual improvement it buys.

    Returns {'params': {name: {'det{i}': stats, 'all': stats}},
             'residual': {'det{i}': {...}},
             'meta': ...}
    """
    n_dets = results['true_params'].shape[1]

    scorecard = {'meta': results['meta'], 'params': {}, 'residual': {}}

    for p_idx, name in enumerate(_PARAM_NAMES):
        entry = {}
        for det_idx in range(n_dets):
            entry[f'det{det_idx}'] = _fit_stats(results['true_params'][:, det_idx, p_idx],
                                                results['pred_params'][:, det_idx, p_idx])
        entry['all'] = _fit_stats(results['true_params'][..., p_idx],
                                  results['pred_params'][..., p_idx])
        scorecard['params'][name] = entry

    for det_idx in range(n_dets):
        r_before = results['local_before'][:, det_idx, 2]
        r_after  = results['local_after'][:, det_idx, 2]
        sig_b    = float(np.std(r_before))
        sig_a    = float(np.std(r_after))

        # The local R coordinate is invariant under an in-plane roll, so for rho
        # its "before" spread is identically zero and the R improvement ratio is
        # meaningless.  The RMS of the full 2D local residual vector responds to
        # every misalignment type, so it is the metric to compare types with.
        xy_b = float(np.sqrt(np.mean(results['local_before'][:, det_idx, :2] ** 2) * 2))
        xy_a = float(np.sqrt(np.mean(results['local_after'][:, det_idx, :2] ** 2) * 2))

        scorecard['residual'][f'det{det_idx}'] = {
            'local_r_sigma_before': sig_b,
            'local_r_sigma_after':  sig_a,
            'local_r_improvement':  float(sig_b / sig_a) if sig_a > 0 else np.nan,
            'local_xy_rms_before':  xy_b,
            'local_xy_rms_after':   xy_a,
            'improvement':          float(xy_b / xy_a) if xy_a > 0 else np.nan,
        }

    return scorecard


def write_scorecard(scorecard, output_dir):
    """Write scorecard.txt (diffable) and scorecard.png (a rendered table)."""
    out_dir = Path(output_dir)

    lines = ['Injection: ' + scorecard['meta']['description'], '']
    header = f'{"param":>6} {"det":>5} {"slope":>10} {"r2":>8} {"bias":>12} {"rmse":>12}'
    lines += [header, '-' * len(header)]
    for name, entry in scorecard['params'].items():
        for det_key, stats in entry.items():
            lines.append(f'{name:>6} {det_key:>5} {stats["slope"]:>10.4f} '
                         f'{stats["r2"]:>8.4f} {stats["bias"]:>12.4e} {stats["rmse"]:>12.4e}')
        lines.append('')

    lines += ['Local residuals (mm) — improvement = before / after, >1 is better',
              '-' * 65]
    for det_key, stats in scorecard['residual'].items():
        lines.append(f'{det_key:>5}  2D xy RMS: before={stats["local_xy_rms_before"]:.5f}  '
                     f'after={stats["local_xy_rms_after"]:.5f}  '
                     f'improvement={stats["improvement"]:.3f}x')
        lines.append(f'{"":>5}  R sigma:   before={stats["local_r_sigma_before"]:.5f}  '
                     f'after={stats["local_r_sigma_after"]:.5f}  '
                     f'improvement={stats["local_r_improvement"]:.3f}x'
                     '   (R is blind to in-plane roll)')

    txt = out_dir / 'scorecard.txt'
    txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'saved {txt}')

    js = out_dir / 'scorecard.json'
    js.write_text(json.dumps(scorecard, indent=2, default=float), encoding='utf-8')
    print(f'saved {js}')

    # rendered table — the "all detectors" row per parameter
    cell_text = []
    for name, entry in scorecard['params'].items():
        s = entry['all']
        cell_text.append([name,
                          f'{s["slope"]:.4f}', f'{s["r2"]:.4f}',
                          f'{s["bias"]:.3e}',  f'{s["rmse"]:.3e}'])

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(cell_text) + 2.2))
    ax.axis('off')
    ax.set_title('Per-parameter tracking scorecard (all detectors pooled)\n'
                 + scorecard['meta']['description'], fontsize=10)
    table = ax.table(cellText=cell_text,
                     colLabels=['param', 'slope', 'r2', 'bias', 'rmse'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    fig.tight_layout()
    out = out_dir / 'scorecard.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(x, w):
    """Centred rolling mean of width w; NaN-padded at edges."""
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    out    = np.convolve(x, kernel, mode='valid')
    pad    = w - 1
    left   = pad // 2
    right  = pad - left
    return np.concatenate([np.full(left, np.nan), out, np.full(right, np.nan)])


def _param_label(p_idx):
    unit = _PARAM_UNITS[p_idx]
    return f'{_PARAM_NAMES[p_idx]} ({unit})' if unit != '-' else _PARAM_NAMES[p_idx]


def plot_trajectory(results, output_dir):
    """Plot the 6 raw misalignment parameters per detector over time."""
    trajectory = results['trajectory']
    n_steps, n_dets, _ = trajectory.shape
    t = np.arange(n_steps)

    fig, axes = plt.subplots(6, n_dets, figsize=(5 * n_dets, 12), squeeze=False)
    fig.suptitle('Injected misalignment trajectory\n' + results['meta']['description'],
                 fontsize=11)

    for det_idx in range(n_dets):
        for p_idx in range(6):
            ax = axes[p_idx, det_idx]
            ax.plot(t, trajectory[:, det_idx, p_idx],
                    color=_PARAM_COLORS[p_idx], linewidth=0.8)
            ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
            ax.set_ylabel(_param_label(p_idx), fontsize=8)
            ax.set_xlabel('Time step', fontsize=8)
            ax.tick_params(labelsize=7)
            if p_idx == 0:
                ax.set_title(f'Detector {det_idx}', fontsize=10)

    fig.tight_layout()
    out = Path(output_dir) / 'trajectory.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def plot_param_tracking(results, output_dir):
    """
    The headline plot: injected vs predicted value of each of the 6 physical
    misalignment parameters, over time, signed.
    """
    t      = results['t']
    n_dets = results['true_params'].shape[1]

    for det_idx in range(n_dets):
        fig, axes = plt.subplots(6, 1, figsize=(11, 14), squeeze=False, sharex=True)
        fig.suptitle(f'Detector {det_idx} — alignment parameter tracking\n'
                     + results['meta']['description'], fontsize=11)

        for p_idx in range(6):
            ax   = axes[p_idx, 0]
            true = results['true_params'][:, det_idx, p_idx]
            pred = results['pred_params'][:, det_idx, p_idx]

            ax.plot(t, true, color='grey', linewidth=1.6, label='injected')
            ax.plot(t, pred, color=_PARAM_COLORS[p_idx], linewidth=1.0,
                    alpha=0.85, label='predicted')
            ax.axhline(0, color='k', linewidth=0.4, linestyle=':')

            s = _fit_stats(true, pred)
            ax.text(0.01, 0.97,
                    f'slope={s["slope"]:.3f}  r2={s["r2"]:.3f}\n'
                    f'bias={s["bias"]:.2e}  rmse={s["rmse"]:.2e}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=7,
                    bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.8))

            ax.set_ylabel(_param_label(p_idx), fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=7)

        axes[-1, 0].set_xlabel('Time step', fontsize=8)
        fig.tight_layout()
        out = Path(output_dir) / f'param_tracking_det{det_idx}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'saved {out}')


def plot_param_error(results, output_dir, smooth_window):
    """(predicted - injected) per parameter over time — exposes lag and bias."""
    t      = results['t']
    n_dets = results['true_params'].shape[1]

    for det_idx in range(n_dets):
        fig, axes = plt.subplots(6, 1, figsize=(11, 14), squeeze=False, sharex=True)
        fig.suptitle(f'Detector {det_idx} — parameter error (predicted - injected)\n'
                     f'faint = raw, solid = rolling mean w={smooth_window}',
                     fontsize=11)

        for p_idx in range(6):
            ax  = axes[p_idx, 0]
            err = results['pred_params'][:, det_idx, p_idx] - \
                  results['true_params'][:, det_idx, p_idx]

            ax.plot(t, err, color=_PARAM_COLORS[p_idx], alpha=0.2, linewidth=0.6)
            ax.plot(t, _smooth(err, smooth_window),
                    color=_PARAM_COLORS[p_idx], linewidth=1.5)
            ax.axhline(0, color='k', linewidth=0.5, linestyle=':')

            ax.set_ylabel(f'd {_param_label(p_idx)}', fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1, 0].set_xlabel('Time step', fontsize=8)
        fig.tight_layout()
        out = Path(output_dir) / f'param_error_det{det_idx}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'saved {out}')


def plot_param_correlation(results, output_dir):
    """
    Predicted vs injected scatter, one panel per (parameter, detector).

    The fitted slope is the model's gain for that misalignment type: 1 means it
    is recovered at full scale, 0 means the model is blind to it.  This is the
    plot to read when asking which misalignment types the model handles well.
    """
    n_dets = results['true_params'].shape[1]

    fig, axes = plt.subplots(6, n_dets, figsize=(4 * n_dets, 18), squeeze=False)
    fig.suptitle('Predicted vs injected alignment parameters\n'
                 + results['meta']['description'], fontsize=11)

    for p_idx in range(6):
        for det_idx in range(n_dets):
            ax   = axes[p_idx, det_idx]
            true = results['true_params'][:, det_idx, p_idx]
            pred = results['pred_params'][:, det_idx, p_idx]

            ax.scatter(true, pred, s=4, alpha=0.35, color=_PARAM_COLORS[p_idx])

            lo = float(min(true.min(), pred.min()))
            hi = float(max(true.max(), pred.max()))
            if hi <= lo:
                lo, hi = lo - 1e-6, hi + 1e-6
            ax.plot([lo, hi], [lo, hi], color='k', linewidth=0.8,
                    linestyle='--', label='y = x')

            s = _fit_stats(true, pred)
            if np.isfinite(s['slope']):
                xs = np.array([lo, hi])
                intercept = np.polyfit(true, pred, 1)[1]
                ax.plot(xs, s['slope'] * xs + intercept,
                        color='firebrick', linewidth=1.0, label='fit')

            ax.text(0.03, 0.96, f'slope={s["slope"]:.3f}\nr2={s["r2"]:.3f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=7,
                    bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.8))

            ax.set_xlabel(f'injected {_param_label(p_idx)}', fontsize=8)
            ax.set_ylabel(f'predicted {_param_label(p_idx)}', fontsize=8)
            ax.tick_params(labelsize=7)
            if p_idx == 0:
                ax.set_title(f'Detector {det_idx}', fontsize=10)
            if p_idx == 0 and det_idx == 0:
                ax.legend(fontsize=7, loc='lower right')

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(output_dir) / 'param_correlation.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def plot_raw9_tracking(results, output_dir):
    """
    True vs predicted for the 9 numbers the network actually emits per detector.

    No inversion is involved, so a disagreement here is the model's, whereas a
    disagreement only in the 6-parameter view points at the inversion.
    """
    t      = results['t']
    n_dets = results['true_raw'].shape[1]

    for det_idx in range(n_dets):
        fig, axes = plt.subplots(9, 1, figsize=(11, 18), squeeze=False, sharex=True)
        fig.suptitle(f'Detector {det_idx} — raw 9-parameter network output\n'
                     + results['meta']['description'], fontsize=11)

        for r_idx in range(9):
            ax   = axes[r_idx, 0]
            true = results['true_raw'][:, det_idx, r_idx]
            pred = results['pred_raw'][:, det_idx, r_idx]

            ax.plot(t, true, color='grey', linewidth=1.6, label='true')
            ax.plot(t, pred, color='steelblue', linewidth=1.0, alpha=0.85, label='predicted')
            ax.axhline(0, color='k', linewidth=0.4, linestyle=':')

            s = _fit_stats(true, pred)
            ax.text(0.01, 0.95, f'slope={s["slope"]:.3f}  r2={s["r2"]:.3f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=7,
                    bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.8))

            ax.set_ylabel(_RAW9_LABELS[r_idx], fontsize=8)
            ax.tick_params(labelsize=7)
            if r_idx == 0:
                ax.legend(fontsize=7, loc='upper right')

        axes[-1, 0].set_xlabel('Time step', fontsize=8)
        fig.tight_layout()
        out = Path(output_dir) / f'raw9_tracking_det{det_idx}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'saved {out}')


def plot_frame_health(results, output_dir):
    """How far the predicted local axes drift from an orthonormal pair."""
    t      = results['t']
    n_dets = results['frame_health'].shape[1]

    specs = [('|cos(lx, ly)|', 'firebrick', 0),
             ('||lx|| - 1',    'steelblue', 1),
             ('||ly|| - 1',    'seagreen',  2)]

    fig, axes = plt.subplots(3, n_dets, figsize=(5 * n_dets, 8), squeeze=False)
    fig.suptitle('Orthonormality of the predicted local axes\n'
                 '(the network is not constrained to emit an orthonormal frame)',
                 fontsize=11)

    for det_idx in range(n_dets):
        for row, (label, colour, idx) in enumerate(specs):
            ax = axes[row, det_idx]
            ax.plot(t, results['frame_health'][:, det_idx, idx],
                    color=colour, linewidth=0.9)
            ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel('Time step', fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f'Detector {det_idx}', fontsize=10)

    fig.tight_layout()
    out = Path(output_dir) / 'frame_health.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def plot_misalignment_tracking(results, output_dir):
    """True vs predicted misalignment magnitude per detector."""
    t      = results['t']
    n_dets = results['true_center_shift'].shape[1]

    fig, axes = plt.subplots(2, n_dets, figsize=(5 * n_dets, 7), squeeze=False)
    fig.suptitle('True vs predicted misalignment magnitude', fontsize=12)

    for det_idx in range(n_dets):
        for row, (key_true, key_pred, ylabel) in enumerate([
            ('true_center_shift', 'pred_center_shift', '|centre shift| (mm)'),
            ('true_orient_shift', 'pred_orient_shift', '|orientation shift| (arb.)'),
        ]):
            ax = axes[row, det_idx]
            ax.plot(t, results[key_true][:, det_idx],
                    color='steelblue', linewidth=1.0, alpha=0.8, label='true')
            ax.plot(t, results[key_pred][:, det_idx],
                    color='darkorange', linewidth=1.0, alpha=0.8, label='predicted')
            ax.set_title(f'Detector {det_idx}', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel('Time step', fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = Path(output_dir) / 'misalignment_tracking.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def plot_residual_evolution(results, output_dir, smooth_window):
    """
    Residual of the newest track (before/after correction) over time.
    One figure per detector, per scope (local / global).
    """
    t      = results['t']
    n_dets = results['local_before'].shape[1]

    for det_idx in range(n_dets):
        for scope, specs, key_b, key_a in [
            ('local',
             [('Local X', 'steelblue',    0),
              ('Local Y', 'darkorange',   1),
              ('Local R', 'mediumpurple', 2)],
             'local_before', 'local_after'),
            ('global',
             [('Global X', 'steelblue',   0),
              ('Global Y', 'seagreen',    1),
              ('Global Z', 'darkorange',  2),
              ('Global R', 'mediumpurple',3)],
             'global_before', 'global_after'),
        ]:
            n_coords = len(specs)
            fig, axes = plt.subplots(1, n_coords, figsize=(4 * n_coords, 4), squeeze=False)
            fig.suptitle(
                f'Detector {det_idx} — {scope} residual of newest track\n'
                f'(faint = raw, solid = rolling mean w={smooth_window})',
                fontsize=10,
            )

            for c_idx, (coord_label, colour, idx) in enumerate(specs):
                ax = axes[0, c_idx]
                b  = results[key_b][:, det_idx, idx]
                a  = results[key_a][:, det_idx, idx]

                ax.plot(t, b, color='grey',  alpha=0.15, linewidth=0.6)
                ax.plot(t, a, color=colour,  alpha=0.15, linewidth=0.6)
                ax.plot(t, _smooth(b, smooth_window), color='grey',  linewidth=1.5, label='before')
                ax.plot(t, _smooth(a, smooth_window), color=colour,  linewidth=1.5, label='after')
                ax.axhline(0, color='k', linewidth=0.5, linestyle=':')

                ax.set_title(coord_label, fontsize=9)
                ax.set_ylabel('Residual (mm)', fontsize=8)
                ax.set_xlabel('Time step', fontsize=8)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

            fig.tight_layout()
            out = Path(output_dir) / f'residual_evolution_det{det_idx}_{scope}.png'
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'saved {out}')


def plot_summary(results, output_dir, smooth_window):
    """
    Three-row summary per detector:
      row 0 — true centre-shift magnitude vs time
      row 1 — local R residual (before / after, rolling mean)
      row 2 — global R residual (before / after, rolling mean)
    """
    t      = results['t']
    n_dets = results['true_center_shift'].shape[1]

    fig, axes = plt.subplots(3, n_dets, figsize=(5 * n_dets, 9), squeeze=False)
    fig.suptitle('Real-time alignment: misalignment drift vs residual correction', fontsize=11)

    for det_idx in range(n_dets):
        # row 0: centre shift
        ax = axes[0, det_idx]
        ax.plot(t, results['true_center_shift'][:, det_idx],
                color='steelblue', linewidth=1.0, label='true |Dcentre|')
        ax.plot(t, results['pred_center_shift'][:, det_idx],
                color='darkorange', linewidth=1.0, alpha=0.7, label='predicted |Dcentre|')
        ax.set_title(f'Detector {det_idx}', fontsize=10)
        ax.set_ylabel('|centre shift| (mm)', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Time step', fontsize=8)

        # row 1: local R
        ax   = axes[1, det_idx]
        b    = results['local_before'][:, det_idx, 2]
        a    = results['local_after'][:, det_idx, 2]
        ax.plot(t, _smooth(b, smooth_window), color='grey',        linewidth=1.5, label='before')
        ax.plot(t, _smooth(a, smooth_window), color='mediumpurple', linewidth=1.5, label='after')
        ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Local R residual (mm)', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Time step', fontsize=8)

        # row 2: global R
        ax  = axes[2, det_idx]
        b   = results['global_before'][:, det_idx, 3]
        a   = results['global_after'][:, det_idx, 3]
        ax.plot(t, _smooth(b, smooth_window), color='grey',        linewidth=1.5, label='before')
        ax.plot(t, _smooth(a, smooth_window), color='mediumpurple', linewidth=1.5, label='after')
        ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Global R residual (mm)', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Time step', fontsize=8)

    fig.tight_layout()
    out = Path(output_dir) / 'summary.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def plot_scan_comparison(scorecards, output_dir):
    """
    Side-by-side comparison of the six isolated single-parameter studies.

    scorecards : {param_name: scorecard dict} — each from a run in which only
                 that parameter was injected.
    """
    names = [n for n in _PARAM_NAMES if n in scorecards]

    def _per_detector(param, key):
        """
        Mean and spread over detectors.  Pooling every detector into one fit
        would hide the fact that they can have very different gains — and a
        common shift of the whole telescope is partly degenerate with the track
        direction, so they genuinely do differ.
        """
        entry  = scorecards[param]['params'][param]
        values = [stats[key] for det_key, stats in entry.items() if det_key != 'all']
        return float(np.nanmean(values)), float(np.nanstd(values))

    slope, slope_err = zip(*[_per_detector(n, 'slope') for n in names])
    r2,    r2_err    = zip(*[_per_detector(n, 'r2')    for n in names])
    rmse,  rmse_err  = zip(*[_per_detector(n, 'rmse')  for n in names])
    impr = [np.mean([d['improvement']
                     for d in scorecards[n]['residual'].values()]) for n in names]
    impr_err = [np.std([d['improvement']
                        for d in scorecards[n]['residual'].values()]) for n in names]

    x = np.arange(len(names))
    specs = [
        ('Response gain (fit slope)', slope, slope_err, 'steelblue', 1.0),
        ('Correlation r2',            r2,    r2_err,    'seagreen',  1.0),
        ('Parameter RMSE',            rmse,  rmse_err,  'darkorange', None),
        ('Local 2D residual improvement\n(xy RMS before / xy RMS after)',
         impr, impr_err, 'mediumpurple', 1.0),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Isolated single-parameter injection — which misalignment types '
                 'does the model recover best?\n'
                 'bars = mean over detectors, error bars = spread across detectors',
                 fontsize=12)

    for ax, (title, values, errors, colour, reference) in zip(axes, specs):
        ax.bar(x, values, yerr=errors, capsize=4, color=colour, alpha=0.85,
               error_kw={'ecolor': 'k', 'linewidth': 0.8})
        if reference is not None:
            ax.axhline(reference, color='k', linewidth=0.8, linestyle='--',
                       label='perfect')
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25)

    axes[2].set_yscale('log')

    fig.tight_layout()
    out = Path(output_dir) / 'scan_comparison.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


def make_all_plots(results, output_dir, smooth_window):
    """Every per-run figure, plus the scorecard.  Returns the scorecard dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_trajectory(results, output_dir)
    plot_param_tracking(results, output_dir)
    plot_param_error(results, output_dir, smooth_window)
    plot_param_correlation(results, output_dir)
    plot_raw9_tracking(results, output_dir)
    plot_frame_health(results, output_dir)
    plot_misalignment_tracking(results, output_dir)
    plot_residual_evolution(results, output_dir, smooth_window)
    plot_summary(results, output_dir, smooth_window)

    scorecard = compute_scorecard(results)
    write_scorecard(scorecard, output_dir)
    return scorecard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_params(spec):
    if spec.strip().lower() == 'all':
        return list(range(6))
    out = []
    for token in spec.split(','):
        token = token.strip()
        if token not in _PARAM_NAMES:
            raise ValueError(f'unknown parameter {token!r}, expected one of {_PARAM_NAMES} or "all"')
        out.append(_PARAM_NAMES.index(token))
    return out


def _parse_dets(spec, n_dets):
    if spec.strip().lower() == 'all':
        return list(range(n_dets))
    out = []
    for token in spec.split(','):
        idx = int(token.strip())
        if not 0 <= idx < n_dets:
            raise ValueError(f'detector index {idx} out of range (0..{n_dets - 1})')
        out.append(idx)
    return out


def get_args():
    p = argparse.ArgumentParser(
        description='Sliding-window real-time alignment simulation'
    )
    p.add_argument('--config',  type=str, default='config_narrow.yaml',
                   help='path to model config yaml')
    p.add_argument('--dataset-config', type=str, default=None,
                   help='data-generation yaml to take the detector geometry and '
                        'beam from; must match what the model was trained on, '
                        'otherwise the model is evaluated out of distribution')
    p.add_argument('--arch',    type=str, default='auto',
                   choices=('auto', 'raw9', 'physical'),
                   help='model architecture; auto infers it from the config '
                        '(num_detectors -> physical, out_features -> raw9)')
    p.add_argument('--device',  type=str, default='cpu', choices=('cuda', 'cpu'))
    p.add_argument('--gpu-id',  type=int, default=0)
    p.add_argument('--output',  type=str, default='plots/sliding_window',
                   help='output directory | default: plots/sliding_window')
    p.add_argument('--steps',   type=int, default=1000,
                   help='total time steps (tracks generated) | default: 1000')
    p.add_argument('--window',  type=int, default=50,
                   help='sliding window size W | default: 50')

    p.add_argument('--profile', type=str, default='walk', choices=_PROFILES,
                   help='time profile of the injected misalignment | default: walk')
    p.add_argument('--params',  type=str, default='all',
                   help='comma-separated subset of %s, or "all" | default: all'
                        % ','.join(_PARAM_NAMES))
    p.add_argument('--dets',    type=str, default='all',
                   help='comma-separated detector indices, or "all" | default: all')
    p.add_argument('--shape',   type=str, default='uniform',
                   choices=('uniform', 'observable'),
                   help='uniform applies the same value to every selected plane, '
                        'which for all six parameters is a blind mode that straight '
                        'tracks cannot see; observable projects onto the detectable '
                        'subspace (the second difference) at the same amplitude')

    p.add_argument('--drift-center', type=float, default=0.002,
                   help='walk: per-step std of dx/dy/dz (mm) | default: 0.002')
    p.add_argument('--drift-angle',  type=float, default=0.002,
                   help='walk: per-step std of nu/nv/rho (rad) | default: 0.002')
    p.add_argument('--amp-center',   type=float, default=0.05,
                   help='ramp/sine/static amplitude for dx/dy/dz (mm); default 0.05 '
                        '= 1 sigma of the training center_shift')
    p.add_argument('--amp-angle',    type=float, default=0.0873,
                   help='ramp/sine/static amplitude for nu/nv/rho (rad); default 0.0873 '
                        '= 1 sigma of the training normal_shift / axes_rotation')
    p.add_argument('--period',       type=int,   default=200,
                   help='sine: steps per full period | default: 200')
    p.add_argument('--sine-phase-step', type=float, default=0.0,
                   help='sine: phase offset added per detector index (rad) | default: 0')

    p.add_argument('--scan', action='store_true',
                   help='run one isolated simulation per parameter and compare them; '
                        'takes 6x as long as a single run')

    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--smooth',  type=int, default=30,
                   help='rolling-mean window for residual plots | default: 30')
    return p.parse_args()


def _load_model(args):
    """Build the right architecture for this config and load its checkpoint."""
    with open(args.config, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    arch = detect_arch(config['model']) if args.arch == 'auto' else args.arch

    config_dir      = Path(args.config).resolve().parent
    checkpoint_path = Path(config['checkpointing']['checkpoint_path'])
    if not checkpoint_path.is_absolute():
        checkpoint_path = config_dir / checkpoint_path

    model_cls = PhysicalMLP if arch == 'physical' else MLP
    model     = model_cls(**config['model']).to(args.device)
    Checkpointer(model, checkpoint_path=checkpoint_path).load(device=args.device)

    print(f'architecture: {arch}')
    return Predictor(model, arch)


def _simulate(predictor, args, active_params, active_dets, amps, drifts, output_dir):
    """Build a trajectory, run the simulation, and emit every plot for it."""
    n_dets = len(_DETECTORS)
    rng    = np.random.default_rng(args.seed)

    trajectory = build_trajectory(
        n_steps       = args.steps,
        n_dets        = n_dets,
        profile       = args.profile,
        active_params = active_params,
        active_dets   = active_dets,
        amps          = amps,
        drifts        = drifts,
        period        = args.period,
        rng           = rng,
        phase_step    = args.sine_phase_step,
        shape         = args.shape,
    )

    description = describe_injection(args.profile, active_params, active_dets,
                                     amps, drifts, args.period)
    blind = weak_modes.blind_fraction(trajectory, detector_positions())
    print(f'\nInjection: {description}')
    print(f'  shape: {args.shape};  unobservable (weak-mode) fraction: {blind * 100:.1f}%')
    if blind > 0.5:
        print('  WARNING: most of this injection is a mode straight tracks cannot '
              'see.\n           The model cannot track it however well it is '
              'trained. Use --shape observable.')

    meta = {
        'description':   description,
        'profile':       args.profile,
        'active_params': [_PARAM_NAMES[p] for p in active_params],
        'active_dets':   list(active_dets),
        'shape':         args.shape,
        'blind_fraction': blind,
        'arch':          predictor.arch,
        'steps':         args.steps,
        'window':        args.window,
        'seed':          args.seed,
    }

    results = run_simulation(
        predictor   = predictor,
        device      = args.device,
        trajectory  = trajectory,
        window_size = args.window,
        seed        = args.seed,
        meta        = meta,
    )

    return make_all_plots(results, output_dir, args.smooth)


def main():
    args = get_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_id)

    if args.dataset_config is not None:
        load_dataset_config(args.dataset_config)

    predictor = _load_model(args)
    n_dets = len(_DETECTORS)

    amps   = np.zeros(6)
    drifts = np.zeros(6)
    for p in _TRANSLATION_PARAMS:
        amps[p], drifts[p] = args.amp_center, args.drift_center
    for p in _ANGLE_PARAMS:
        amps[p], drifts[p] = args.amp_angle, args.drift_angle

    active_dets = _parse_dets(args.dets, n_dets)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.scan:
        scorecards = {}
        for p_idx, name in enumerate(_PARAM_NAMES):
            print(f'\n{"=" * 70}\nScan {p_idx + 1}/6 — isolating {name}\n{"=" * 70}')
            scorecards[name] = _simulate(
                predictor, args,
                active_params = [p_idx],
                active_dets   = active_dets,
                amps          = amps,
                drifts        = drifts,
                output_dir    = output_root / name,
            )
        plot_scan_comparison(scorecards, output_root)
    else:
        _simulate(
            predictor, args,
            active_params = _parse_params(args.params),
            active_dets   = active_dets,
            amps          = amps,
            drifts        = drifts,
            output_dir    = output_root,
        )

    print(f'\nDone. Plots saved to {output_root}/')


if __name__ == '__main__':
    main()
