"""
Weak (unobservable) misalignment modes for a telescope of planes read out by
straight tracks.

The problem
-----------
A straight track that crosses planes at positions L_0, L_1, ... is detected
only through the fact that its hits are collinear. A misalignment whose induced
transverse displacement is an *affine function of L* maps every straight track
to another straight track, so it changes nothing that can be measured. Such
modes are invisible no matter how much data is collected.

For three equally spaced planes an eigenanalysis of the straightness residual
finds exactly nine of the eighteen parameters to be blind, at ~1e-8 of the
best-determined mode:

    dx, dy, dz   constant and linear in L   (6)  global translation and shear
    nu, nv, rho  constant                   (3)  global rotation and roll

Generating misalignments that contain these components asks a network to
predict something no measurement can determine. It cannot, so the components
show up as irreducible loss -- and because the loss still rewards trying, they
push the shared trunk to fit noise.

The fix
-------
`constrain` removes them, which is the same convention real alignment uses:
the global frame is *defined* by the detector, so the global degrees of freedom
are fixed rather than fitted. What remains is the part a measurement can
actually pin down.
"""
import numpy as np
import torch

from rtal.geometry.misalign import Misalign

# Order of the six per-detector parameters, matching rtal.geometry.misalign.
PARAM_NAMES = ('dx', 'dy', 'dz', 'nu', 'nv', 'rho')

# Which family of L-dependence is unobservable for each parameter.
#   'affine'   -> constant and linear in L are both blind
#   'constant' -> only the common mode is blind
_BLIND_FAMILY = {
    'dx':  'affine',    'dy':  'affine',    'dz':  'affine',
    'nu':  'constant',  'nv':  'constant',  'rho': 'constant',
}


def blind_family(name):
    """Which L-dependence family is blind for a given parameter."""
    return _BLIND_FAMILY[name]


def blind_basis(positions, family):
    """
    Orthonormal basis (n_dets, k) of the blind subspace for one parameter.

    'constant' spans {1}; 'affine' spans {1, L}.
    """
    positions = np.asarray(positions, dtype=np.float64)
    ones = np.ones_like(positions)

    columns = [ones] if family == 'constant' else [ones, positions]
    basis, _ = np.linalg.qr(np.stack(columns, axis=-1))
    return basis


def constrain(params, positions, families=None):
    """
    Project a misalignment onto the observable subspace.

    params    : (..., n_dets, 6) — [dx, dy, dz, nu, nv, rho] per detector
    positions : (n_dets,) — the plane coordinate along the beam
    families  : optional override of the per-parameter blind family

    Returns an array of the same shape with the blind components removed.

    With three planes this keeps 1 of 3 degrees of freedom for each translation
    (the "curvature", i.e. the second difference) and 2 of 3 for each rotation.
    """
    params    = np.asarray(params, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    families  = families or _BLIND_FAMILY

    n_dets = params.shape[-2]
    if n_dets != len(positions):
        raise ValueError(f'{n_dets} detectors but {len(positions)} positions')

    out = params.copy()
    for j, name in enumerate(PARAM_NAMES):
        family = families[name]
        rank = 1 if family == 'constant' else 2
        if n_dets <= rank:
            # Nothing survives the projection — every mode of this parameter is
            # blind. Zero it rather than leave an unobservable target behind.
            out[..., :, j] = 0.0
            continue

        basis = blind_basis(positions, family)              # (n_dets, rank)
        values = out[..., :, j]                             # (..., n_dets)
        coeff = values @ basis                              # (..., rank)
        out[..., :, j] = values - coeff @ basis.T

    return out


def blind_fraction(params, positions, families=None):
    """
    Fraction of the squared parameter magnitude that lies in the blind
    subspace — i.e. how much of this misalignment is unknowable.

    Computed in raw parameter units, so scale each parameter by its generation
    sigma first if you want a unit-free number.
    """
    params = np.asarray(params, dtype=np.float64)
    kept   = constrain(params, positions, families)

    total = float(np.sum(params ** 2))
    if total == 0.0:
        return 0.0
    return float(np.sum((params - kept) ** 2) / total)


# ---------------------------------------------------------------------------
# Applying the constraint to Detector objects
# ---------------------------------------------------------------------------

def beam_positions(detectors):
    """
    The plane coordinate along the beam, taken as the axis along which the
    detector centres are most spread out.
    """
    centers = np.stack([d.center['start'] for d in detectors])
    axis = int(np.argmax(centers.std(axis=0)))
    return centers[:, axis], axis


def constrain_detectors(detectors, eps=1e-8):
    """
    Remove the blind modes from an already-misaligned set of detectors.

    Reads the six physical parameters out of each detector's current geometry,
    projects the set onto the observable subspace, and writes the resulting
    geometry back. The stored `curr` state is rebuilt exactly from the
    constrained parameters, so the saved truth and the saved readout stay
    consistent — no small-angle approximation leaks into the dataset.

    Requires every detector to start in the Misalign reference frame
    (local_x = [-1, 0, 0], local_y = [0, 0, 1]), which is what makes the
    per-detector parameters comparable in the first place.

    Returns the fraction of squared parameter magnitude that was removed.
    """
    misaligner = Misalign(dtype=torch.float64)
    u_ref = misaligner.u_ref.numpy()
    v_ref = misaligner.v_ref.numpy()

    for i, det in enumerate(detectors):
        if not (np.allclose(det.local_x['start'], u_ref, atol=1e-8) and
                np.allclose(det.local_y['start'], v_ref, atol=1e-8)):
            raise ValueError(
                f'detector {i} does not start in the Misalign reference frame; '
                f'the weak-mode constraint assumes a shared reference orientation'
            )

    local_x = torch.tensor(np.stack([d.local_x['curr'] for d in detectors]),
                           dtype=torch.float64)
    local_y = torch.tensor(np.stack([d.local_y['curr'] for d in detectors]),
                           dtype=torch.float64)
    nu, nv, rho = misaligner.get_misalignment(local_x, local_y)

    centers_start = np.stack([d.center['start'] for d in detectors])
    centers_curr  = np.stack([d.center['curr']  for d in detectors])

    params = np.concatenate([centers_curr - centers_start,
                             nu.numpy()[:, None],
                             nv.numpy()[:, None],
                             rho.numpy()[:, None]], axis=-1)      # (n_dets, 6)

    positions, _ = beam_positions(detectors)
    kept = constrain(params, positions)

    removed = float(np.sum((params - kept) ** 2))
    total   = float(np.sum(params ** 2))

    st = torch.as_tensor(kept, dtype=torch.float64)
    new_x, new_y = misaligner.misalign(st[:, 3], st[:, 4], st[:, 5])

    for i, det in enumerate(detectors):
        det.center['curr']  = centers_start[i] + kept[i, :3]
        det.local_x['curr'] = new_x[i].numpy()
        det.local_y['curr'] = new_y[i].numpy()
        det.normal['curr']  = np.cross(det.local_x['curr'], det.local_y['curr'])

    return removed / total if total > eps else 0.0
