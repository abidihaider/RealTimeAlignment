"""
MLP that predicts the six physical misalignment parameters per detector.

Why not the 27 raw numbers
--------------------------
The original head regresses [d_center(3), d_local_x(3), d_local_y(3)] per
detector.  Two problems with that target:

  * It is over-parameterised — 9 numbers for 6 degrees of freedom — so nothing
    keeps the predicted local axes orthonormal.
  * In-plane roll and normal tilt are *mixed* into the same d_local_x /
    d_local_y components.  Measured on the training distribution, a 1-sigma
    tilt moves the hits by only 0.1-0.3 bins whereas a 1-sigma roll moves them
    by 1-3 bins, yet tilt dominates the variance of those components.  A plain
    MSE therefore spends most of its gradient on the direction that carries
    almost no information, burying the roll signal that is actually learnable.

Predicting (dx, dy, dz, nu, nv, rho) separates the two, guarantees a valid
orthonormal frame by construction, and lets the loss weight each degree of
freedom by how observable it actually is.

The parameterisation is the one implemented by rtal.geometry.misalign.Misalign,
so it is exactly the one the diagnostics in simulate_sliding_window.py report.
"""
import torch
from torch import nn

from rtal.geometry.misalign import Misalign
from rtal.models.mlp_no_residual import SubsetSolver, CloudBatchNorm
from rtal.utils.utils import get_activ_layer

# Default per-parameter scale, in the units of the training distribution:
# 0.05 mm centre shift, 0.0873 rad tilt / roll.  The network works in units of
# these, so its raw outputs are O(1) and the head is well conditioned.
_DEFAULT_PARAM_SCALE = (0.05, 0.05, 0.05, 0.0873, 0.0873, 0.0873)

# Misalign builds the normal as sqrt(1 - nu^2 - nv^2), so the tilt pair must
# stay strictly inside the unit disc.
_TILT_LIMIT = 0.95
_EPS = 1e-8


class MisalignLayer(nn.Module):
    """
    Differentiable (nu, nv, rho) -> (local_x, local_y), device- and dtype-aware.

    Wraps rtal.geometry.misalign.Misalign rather than reimplementing it, so the
    forward map here and the inverse map used by the diagnostics cannot drift
    apart.  The reference vectors are registered as buffers so they follow
    .to(device) and are saved with the state dict.
    """

    def __init__(self,
                 u_ref = (-1.0, 0.0, 0.0),
                 v_ref = ( 0.0, 0.0, 1.0)):
        super().__init__()

        misaligner = Misalign(u_ref=torch.tensor(u_ref, dtype=torch.float32),
                              v_ref=torch.tensor(v_ref, dtype=torch.float32))
        self.register_buffer('u_ref', misaligner.u_ref)
        self.register_buffer('v_ref', misaligner.v_ref)
        self.register_buffer('n_ref', misaligner.n_ref)

        # kept as a plain attribute, not a submodule — it holds no parameters
        self._misaligner = misaligner

    def _sync(self):
        self._misaligner.u_ref = self.u_ref
        self._misaligner.v_ref = self.v_ref
        self._misaligner.n_ref = self.n_ref

    def forward(self, nu, nv, rho):
        """nu, nv, rho: (...) -> local_x, local_y: (..., 3)"""
        self._sync()
        return self._misaligner.misalign(nu, nv, rho)

    def inverse(self, local_x, local_y):
        """local_x, local_y: (..., 3) -> nu, nv, rho: (...)"""
        self._sync()
        return self._misaligner.get_misalignment(local_x, local_y)


def detector_to_params(detector, detector_start, misalign_layer):
    """
    Inverse of `params_to_detector` — turns a pair of 9-number detector vectors
    into the 6 physical parameters.  Used to build the regression target from
    the (detector_start, detector_curr) pair stored in the dataset.

    detector, detector_start : (..., n_dets, 9)
    Returns                    (..., n_dets, 6)
    """
    nu, nv, rho = misalign_layer.inverse(detector[..., 3:6], detector[..., 6:9])
    d_center = detector[..., :3] - detector_start[..., :3]
    return torch.cat([d_center,
                      nu.unsqueeze(-1),
                      nv.unsqueeze(-1),
                      rho.unsqueeze(-1)], dim=-1)


def bound_tilt(nu, nv, limit=_TILT_LIMIT):
    """
    Squash the tilt pair into the unit disc, smoothly and jointly.

    Clamping each component separately would still allow nu^2 + nv^2 up to 2,
    so the radius is squashed instead.  Near the training scale (|nu| ~ 0.09)
    this is within 0.3% of the identity, so it costs essentially nothing.
    """
    radius = torch.sqrt(nu ** 2 + nv ** 2 + _EPS)
    factor = limit * torch.tanh(radius / limit) / radius
    return nu * factor, nv * factor


def params_to_detector(params, detector_start, misalign_layer):
    """
    Physical parameters -> the 9-number detector representation.

    params         : (..., n_dets, 6) — [dx, dy, dz, nu, nv, rho]
    detector_start : (..., n_dets, 9) — nominal geometry
    Returns          (..., n_dets, 9)

    Differentiable, so the geometric term of the loss can be taken through it.
    """
    center = detector_start[..., :3] + params[..., :3]
    local_x, local_y = misalign_layer(params[..., 3], params[..., 4], params[..., 5])
    return torch.cat([center, local_x, local_y], dim=-1)


class PhysicalMLP(nn.Module):
    """
    Same trunk as rtal.models.mlp_no_residual.MLP — pointwise embedding, a stack
    of subset solvers, mean pooling — with two changes:

      * the input readout is normalised before the first linear layer;
      * the head emits 6 physical parameters per detector instead of 9 raw
        numbers, with the tilt pair bounded to a physically valid range.
    """

    def __init__(self,
                 in_features,
                 num_detectors,
                 embedding_features,
                 subset_config,
                 activ,
                 batchnorm,
                 randperm,
                 input_scale = 25.0,
                 input_mean  = 0.0,
                 param_scale = _DEFAULT_PARAM_SCALE):

        super().__init__()

        self.num_detectors = num_detectors

        # Input normalisation.  The raw readout spans roughly +-50 bins while a
        # 1-sigma tilt modulates it by ~0.2%; without this the signal sits far
        # below what the first unnormalised linear layer can resolve.  It is a
        # fixed affine, so ONNX export and FPGA deployment are unaffected.
        self.register_buffer('input_scale', torch.as_tensor(input_scale, dtype=torch.float32))
        self.register_buffer('input_mean',  torch.as_tensor(input_mean,  dtype=torch.float32))

        # Output scaling, so the network regresses O(1) numbers.
        self.register_buffer('param_scale',
                             torch.as_tensor(param_scale, dtype=torch.float32))

        in_f = in_features

        layers = []
        for out_f in embedding_features:
            norm = CloudBatchNorm(in_f) if batchnorm else nn.Identity()
            layers += [norm, nn.Linear(in_f, out_f), get_activ_layer(activ)]
            in_f = out_f
        self.embed = nn.Sequential(*layers)

        layers = []
        for config in subset_config:
            layers += [SubsetSolver(in_features = in_f,
                                    subset_size = config[0],
                                    features    = config[1:],
                                    activ       = activ,
                                    batchnorm   = batchnorm,
                                    randperm    = randperm)]
            in_f = config[-1]
        self.solvers = nn.ModuleList(layers)

        self.output = nn.Linear(in_f, num_detectors * 6)

        self.misalign_layer = MisalignLayer()

    def normalize(self, data):
        return (data - self.input_mean) / self.input_scale

    def _head(self, pooled):
        """(B, F) -> physical parameters (B, n_dets, 6)."""
        params = self.output(pooled).reshape(-1, self.num_detectors, 6)
        params = params * self.param_scale

        nu, nv = bound_tilt(params[..., 3], params[..., 4])
        return torch.stack([params[..., 0], params[..., 1], params[..., 2],
                            nu, nv, params[..., 5]], dim=-1)

    def forward(self, data):
        """data: (batch_size, num_entries, in_features) -> (B, n_dets, 6)"""
        data = self.embed(self.normalize(data))
        for solver in self.solvers:
            data = solver(data)
        return self._head(data.mean(dim=1))

    def inference(self, data, randperm=False):
        """As forward, but with the option to disable random permutation."""
        data = self.embed(self.normalize(data))
        for solver in self.solvers:
            data = solver.inference(data, randperm)
        return self._head(data.mean(dim=1))

    def predict_detector(self, data, detector_start, randperm=False):
        """Convenience: run inference and return the 9-number geometry."""
        params = self.inference(data, randperm=randperm)
        return params_to_detector(params, detector_start, self.misalign_layer), params
