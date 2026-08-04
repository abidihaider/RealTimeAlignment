# Real-time alignment: findings and changes

Session of 2026-08-04. Branch `feature/physical-params-multi-gpu`.

Everything below was measured on this repo unless explicitly flagged as unverified.
Numbers are reproducible from the scripts named in each section.

---

## 0. The headline: an earlier conclusion in this session was wrong

Partway through this work I concluded that the model "has no handle on detector
orientation". **That conclusion was an artifact of the diagnostics, not a property of
the model.**

`simulate_sliding_window.py` injected misalignment by applying the *same value to every
plane*. For all six parameters that is a **weak mode** — a misalignment straight tracks
cannot see (§2). So the entire `--scan` study measured the model's response to modes no
detector can resolve.

Same checkpoint (`checkpoints_narrow`), same weights, only the injection pattern changed:

| param | uniform (blind) slope / r² | observable slope / r² | 2D residual improvement (det 0/1/2) |
|-------|---------------------------|-----------------------|-------------------------------------|
| dx    | 0.112 / 0.017             | **0.904 / 0.967**     | 3.4× / 8.1× / 1.5× |
| dy    | −0.011 / 0.003            | 0.516 / 0.895         | 0.84× / 0.87× / 0.59× |
| dz    | 0.181 / 0.047             | **0.827 / 0.961**     | 3.6× / 9.0× / 1.5× |
| nu    | 0.046 / 0.082             | 0.611 / 0.931         | 0.46× / 1.01× / 0.99× |
| nv    | 0.002 / 0.000             | 0.685 / 0.907         | 0.51× / 1.05× / 0.96× |
| rho   | −0.004 / 0.001            | **0.819 / 0.947**     | 1.4× / 12.3× / 2.3× |

The original narrow model tracks tilt at r² = 0.93 and roll at r² = 0.95. It was never
broken.

But note the last column, which is the quantity that actually matters for alignment: for
`dx`, `dz` and `rho` the correction shrinks the residual by up to 12×, whereas for `nu`,
`nv` and `dy` it is **at or below 1.0** — the correction does not help and on detector 0
makes things worse. At slope ~0.6 the model under-predicts these parameters, and the
noise it injects exceeds the error it removes. That, not an inability to see tilt, is the
real deficiency to target.

Consequences for the rest of this document:

- §2 (weak modes), §3 (tilt scaling) and §4 (vertex spread) are independently measured
  and stand on their own.
- §5 (the physical-parameter model) was **motivated by the wrong premise**. Its
  individual merits are real but its headline justification — that roll was being
  buried by tilt in the shared 9-parameter components — is not supported now that roll
  measures r² = 0.95 with a 12× residual improvement on the old architecture. Treat it
  as an open decision, not a settled improvement.

---

## 1. Bugs found

### 1.1 `generate_dataset` wrote N identical copies of one event

`generate_one` called `np.random.seed(config['dataset']['random_seed'])` on every
invocation, so every sample in a run was byte-identical. Verified directly:
`sample_0 == sample_1 == sample_2`.

The README's data-generation command could not produce a training set.
`plot_residuals.py` sidestepped this by varying the seed in the config per call.

**Fixed:** `generate_one` takes an optional `seed`; `generate_dataset` advances it per
sample. Existing call sites are unaffected.

### 1.2 `ROMDataset` was unpicklable, breaking DataLoader workers

`__init__` stored `self.readout_processor = self.__get_raw_readout` — a bound reference
to a private (name-mangled) staticmethod, which cannot be pickled. Any `num_workers > 0`
crashed. This blocks multi-GPU training outright, since npz decompression is the input
bottleneck.

**Fixed:** stores the mode string and dispatches in a method.

### 1.3 `rtal` was not importable, so `torchrun` failed on all ranks

`python train/.../train.py` puts the *script's* directory on `sys.path`, not the working
directory. `setup.py` declared `packages=['rtal']`, so `rtal.data`, `rtal.models`,
`rtal.geometry` and `rtal.datasets` would not have been installed even with
`pip install -e .`; those directories also had no `__init__.py` and only resolved as
implicit namespace packages when the repo root happened to be on the path.

**Fixed:** added the five missing `__init__.py`, switched to `find_packages()`, declared
`pandas` and `matplotlib` (both already imported by scripts in the repo), added
`onnx`/`qat` extras. `pip install -e .` now resolves all subpackages from any directory.

### 1.4 The diagnostics injected only unobservable modes

See §0 and §2. **Fixed:** `--shape observable`, plus the weak-mode fraction is printed
on every run with a warning above 50%.

A second defect in the first version of that fix: for the rotation parameters the
observable subspace is 2-dimensional, and the representative was taken straight from an
SVD, whose choice among equal singular values is arbitrary. It came out as `(0, +1, -1)`,
giving **detector 0 no misalignment at all** and silently leaving a third of the scan
untested. Now built from an alternating template projected onto the observable subspace,
which yields the second difference `(+1, -2, +1)` — observable for every parameter and
non-zero on every detector.

### 1.5 `environment.yml` will not reproduce

It is a fully-pinned `linux-64` export from another machine
(`prefix: /home/yhuang2/miniconda3/envs/qat`), pins `python=3.14` and every build hash,
pins `torch==2.10.0+cu130` with no PyTorch index URL, and **contains no matplotlib** —
only `matplotlib-inline`, which is an IPython shim. The diagnostics cannot run in it.

**Not fixed** (left as-is). Use the explicit env in §8 instead.

### 1.6 Known-broken, deliberately not touched

`train/*/evaluate.py` (three byte-identical copies) each: import `rtal.models.mlp`
rather than the no-residual model, so the config keys and checkpoint will not match;
treat the model output as absolute detector parameters when `train.py` trains it as a
delta, making `diff_pc` wrong by construction; and have inverted `num_eval_particles`
logic. Out of scope for this session.

---

## 2. Weak modes — 9 of 18 parameters are unobservable

**Script:** `scratchpad/weak_modes.py` (analysis), `rtal/geometry/weak_modes.py` (library).

A straight track is detected only through the collinearity of its hits. A misalignment
whose induced transverse displacement is an **affine function of the plane position**
maps every straight track to another straight track, so it changes nothing measurable.

Eigenanalysis of the straightness residual — the second difference
`p0 − 2·p1 + p2` of the reconstructed transverse position, which is the only
misalignment information available *without knowing the track*, i.e. the network's
actual situation — gives, in µm of residual per 1σ excursion:

```
rank  microns/sigma  mode
   0      663.91      rho (curved)
   1      169.40      rho (linear-in-L)
   2      124.49      dx, dz (curved)
   3      124.43      dx, dz (curved)
   4       62.28      nu, nv (curved)
   5       61.20      nu, nv (curved)
   6       21.92      nv (linear-in-L)
   7       19.49      nu (linear-in-L)
   8       17.69      dy (curved)
   9        0.00001   rho (common)                        <-- BLIND
  10        0.00001   dz (linear-in-L)                    <-- BLIND
  11-16     0.00000   dx/dz (common), dx/dy (linear)      <-- BLIND
            0.00000     mixed with nu/nv (common)
  17        0.00000   dy (linear-in-L)                    <-- BLIND
```

Nine modes sit at ~10⁻⁸ of the best-determined mode, i.e. numerically zero:

| blind mode | physical meaning |
|---|---|
| `dx`, `dz` constant | global transverse translation |
| `dx`, `dz` linear-in-L | global shear |
| `dy` constant + linear | translation along the beam |
| `rho` constant | global roll about the beam axis |
| `nu`, `nv` constant (mixed with `dx`/`dy` linear) | **global rigid rotation** |

Modes 13/14 mixing `dx(linear-in-L)` with `nu(common)` are the signature of a rigid
rotation about the z-axis: tilt every plane and shift each proportionally to its
distance.

**Measured impact on generation:**

| | value |
|---|---|
| blind fraction of an unconstrained misalignment (through the real generator) | **45.3%** |
| blind fraction after `remove_weak_modes: true` | **0.0%** |
| observable signal lost by constraining | **0.6%** (397.08 → 394.59 µm) |
| target rms, unconstrained → constrained | 0.0515 → 0.0390 |
| surviving modes after projection | 9, condition number 36.9 |

45% of the training target was unlearnable. Removing it costs 0.6% of the signal.

**Fix:** `rtal/geometry/weak_modes.py` + `remove_weak_modes: true` in the generation
config. It reads the six physical parameters out of each detector, projects onto the
observable subspace, and rebuilds the geometry exactly from the constrained parameters,
so saved truth and saved readout stay consistent — no small-angle approximation enters
the dataset. This is the convention real alignment uses: the global frame is *defined*
by the detector, not fitted.

---

## 3. Why tilt is weakly observed — the `θr²/L` scaling

**Script:** `scratchpad/tilt_study.py`.

Tilting a plane by `θ` about its centre shifts a hit at local radius `r` by
`Δ ≈ r·θ·tanα`, where `α` is the incidence angle. Two consequences:

- At **normal incidence the effect vanishes at first order**, at any `r`. Spreading
  tracks over the sensor is not sufficient on its own; oblique tracks are required.
- With a near-point vertex, a hit at radius `r` on a plane at distance `L` must have
  arrived at `tanα ≈ r/L`. The two factors collapse: **`Δ ≈ θ·r²/L`**.

Verified against simulation within ~10% across every configuration tried.

The bottleneck is the **aperture stack**, not the beam. Three equal 10×10 mm sensors at
L = 10/20/30 mm: a track steep enough to reach the edge of detector 0 has already missed
detector 2, so detector 0 only ever uses the inner 23% of its own sensor
(`r_rms = 1.13 mm` of a 5 mm half-size). Widening the beam barely helps:

| | r_rms det0 | acceptance | tilt signal det0 |
|---|---|---|---|
| `direction_std` 0.1 | 1.13 mm | 81% | 0.10 bins |
| `direction_std` 0.35 | 1.35 mm | 13% | 0.14 bins |

19% more radius for a 6× loss of statistics — geometry-limited, not statistics-limited.

**Layout options (measured, not implemented):**

| layout | det0 | det1 | det2 | acceptance |
|---|---|---|---|---|
| current, 5/5/5 mm at L=10/20/30 | 0.10 | 0.21 | 0.31 bins | 81% |
| projective, half-sizes 5/10/15 mm | **0.97** | **1.94** | **2.92** | 71% |
| equal sensors, planes at 10/12/14 | 0.56 | 0.68 | 0.79 | 48% |

The projective layout is ~10× and puts tilt on par with roll. It requires a code change:
`Detector._RANGE_X`/`_RANGE_Y` are class constants shared by all instances, commented
"There should be nothing that can change this". The closer-plane option is config-only
but shortens the tracking baseline, degrading the direction fit.

---

## 4. Spreading the vertex decouples radius from incidence angle

**Script:** `scratchpad/vertex_study.py`.

The `r²/L` form exists *only* because the vertex is point-like. Spreading it transversely
breaks the lock, and because an offset track stays offset on all three planes it does so
without the aperture-stacking penalty that made simply widening the beam useless.

| config | corr(r, tanα) | acceptance |
|---|---|---|
| current, `vertex_std` 0.1, `direction_std` 0.10 | **0.99** | 81% |
| `vertex_std` 2, `direction_std` 0.15 | 0.12 | 47% |
| **`vertex_std` 3, `direction_std` 0.15** | **0.00** | **39%** |
| `vertex_std` 4, `direction_std` 0.20 | −0.01 | 23% |

Sensitivity in bins of hit displacement per 1σ, at the chosen operating point:

| param | det0 | det1 | det2 | mean |
|---|---|---|---|---|
| tilt, before | 0.10 | 0.20 | 0.30 | 0.20 |
| tilt, after | **0.26** | 0.27 | 0.39 | **0.31** (+55%) |
| roll, before | 0.97 | 1.93 | 2.90 | 1.93 |
| roll, after | 2.52 | 2.63 | 3.27 | 2.81 (+46%) |
| dx / dz | 0.50 | 0.50 | 0.50 | 0.50 (unchanged) |

The gain concentrates on detector 0 (**+160%**) — the plane the aperture stack was
starving. Translations are unchanged, as they should be: they do not depend on `r` or `α`.

Side effects: acceptance 81% → 39%, so `num_particles` goes 200 → 600 (generation time,
not lost information; events still yield 203–270 usable hits). The readout spread becomes
much more uniform across planes, `[8, 8, 16, 16, 24, 24]` → `[21, 21, 22, 22, 27, 27]`
bins, which is why `input_scale` had to be re-measured.

**Not measured:** the identifiability gain from decorrelation itself. Under a point vertex
both tilt (`∝ r²/L`) and normal-translation `dy` (`∝ r/L`) are pure functions of `r`;
decorrelated, tilt has a two-variable signature nothing else mimics. Only the magnitude
change above was quantified.

---

## 5. The physical-parameter model (motivation now in question — see §0)

**Files:** `rtal/models/mlp_physical.py`, `train/mlp_physical/`.

Same trunk as `mlp_no_residual` (pointwise embedding → subset solvers → mean pool), with:

- **Input normalization.** The raw readout went into the first `nn.Linear` at ±50 bins
  with a ~0.2% tilt modulation on top. Now normalized by measured per-feature std. It is
  a fixed affine, so ONNX export and FPGA deployment are unaffected.
- **6 physical parameters per detector** `(dx, dy, dz, nu, nv, rho)` instead of the 9 raw
  numbers, mapped back through `Misalign` — the existing forward map in the repo, reused
  so it cannot drift from the inverse used by the diagnostics. Removes 3 redundant dof
  and guarantees an orthonormal frame. The tilt pair is smoothly bounded to `nu²+nv² < 1`.
- **Observability-weighted loss** plus a real geometric term. The original computed the
  residual and then multiplied it by zero (`loss = diff + 0 * residual`).
- **DistributedDataParallel** via `torchrun`.

Loss weights, measured against the straightness observable on constrained draws:

```
param_weights: [1.0, 0.146, 1.0, 0.26, 0.245, 2.028]      # dx dy dz nu nv rho
```

An earlier version of these weights was measured from *raw hit displacement* with the
same value on all three planes — itself a blind mode — and overstated `rho` (3.21) and
tilt (0.35). Superseded.

**Measurement that motivated the 6-parameter head, and still stands:** under a plain
`nn.MSELoss` on the 27 raw numbers, the orientation block takes **77%** of the loss
(translation 23%), and within it the near-unobservable tilt dominates the variance. What
does *not* stand is the inference that this was preventing roll from being learned — §0
shows roll at r² = 0.95, with a 12× residual improvement, on the old architecture.

### Verified

| check | result |
|---|---|
| parameter round-trip (forward then inverse map) | 1.8e-15 |
| physical model: direct params vs inverted geometry | agree to 2.3e-7 |
| physical model frame orthonormality | ~1e-8 (raw-9 model drifts to ~1e-3) |
| DDP (2 procs, gloo/CPU): reduced metrics vs single process | match to 4 digits |
| DDP checkpoint keys | no `module.` prefix — diagnostics and ONNX can load them |
| both architectures through the full diagnostic plot set | pass |

### Not verified

- **Convergence.** Only 2-epoch smoke tests were run. `rms_rho` sat at the width of the
  training distribution, i.e. the model predicting zero — which is expected at 2 epochs
  and says nothing.
- **NCCL and 8-GPU scaling.** Only gloo on CPU with 2 processes.
- **`geom_weight: 1.0`** is a reasoned starting point, not tuned.
- Whether the architecture change is worth keeping at all, given §0.

---

## 6. Diagnostics (`train/mlp_no-residual/simulate_sliding_window.py`)

Rewritten this session. Beyond the original residual plots it now produces per-parameter,
signed, time-resolved true-vs-predicted tracking, and supports both architectures.

Key flags:

| flag | purpose |
|---|---|
| `--profile {walk,ramp,sine,static}` | time profile of the injected misalignment |
| `--params dx,dy,dz,nu,nv,rho\|all` | isolate one misalignment type |
| `--shape {uniform,observable}` | **use `observable`** — `uniform` is a blind mode (§0) |
| `--scan` | one isolated study per parameter + comparison figure |
| `--arch {auto,raw9,physical}` | inferred from the config by default |
| `--dataset-config` | take detector geometry and beam from a generation config |

`--dataset-config` matters: the script previously had the beam hardcoded, so evaluating a
spread-vertex model with the default collimated tracks would silently measure it out of
distribution.

The weak-mode fraction of the injection is printed on **every** run.

Outputs: `param_tracking_det{i}.png`, `param_error_det{i}.png`, `param_correlation.png`,
`raw9_tracking_det{i}.png`, `frame_health.png`, `scorecard.{png,txt,json}`, plus the
original trajectory/residual/summary plots and `scan_comparison.png` under `--scan`.

One caveat found in the metrics: **local R is invariant under in-plane roll**, so σ_before
is identically zero for `rho` and the R-improvement ratio is undefined for it. A 2D local
xy RMS metric was added and is what `scan_comparison.png` uses; local R is still reported
alongside with the caveat noted.

---

## 7. Open questions

1. **Is the physical-parameter architecture still justified?** (§0). The cheapest way to
   decide: re-run the scan with `--shape observable` on the existing narrow checkpoint
   and on a trained physical checkpoint, and compare.
2. **Projective detector layout** — ~10× on tilt (§3), needs per-detector sensor sizes,
   currently blocked by `Detector._RANGE_X/_RANGE_Y` being global constants.
3. **Does removing weak modes actually improve training?** Predicted yes (45% less noise
   in the target), unmeasured.
4. **`geom_weight` tuning.** If `rms_rho` is flat after ~20 epochs, raise it first.
5. Whether the decorrelation in §4 buys identifiability beyond the +55% magnitude.

---

## 8. Commands

### Environment

Do not use `environment.yml` (§1.5).

```bash
conda create -p /path/to/envs/rtal python=3.11 -y
conda activate /path/to/envs/rtal

# match your node's CUDA; check with nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pyyaml tqdm pandas matplotlib

pip install -e .        # so `rtal` imports from anywhere (§1.3)
```

`conda config --add pkgs_dirs /path/to/conda/pkgs` and `export PIP_CACHE_DIR=...` keep the
multi-GB wheels off your home directory.

### Generate the dataset

Spread vertex (§4) with weak modes removed (§2):

```bash
cd /path/to/RealTimeAlignment

python -m rtal.data.make_dataset \
    --config      train/mlp_physical/dataset_spread_vertex.yaml \
    --output-root /path/to/data/rom_spread_constrained \
    --num-train   200000 \
    --num-test    20000

export DATAROOT=/path/to/data/rom_spread_constrained
```

600 particles generated per event, 203–270 accepted and stored. Measured: ~7 minutes and
~7.4 GB for 200k/20k. Add `--overwrite` to regenerate in place.

### Train on 8 GPUs

```bash
torchrun --standalone --nproc_per_node=8 \
    train/mlp_physical/train.py \
    --config train/mlp_physical/config_spread_vertex.yaml \
    --num-workers 8
```

`batch_size: 128` is **per GPU** (effective 1024); `scale_lr: true` takes the base 1e-4 to
8e-4. Rank 0 owns logging and checkpointing; metrics are all-reduced first. Checkpoints
and `train_log.csv` / `valid_log.csv` land in `train/mlp_physical/checkpoints_spread_vertex/`.
`resume: true` means re-running continues where it stopped.

Single GPU:

```bash
python train/mlp_physical/train.py \
    --config train/mlp_physical/config_spread_vertex.yaml \
    --device cuda --gpu-id 0
```

**Watch `rms_rho`, `rms_nu`, `rms_nv`**, not the total loss. At init they sit at the width
of the training distribution (the model predicting zero); falling well below that is the
signal that training is working.

### Evaluate

Always pass `--shape observable` and `--dataset-config`:

```bash
python train/mlp_no-residual/simulate_sliding_window.py \
    --config         train/mlp_physical/config_spread_vertex.yaml \
    --dataset-config train/mlp_physical/dataset_spread_vertex.yaml \
    --steps 600 --window 50 --profile ramp --scan --shape observable \
    --amp-center 0.5 --amp-angle 0.25 \
    --output train/mlp_physical/plots/scan_observable
```

Read `scan_comparison.png` first, then `<param>/param_tracking_det0.png`.

Baseline for comparison — the existing narrow model, same injection:

```bash
python train/mlp_no-residual/simulate_sliding_window.py \
    --config train/mlp_no-residual/config_narrow.yaml \
    --steps 600 --window 50 --profile ramp --scan --shape observable \
    --amp-center 0.5 --amp-angle 0.25 \
    --output train/mlp_no-residual/plots/scan_observable
```

Sanity check that a blind injection is still detected as such:

```bash
python train/mlp_no-residual/simulate_sliding_window.py \
    --config train/mlp_no-residual/config_narrow.yaml \
    --steps 300 --window 50 --profile ramp --params rho \
    --output /tmp/blind_check
# expect: "unobservable (weak-mode) fraction: 100.0%" and a WARNING
```

---

## 9. Files changed

**New**

| file | purpose |
|---|---|
| `rtal/geometry/weak_modes.py` | weak-mode null space, projection, `constrain_detectors` |
| `rtal/data/make_dataset.py` | train/test splits with disjoint seed ranges |
| `rtal/models/mlp_physical.py` | 6-parameter head, input normalization |
| `train/mlp_physical/train.py` | DDP training, weighted + geometric loss |
| `train/mlp_physical/config.yaml` | point-vertex model config |
| `train/mlp_physical/dataset.yaml` | point-vertex generation config |
| `train/mlp_physical/config_spread_vertex.yaml` | **the one to use** |
| `train/mlp_physical/dataset_spread_vertex.yaml` | **the one to use** |
| `docs/ALIGNMENT_FINDINGS.md` | this file |

**Modified**

| file | change |
|---|---|
| `rtal/data/generate.py` | per-sample seed (§1.1); `remove_weak_modes` flag (§2) |
| `rtal/datasets/dataset.py` | picklable dispatch (§1.2) |
| `setup.py` | `find_packages`, deps, extras (§1.3) |
| `rtal/{,data,models,geometry,datasets}/__init__.py` | added (§1.3) |
| `train/mlp_no-residual/simulate_sliding_window.py` | rewritten (§6) |
| `.gitignore` | anchored `/data/`, `/generated_data/`, `/plots/`, `train/*/plots/` |

Analysis scripts live in the session scratchpad and are not committed:
`sensitivity.py`, `target_scales.py`, `tilt_study.py`, `vertex_study.py`, `weak_modes.py`.
