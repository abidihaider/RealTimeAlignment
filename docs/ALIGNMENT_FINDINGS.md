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

Two further defects in that fix, both found only by questioning it afterwards:

**(a) A zeroed detector.** For the rotation parameters the observable subspace is
2-dimensional, and the representative was taken straight from an SVD, whose choice among
equal singular values is arbitrary. It came out as `(0, +1, -1)`, giving **detector 0 no
misalignment at all** and silently leaving a third of the scan untested. Now built from an
alternating template projected onto the observable subspace, giving the second difference
`(+1, -2, +1)` — observable for every parameter and non-zero on every detector.

**(b) `observable` did not work for `walk` or phased `sine`.** The shape weights were
applied to each detector's *own* time series. That is equivalent to an observable spatial
pattern only when those series are identical, which holds for `ramp` and `static` but not
for `walk` (independent per-detector realisations) or `sine` with a phase offset. Measured
blind fraction under `--shape observable` before the fix:

| profile | blind |
|---|---|
| ramp, static | 0.0% ✓ |
| walk | **38.8%** ✗ |
| sine, phase 2.09 | **62.2%** ✗ — worse than leaving it `uniform` (25.7%) |

The time series is now drawn **once per parameter** and distributed across detectors by the
observable weights, so all four profiles give exactly 0.0%. `--sine-phase-step` is ignored
under `observable`, where the spatial pattern is fixed by construction. Default `uniform`
behaviour is unchanged, including the independent per-detector random walks.

### 1.7 Ragged batches from a variable number of accepted hits

`ROMDataset.__getitem__` took `[:num_particles]` of the accepted hits. Events hold a
variable number, so any event with fewer returns a short array and `default_collate`
fails with `RuntimeError: Trying to resize storage that is not resizable`.

`num_particles: 192` was chosen from a few-hundred-event sample whose minimum was 203.
Measured properly over 30000 events the distribution is:

    min 185   1st pct 205   median 234   max 283

so 0.04% of events fall below 192 — roughly 80 in a 200k dataset, enough to crash in
epoch 1. It did.

**Fixed twice over:** `num_particles` lowered to 176, which clears the measured minimum;
and `ROMDataset` now pads short events by resampling with replacement, so a shape
mismatch cannot occur regardless of the config. Verified with 24083 of 30000 events
shorter than requested — every item still collates and a full epoch trains.

Lesson worth keeping: a minimum estimated from hundreds of samples is not a minimum over
hundreds of thousands. Size such thresholds from a percentile of a large sample, or remove
the sensitivity entirely.

### 1.8 Input normalisation stopped the model learning at all

The change I was most confident about — I argued it was the biggest conditioning win and
carried no risk (§5) — is what broke training.

Symptom: the loss was flat to four significant figures across epochs. The decisive test is
whether the model can overfit a single batch of 32; it could not, freezing after ~100 steps
at exactly the predict-zero level. Instrumented, the trunk was alive (`pooled std` healthy)
but the **embedding received ~1e-9 gradient from step 0**, three orders below the solvers.
The input pathway was effectively disconnected and the model settled on a constant output.

Same batch, same target, same optimiser, the original `MLP` fits normally, so the fault was
in `PhysicalMLP`. Ablating its two additions (predict-zero baseline 1.293e-3):

| variant | end loss |
|---|---|
| as shipped — input 20.7–26.7, param 0.05–0.087 | 1.226e-3 stuck |
| `input_scale: 1.0` | 8.0e-4 |
| `param_scale: 1.0` | 8.2e-4 |
| **both 1.0** | **6.4e-4** — matches the original architecture exactly |

My first explanation was that the *per-feature* scales broke the inter-plane geometry the
collinearity signal depends on. Wrong: uniform scales of 10 and 25 fail identically. It is
input magnitude — small inputs and small outputs together starve the gradient.

**Fixed:** both configs set `input_scale: 1.0` and `param_scale: [1,...,1]`.

Lesson: an overfit-one-batch test costs a minute and would have caught this immediately.
It should be run before any long training, and it is now the first thing to try whenever a
loss curve looks flat.

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

### First real training result

10 epochs, 30000 spread-vertex weak-mode-constrained events, single CPU, compressed
schedule — preliminary, but the first time this model has actually trained:

| param | rms epoch 1 | rms epoch 10 | ratio | observability (bins/sigma) |
|---|---|---|---|---|
| dx | 0.02325 | 0.02322 | 1.00 | 0.50 |
| dy | 0.02341 | 0.02341 | 1.00 | 0.07 |
| dz | 0.02404 | 0.02400 | 1.00 | 0.50 |
| nu | 0.04693 | 0.04694 | 1.00 | 0.31 |
| nv | 0.04541 | 0.04530 | 1.00 | 0.31 |
| **rho** | 0.04256 | **0.00689** | **0.16** | **2.81** |

Total loss fell 8.1x, essentially all of it from `rho` — the most observable parameter by a
factor of ~6. The other five have not moved from the predict-zero level.

Whether that is "needs more data and epochs" (30000 events and 10 epochs against the
planned 200000 and 200) or something structural is **unresolved**, and the full run is the
test. But the ordering matches the observability analysis exactly, which is at least
consistent with the weakest parameters simply needing far more data.

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

### `--profile` (time) vs `--shape` (space)

These are independent axes and are easy to confuse.

`--profile` sets how a parameter evolves **over time**; `--shape` sets how it is
distributed **across the planes**. Only the second determines observability.

| profile | what it tests |
|---|---|
| `ramp` | response gain — slope reads off directly whether the full magnitude is recovered |
| `sine` | temporal response — does the sliding window lag or attenuate? |
| `walk` | realism — closest to how a detector actually drifts |
| `static` | steady-state bias at a fixed offset |

Measured blind fraction by profile and shape (mean over the six parameters):

| profile | `uniform` | `observable` |
|---|---|---|
| ramp | 100.0% | 0.0% |
| static | 100.0% | 0.0% |
| sine (no phase step) | 100.0% | 0.0% |
| sine (phase step 2.09) | 25.7% | 0.0% |
| **walk** | **45.5%** | 0.0% |

Note `walk` under `uniform` is 45.5%, not 100%: it draws an independent realisation per
detector, which naturally contains both observable and blind components — roughly half
each, as expected when the blind subspace is 9 of 18 dimensions. So the *original default
run* was diluted, not meaningless. It is the ramp-based `--scan` that was 100% blind and
produced the misleading table in §0.

### Same model, different profile

`rho`, `--shape observable`, narrow checkpoint:

| profile | slope | r² |
|---|---|---|
| ramp | 0.82 | 0.95 |
| walk | 0.87 | 0.79 |
| **sine, period 150** | **0.43** | **0.24** |

The sine result is the informative one. With a 150-step period against a 50-track window,
the model recovers less than half the amplitude. That is the window averaging over a
misalignment which changes appreciably within it — a smoothing/lag effect that `ramp`
cannot see, because a ramp is locally constant over 50 steps.

### The sliding window is a low-pass filter — measured

Sweeping `--period` at fixed `--window 50`, `rho`, `--shape observable`, narrow checkpoint:

| period | period / W | slope | r² |
|---|---|---|---|
| 50 | 1× | −0.000 | 0.000 |
| 100 | 2× | 0.026 | 0.002 |
| 200 | 4× | 0.633 | 0.490 |
| 400 | 8× | 0.872 | 0.790 |
| 800 | 16× | 0.925 | 0.871 |

A boxcar average of width W over a sinusoid of period W integrates to exactly zero, which
is what the first row shows — the model is not failing, the information is not in its
input. Recovery begins around 4W and approaches full gain by 8–16W.

**Operational consequence: the window must be ≲ ¼ of the drift period to track it, and
≲ ⅛ for near-full amplitude.** For real-time alignment this is the binding constraint and
is more decision-relevant than the ramp slope, because it couples W to the physical drift
timescale of the detector. Choosing W is then a bias/variance trade: shorter W tracks
faster drift but averages fewer tracks, so the per-window statistical error grows.

### Metric caveats

**Local R is invariant under in-plane roll**, so σ_before is identically zero for `rho`
and the R-improvement ratio is undefined for it. A 2D local xy RMS metric was added and is
what `scan_comparison.png` uses; local R is still reported alongside with the caveat noted.

Outputs: `param_tracking_det{i}.png`, `param_error_det{i}.png`, `param_correlation.png`,
`raw9_tracking_det{i}.png`, `frame_health.png`, `scorecard.{png,txt,json}`, plus the
original trajectory/residual/summary plots and `scan_comparison.png` under `--scan`.

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
6. **Choosing W against the real drift timescale.** §6 now measures the window's low-pass
   response: full attenuation at period = W, recovery from ~4W. What is *not* known is the
   actual drift timescale of the detector this is meant to run on, which is what sets W.
   The bias/variance trade (shorter W tracks faster drift but averages fewer tracks) is
   also unmeasured — a W scan at fixed period would map it.
7. **Is the deficiency gain rather than visibility?** `nu`, `nv` and `dy` correlate at
   r² ≈ 0.9 but sit at slope ~0.6 with residual improvement ≤ 1.0 (§0) — the correction is
   net harmful for them. Applying a per-parameter gain correction of 1/slope to an existing
   checkpoint's predictions and re-measuring would test, without any retraining, whether
   calibration alone recovers it.

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

#### Re-running: three traps

All three verified.

**`resume: true` resumes silently.** If the checkpoint directory holds a `ckpt_last.pth`,
the run continues from it *even if you have changed the dataset or the loss weights*. Its
only signal is printing `Load model ...` rather than `Train from scratch`. If the saved
epoch already equals `num_epochs`, the loop body never executes and the job exits having
done nothing. After changing anything, delete the directory or set `resume: false`, and
check the first line of output.

**Logs append even when the checkpoint is gone.** Deleting `ckpt_last.pth` but leaving
`train_log.csv` produces a file with epochs `1 2 1 2` and no marker between runs. Delete
the whole directory, not just the checkpoint.

**`checkpoints_*` is not gitignored.** `.gitignore` has `checkpoints` as an exact name, so
`checkpoints/` is ignored but `checkpoints_narrow/` and `checkpoints_spread_vertex/` are
not — which is deliberate for `checkpoints_narrow`, tracked in the repo. New `.pth` files
will show as untracked and a `git add -A` would sweep them in.

Safe by contrast: `make_dataset` refuses a non-empty split with a clear error rather than
mixing generations, and an architecture mismatch fails loudly on `load_state_dict`.

Different configs write to different directories — `config.yaml` → `checkpoints/`,
`config_spread_vertex.yaml` → `checkpoints_spread_vertex/` — so the two do not collide.

Single GPU:

```bash
python train/mlp_physical/train.py \
    --config train/mlp_physical/config_spread_vertex.yaml \
    --device cuda --gpu-id 0
```

**Watch `rms_rho`, `rms_nu`, `rms_nv`**, not the total loss. At init they sit at the width
of the training distribution (the model predicting zero); falling well below that is the
signal that training is working.

### Plot the training curves

```bash
python train/mlp_physical/plot_training.py \
    --checkpoint-dir train/mlp_physical/checkpoints_spread_vertex
```

Writes `training_curves.png` into that directory. Safe to run while training is still
going. Six panels: total loss, the two loss terms, per-parameter RMS error, the same
relative to epoch 1, and the learning rate.

**Read the bottom-middle panel first.** `rms_<param>` starts at the width of the training
distribution, because a model that has learned nothing predicts zero. That panel plots
`rms / rms(epoch 1)`, so a curve sitting at 1.0 means the parameter is not being learned
at all — regardless of what the total loss is doing. Since the six parameters differ in
observability by more than an order of magnitude, the total loss can fall convincingly
while `nu`, `nv` and `dy` never move.

Several runs can be overlaid for comparison:

```bash
python train/mlp_physical/plot_training.py \
    --checkpoint-dir run_a run_b --labels unconstrained constrained \
    --output comparison.png
```

The script warns if a log holds more than one run appended together (epochs resetting
`1 2 1 2`), which happens when a checkpoint is deleted but the log is not.

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

Window-vs-drift-timescale check (§6, open question 6) — vary `--period` at fixed
`--window` and watch the slope fall as the misalignment starts moving within the window:

```bash
for period in 50 100 200 400 800; do
  python train/mlp_no-residual/simulate_sliding_window.py \
      --config train/mlp_no-residual/config_narrow.yaml \
      --steps 800 --window 50 --profile sine --params rho --shape observable \
      --amp-angle 0.25 --period $period \
      --output /tmp/period_$period
done
grep -H "  rho   all" /tmp/period_*/scorecard.txt
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
