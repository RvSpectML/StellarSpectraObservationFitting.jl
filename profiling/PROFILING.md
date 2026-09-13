# Profiling `examples/example.jl` (Enzyme backend, `try_enzyme` branch)

Date: 2026-09-13. Julia 1.12.7, 4 threads (`-t 4`), `OPENBLAS_NUM_THREADS=4`,
run on the NEID order-81 simulated dataset (50 observations, `max_n_tel=max_n_star=2`).

Driver script, raw `Profile` dumps (flat + tree, `mincount=5`), and the full
stdout log are in this directory. The driver wraps each pipeline stage in
`@timed_stage` (wall clock, written to `timings.txt` as it goes) and
`Profile.@profile` (statistical sampler, 5ms interval, dumped to
`profile_{flat,tree}_<stage>.txt` right after each stage).

## Two bugs found and fixed along the way

Both were latent API breaks unrelated to the Enzyme migration, just never hit
until a full run got far enough:

1. `src/optimization_functions.jl` `optim_print(x)` read `x.f_x` / `x.g_x` off
   an `Optim.OptimizationState`. Current Optim.jl renamed the field to
   `value` and no longer exposes `g_x` (only a precomputed `g_norm`). This
   crashed `improve_model!` → `finalize_scores!`'s L-BFGS refinement step
   every time, ~1150s into the run. Fixed to `x.value` / `x.g_norm`.
2. `src/optimization_functions.jl` / `src/regularization_functions.jl`: added
   `flush(stdout)` after the progress `println`s so a backgrounded, redirected
   run shows live progress instead of buffering silently until exit (per
   project convention for long Julia background jobs).

**Still open, not fixed here:** `estimate_σ_curvature` (`src/error_estimation.jl:90`)
throws `DomainError` from `sqrt(1 / (2*poly_f.w[3]))` when the local quadratic
fit to the loss curve comes out with negative curvature for some parameter.
This is a numerical/robustness issue in the curvature estimator itself (not
a version-compat break), so I didn't want to silently clamp/`abs()` it without
you weighing in on whether that's an actual sign of a badly-conditioned fit.
It killed the run before the final stage could be profiled; see "Not
profiled" below.

## Wall-clock breakdown

| Stage | Wall time | % of profiled total |
|---|---:|---:|
| `setup_simulate_data` (simulate spectra, LSF, blaze, noise) | 4.9 s | 0.4% |
| `calculate_initial_model` (AIC model-selection search) | 461 s | 37.0% |
| `ModelWorkspace` construction | 0.1 s | ~0% |
| `fit_regularization!` | 667 s | 53.6% |
| `improve_model!` (500 Adam iters + L-BFGS score refinement) | 112 s | 9.0% |
| `estimate_σ_curvature` | **crashed**, not measured | — |
| **Total (measured)** | **~1245 s (~20.8 min)** | |

`fit_regularization!` and `calculate_initial_model` dominate — together over
90% of the measured wall time — and `improve_model!`, the part most people
would guess is "the fit," is actually the smallest completed stage.

## What the sampler shows in the hot stages

Excluding thread-idle samples (`Base/task.jl:1216 poptask`, which is huge in
every stage because 3 of the 4 requested threads sit idle — see below), the
self-time leaders are nearly identical in `calculate_initial_model`,
`fit_regularization!`, and `improve_model!`:

1. **Enzyme reverse-mode AD itself** — `Enzyme/src/compiler.jl:7003` (the
   generated combined forward+reverse call) is the single largest
   non-Base/non-JIT self-time consumer in every stage (2440 / 3995 / 869
   samples respectively). This is `value_and_gradient!` → `autodiff` →
   `enzyme_call`, i.e. the actual gradient evaluation of `l_total` /
   `l_frozen_tel`. This is inherent cost of the model, not overhead.
2. **Indexing/allocation overhead** — `Base/essentials.jl:920 getindex` and
   `Base/boot.jl:588 GenericMemory` are the #2 and #3 self-time consumers
   everywhere (21.7k / 41.0k / 7.3k and 14.8k / 22.0k / 4.4k samples). These
   are almost certainly allocations happening *inside* the Enzyme-generated
   derivative code and/or the loss-function closures (shadow buffers,
   `make_zero!`, intermediate arrays) rather than in your own hot loops —
   the sheer volume relative to actual arithmetic ops (`+`, `*`, `exp`,
   `gemm!`, each in the hundreds-to-low-thousands) suggests allocation
   pressure, not compute, is competing with the AD for time.
3. **Dense linear algebra** (`LinearAlgebra/src/blas.jl:1642 gemm!`) is
   present but small (≈700–1700 samples per stage) — the models here are not
   BLAS-bound, so throwing more BLAS threads at this will not help much.
4. **JIT/compilation overhead is concentrated in `calculate_initial_model`
   specifically** — `Compiler/…/typeinf`, `call_get_staged`,
   `EnzymeCreateAugmentedPrimal`, `EnzymeCreatePrimalAndGradient`, `check_ir!`
   all show up with real self-time (918, 1553, 1984, 415, 574 samples) only
   in this stage's profile, not in `fit_regularization!` or `improve_model!`.
   That's consistent with `calculate_initial_model` repeatedly building *new*
   `OrderModel`/`TotalWorkspace`/`FrozenTelWorkspace` objects (one per
   candidate `n_comp`) and therefore forcing Enzyme to re-differentiate and
   re-JIT a fresh loss closure for every candidate it tries during the AIC
   search, instead of reusing one compiled derivative.

## Threads are mostly idle

`Base/task.jl:1216 poptask` (a worker thread parked waiting for work) accounts
for the majority of *total* samples in every stage (280k/402k/69k out of the
totals) even though we asked for 4 threads. `calculate_initial_model`,
`fit_regularization!`, and `improve_model!` are effectively single-threaded
in their hot path — the only place `Threads.@threads` is actually used is
inside `estimate_σ_curvature_helper` (`src/error_estimation.jl:28`), the one
stage that never got to run to completion. So the 4 cores requested for this
run bought essentially nothing for 90%+ of the wall time.

## Not profiled

`estimate_σ_curvature` crashed almost immediately after starting (DomainError
above), so there's no profile for it. Structurally it's an embarrassingly
parallel per-parameter loop (already has a `Threads.@threads` path) evaluating
`ℓ(x)` `n=7` times per parameter around the best-fit value — its cost scales
with the number of free parameters (RV per obs + all time-varying scores),
which for this dataset is probably a few hundred, each needing `n=7` extra
forward evaluations (or 7 gradient evaluations if `use_gradient=true`). Worth
re-profiling once the DomainError is addressed.

## Implications for future optimization / parallelization / GPU work

1. **Biggest win is probably avoiding repeated Enzyme (re)compilation across
   the AIC model-selection search in `calculate_initial_model`.** Every
   candidate `n_comp` currently builds a distinct `OrderModel`/workspace and
   (it appears) triggers fresh differentiation. If the loss closures can be
   restructured so the same compiled derivative is reused across candidates
   (e.g. fixed-shape buffers sized to `max_n_tel`/`max_n_star` with masking,
   rather than genuinely different array shapes per candidate), the JIT
   overhead specific to this stage should mostly disappear.
2. **The independent evaluations inside the search loops are real
   parallelization opportunities that aren't being used today:**
   - `calculate_initial_model`'s "try adding a telluric component" vs. "try
     adding a stellar component" branches (`src/optimization_functions.jl`
     ~1811 and ~1831) are independent and currently run sequentially.
   - `fit_regularization!`'s per-`reg_key` grid search
     (`eval_regularization` calls in `regularization_functions.jl`) are also
     independent evaluations run one at a time.
   Both are natural `Threads.@threads` / `Threads.@spawn` targets, and would
   actually make use of the cores this run requested but never touched.
3. **Reduce allocation inside the AD path before reaching for a GPU.** The
   `getindex`/`GenericMemory` self-time is large enough, relative to actual
   arithmetic, that it's worth checking `src/ad_backend.jl`'s
   `prepare_gradient`/`value_and_gradient!`/`_enzyme_copy_nested!` and the
   loss closures for buffers that get reallocated every call instead of
   reused across iterations (Enzyme's `make_zero!` shadow allocation shows up
   directly in the profile). This is cheaper to fix than a GPU port and will
   make any later GPU port more effective too (fewer host-side allocations
   fighting the device work).
4. **GPU is a plausible follow-on for `improve_model!`'s 500-iteration Adam
   loop specifically** (the stage that's actually a tight, repeated,
   AD-through-linear-algebra loop over the full order's flux/variance
   arrays), but the current profile doesn't show it as BLAS-bound, so a GPU
   port would mainly be trading kernel-launch/host-transfer overhead against
   Enzyme's CPU AD overhead — likely only worth it once (1) and (3) above are
   done and the arrays involved are confirmed large enough (order length ×
   n_obs, here 9232×50) to amortize device transfer cost. I'd profile again
   post-fix before committing to a GPU rewrite.
5. **`fit_regularization!` at 667s (54% of wall time) is worth a second,
   more detailed profiling pass on its own** — the flat profile shows it
   funnels through the same `train_SubModel!`/`update!`/`l_total` Adam path
   as `improve_model!`, just called many more times (once per regularization
   key × per bisection step × train/test split), so (1)+(2) above should
   help it proportionally the most of any stage.

## Files in this directory

- `timings.txt` — per-stage wall-clock seconds (CSV).
- `run.log` — full stdout/stderr of the profiled run.
- `profile_flat_<stage>.txt` / `profile_tree_<stage>.txt` — raw
  `Profile.print` dumps (`format=:flat`/`:tree`, `mincount=5`) for
  `calculate_initial_model`, `fit_regularization`, and `improve_model`.
