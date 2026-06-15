# Enzyme Migration Benchmark Results

Comparing **Nabla (master)** vs **Enzyme (try_enzyme)** on Julia 1.11.2.
Benchmark script: `benchmark_ad.jl`. Data: `examples/data/{results,data}.jld2`.

Both runs used the same Julia version (1.11.2) and the same benchmark script
(committed on `try_enzyme` as `642969d`, copied to master worktree for the run).

---

## Results

| Benchmark | master (Nabla) | try_enzyme (Enzyme) | Speedup |
|-----------|---------------|---------------------|---------|
| **1. ModelWorkspace 1st construction** | 25.31 s | 6.01 s | **4.2×** |
| **1. ModelWorkspace 2nd construction** | 0.43 s | 0.16 s | **2.7×** |
| **2. Adam `update!` (median, 10 calls)** | 259.59 ms | 225.99 ms | **1.15×** |
| **2. Adam `update!` (min)** | 169.67 ms | 153.02 ms | 1.11× |
| **2. Adam `update!` (max)** | 276.62 ms | 237.55 ms | 1.16× |
| **2. Adam `update!` (std)** | 29.59 ms | 26.27 ms | — |
| **3. `finalize_scores!` 1st call** | 43.02 s | *SKIPPED* | — |
| **3. `finalize_scores!` 2nd call** | 14.70 s | *SKIPPED* | — |

Benchmark 3 on `try_enzyme` is skipped due to a known Optim v1.11.0 bug:
`ManifoldObjective.value_gradient!` returns a scalar instead of `(f, g)`,
causing a `BoundsError` in `LineSearches.make_ϕdϕ`. Fixed in Optim v2.0.1.

---

## Raw output

### master (Nabla) — Julia 1.11.2 — 2026-06-14

```
================================================================
Branch : master (6d97f37)
Julia  : 1.11.2
Date   : 2026-06-14T21:20:10.678
================================================================
Loaded model (OrderModelWobble{Float64}) and data (LSFData{Float64, Matrix{Float64}, Matrix{Float64}})

─── 1. ModelWorkspace construction (AD compile) ─────────────────
  1st construction :   25.31 s
  2nd construction :    0.43 s

─── 2. Adam update! — 10 samples after 2-step warmup ────────────
  median : 259.59 ms
  min    : 169.67 ms
  max    : 276.62 ms
  std    :  29.59 ms

─── 3. finalize_scores! (1st and 2nd call) ──────────────────────
  1st call :   43.02 s
  2nd call :   14.70 s
```

### try_enzyme (Enzyme) — Julia 1.11.2 — 2026-06-14

```
================================================================
Branch : try_enzyme (642969d)
Julia  : 1.11.2
Date   : 2026-06-14T21:22:53.939
================================================================
Loaded model (OrderModelWobble{Float64}) and data (LSFData{Float64, Matrix{Float64}, Matrix{Float64}, SparseMatrixCSC{Float64, Int64}})

─── 1. ModelWorkspace construction (AD compile) ─────────────────
  1st construction :    6.01 s
  2nd construction :    0.16 s

─── 2. Adam update! — 10 samples after 2-step warmup ────────────
  median : 225.99 ms
  min    : 153.02 ms
  max    : 237.55 ms
  std    :  26.27 ms

─── 3. finalize_scores! (1st and 2nd call) ──────────────────────
  SKIPPED — BoundsError: attempt to access Float64 at index [2]
  (Known Optim v1.11.0 ManifoldObjective bug; upgrade to v2+ to enable)
```

---

## Interpretation

**Workspace construction (benchmark 1):** The 4.2× speedup reflects the
fundamental difference in how each AD system is set up. Nabla traces the
computation graph lazily on the first gradient call (bundled with `ModelWorkspace`
construction here), which requires ~25 s of JIT overhead. Enzyme's
`prepare_gradient` compiles the adjoint rule eagerly but much more cheaply
(6 s), and subsequent constructions reuse cached compiled rules (0.16 s vs 0.43 s).

**Adam steady-state step (benchmark 2):** The 15% speedup is the recurring per-step
win. Each `update!` call invokes `value_and_gradient!` once; Enzyme is somewhat
faster here because it avoids Nabla's tape overhead. The noise in both measurements
(std ~26–30 ms on a ~230–260 ms median) suggests runtime variability dominates;
the 15% difference is real but modest.

**finalize_scores! (benchmark 3):** Untestable on `try_enzyme` until the package
requires Optim ≥ 2.0. The master Nabla result (43 s first call, 15 s second) gives
the baseline; the second-call cost reflects L-BFGS converging faster on an already
near-optimal point. To enable this comparison on `try_enzyme`, bump the Optim
compat lower bound in `Project.toml` to `"2"`.
