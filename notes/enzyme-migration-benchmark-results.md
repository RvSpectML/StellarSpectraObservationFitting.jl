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
| **3. `finalize_scores!` 1st call** | 43.02 s | 59.38 s | 0.72× ⚠️ |
| **3. `finalize_scores!` 2nd call** | 14.70 s | 22.63 s | 0.65× ⚠️ |

try_enzyme timings for benchmark 3 use Optim v2.2.0 (required after upgrading from
v1.11.0, which had a `ManifoldObjective.value_gradient!` bug causing `BoundsError`).

---

## Regression test (finalize_scores!)

Comparing RVs and final loss between Nabla (master) and Enzyme (try_enzyme) on the
same model and data, run 2026-06-15.

| Metric | Value |
|--------|-------|
| RV max \|Δ\| | **0.000327 m/s** ✓ |
| RV median \|Δ\| | 0.000010 m/s |
| RV rms \|Δ\| | 0.000071 m/s |
| loss Nabla  | 718790.8831332659 |
| loss Enzyme | 718790.8830744773 |
| loss Δ (Enzyme − Nabla) | −5.9×10⁻⁵ (Enzyme converges to slightly lower loss) |

Both pass:
- ✓ max RV deviation 0.000327 m/s < 1.0 m/s threshold
- ✓ Enzyme final loss ≤ 1% above Nabla (actually 0.000008% *below*)

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
  1st call :   59.38 s
  2nd call :   22.63 s
```

(Optim v2.2.0; run after upgrading from v1.11.0 which had a ManifoldObjective bug)

---

## FlatLoss result — Julia 1.12.6 — 2026-06-16

After adding `FlatLoss{L,U}` (see `notes/flatloss-plan.md`), the L-BFGS path avoids
passing `Vector{Any}` from `ParameterHandling.Vector_from_vec` through Enzyme.
`set_runtime_activity` is still required (because `spectra_interp` broadcasts a Const
SIH field against Active RVs), but Enzyme performs far fewer runtime checks overall.

**Note: this run used Julia 1.12.6; prior runs used 1.11.2. The Julia version upgrade
may account for part of the speedup — an apples-to-apples comparison would require
running the Nabla baseline on 1.12 as well.**

```
================================================================
Branch : try_enzyme (9dfa822)  [+ uncommitted FlatLoss changes]
Julia  : 1.12.6
Date   : 2026-06-16T18:37:27.955
================================================================

─── 1. ModelWorkspace construction (AD compile) ─────────────────
  1st construction :    0.15 s   (cache warm from prior session eval)
  2nd construction :    0.09 s

─── 2. Adam update! — 10 samples after 2-step warmup ────────────
  median : 221.98 ms
  min    : 143.18 ms
  max    : 230.20 ms
  std    :  27.49 ms

─── 3. finalize_scores! (1st and 2nd call) ──────────────────────
  1st call :   44.23 s
  2nd call :    6.22 s
```

| Benchmark | master (Nabla, 1.11.2) | try_enzyme pre-FlatLoss (1.11.2) | try_enzyme FlatLoss (1.12.6) |
|-----------|------------------------|----------------------------------|------------------------------|
| Adam median | 259.59 ms | 225.99 ms | **221.98 ms** |
| `finalize_scores!` 2nd call | 14.70 s | 22.63 s | **6.22 s** |

---

## Interpretation

**Workspace construction (benchmark 1):** The 4.2× speedup (vs Nabla on 1.11.2) reflects the
fundamental difference in how each AD system is set up. Nabla traces the
computation graph lazily on the first gradient call (bundled with `ModelWorkspace`
construction here), which requires ~25 s of JIT overhead. Enzyme's
`prepare_gradient` compiles the adjoint rule eagerly but much more cheaply
(6 s), and subsequent constructions reuse cached compiled rules (0.16 s vs 0.43 s).

**Adam steady-state step (benchmark 2):** The 15% speedup (vs Nabla) is the recurring per-step
win. Each `update!` call invokes `value_and_gradient!` once; Enzyme is somewhat
faster here because it avoids Nabla's tape overhead.

**finalize_scores! (benchmark 3):** The pre-FlatLoss Enzyme path was **54% slower**
than Nabla (22.6 s vs 14.7 s) because `EnzymeFlatCache` used `set_runtime_activity`
on the `loss ∘ unflatten` composition, inserting runtime activity checks for every
element of `Vector_from_vec`'s `Vector{Any}` output.

FlatLoss differentiates `loss(nested)` where `nested` is typed, so Enzyme never
sees `Vector{Any}`. The result is 6.22 s — a **3.6× improvement over the 22.6 s
pre-FlatLoss Enzyme time**, and **2.4× faster than Nabla's 14.7 s**. The Julia
1.11→1.12 upgrade likely contributes some fraction of the gain; a controlled
comparison is needed to isolate this effect.
