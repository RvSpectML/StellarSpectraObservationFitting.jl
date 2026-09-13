# Handoff: acting on PROFILING.md Implications #1 and #3

Branch: `try_enzyme`. Written 2026-09-13. Companion to `profiling/PROFILING.md`.

`profiling/` is currently untracked (`git status` shows `?? profiling/`). **`git add profiling/`
before handing this off**, or the collaborator can't see it.

> **Julia version — settle this first.** `PROFILING.md`'s numbers were taken on **1.12.7**;
> `CLAUDE.md` specifies **1.11.2** for this repo. Pick one, record it here, and use it for
> every run below. Enzyme's runtime-generic path is exactly the kind of thing that shifts
> across a Julia minor version, so **none of PROFILING.md's absolute sample counts may be used
> as a comparison baseline unless you stay on 1.12.7.** Step 0 re-runs the unmodified
> `2fa1302` baseline on the chosen version for this reason.
>
> Version chosen for this work: `1.12.7`

---

## 0. Revised diagnosis — read this before doing anything in PROFILING.md §"Implications"

PROFILING.md treats Implication #1 (Enzyme recompilation across the AIC search) and
Implication #3 (allocation inside the AD path) as two separate problems. Re-reading the
flat profiles shows they are almost certainly **one problem**, and PROFILING.md named the
symptom rather than the cause.

Grep the flat profiles for Enzyme's runtime-dispatch fallback:

```bash
grep -E "runtime_generic|compiler.jl 7003" profiling/profile_flat_*.txt
```

| Stage | samples under `enzyme_call` | of which `runtime_generic_augfwd` + `runtime_generic_rev` |
|---|---:|---:|
| `improve_model` | 16621 | 7299 + 9293 = **16592 (99.8%)** |
| `fit_regularization` | 90741 | 37963 + 51339 = **89302 (98.4%)** |
| `calculate_initial_model` | 55737 | 21983 + 25492 = **47475 (85.2%)** |

`runtime_generic_augfwd`/`runtime_generic_rev` (`Enzyme/src/rules/jitrules.jl:600` and `:824`)
is Enzyme's fallback for calls it **could not resolve to a single method at compile time**.
Every such call is differentiated by boxing arguments, dispatching dynamically, and running
Julia's type inference at runtime. So:

- **PROFILING.md #3 is explained by this.** `Base/essentials.jl:920 getindex` (21.7k / 41.0k /
  7.3k self-samples) and `Base/boot.jl:588 GenericMemory` (14.8k / 22.0k / 4.4k) are the boxing
  and shadow allocation *of the runtime-generic path itself*. They are not leaked buffers in
  `prepare_gradient`, `value_and_gradient!`, or `_enzyme_copy_nested!` — those three functions
  total <40 self-samples in every stage. (Grepping `ad_backend` also turns up `:289
  augmented_primal` at 4818 and `:303 reverse` at 4735 in `fit_regularization` — those are the
  custom `spectra_interp` EnzymeRules, i.e. real derivative work, not buffer churn.) Chasing
  buffer reuse in `src/ad_backend.jl` is the wrong target.
- **PROFILING.md #1 is plausibly explained by this too.** `runtime_generic_*` discovers each new
  callee signature lazily and calls `typeinf` / `EnzymeCreateAugmentedPrimal` at runtime. That
  is exactly why those frames show real self-time *only* in `calculate_initial_model` (the first
  stage to run, where every signature is seen for the first time) and not in the later stages.
  It does **not** require the per-`n_comp` re-differentiation story.

### Why the loss closures aren't statically resolvable

Abstract field types. Every `om.tel.lm.M` inside a loss body is a dynamic field access:

| Location | Field | Declared as | Problem |
|---|---|---|---|
| `src/model_functions.jl:513` | `Submodel.lm` | `LinearModel` | abstract supertype |
| `src/model_functions.jl:515` | `Submodel.A_sde` | `StaticMatrix` | abstract |
| `src/model_functions.jl:517` | `Submodel.Σ_sde` | `StaticMatrix` | abstract |
| `src/model_functions.jl:692,694` | `OrderModelWobble.tel`, `.star` | `Submodel` | unparameterized UnionAll |
| `src/model_functions.jl:696` | `OrderModelWobble.rv` | `AbstractVector` | abstract |
| `src/model_functions.jl:704` | `OrderModelWobble.bary_rvs` | `AbstractVector{<:Real}` | abstract |
| `src/model_functions.jl:707` | `OrderModelWobble.t2o` | `AbstractVector{<:SparseMatrixCSC}` | abstract |
| `src/model_functions.jl:666,668,670` | `OrderModelDPCA.tel/.star/.rv` | `Submodel` | unparameterized |
| `src/model_functions.jl:676,678` | `OrderModelDPCA.b2o`, `.t2o` | `AbstractVector{<:SparseMatrixCSC}` | abstract |

`Submodel` is additionally `mutable`, so Julia can't even constant-fold through it.

Note commit `2fa1302` ("Perf: concrete field types for SIH, AdamSubWorkspace,
OptimSubWorkspace") already started this work. `Submodel` and the two `OrderModel` types are
the remaining pieces — and they are the ones the loss bodies actually touch.

Secondary suspect, to be checked only if Step 1 doesn't fully clear it:
`model_prior(lm, reg::Dict, sm::Submodel)` (`src/model_functions.jl:1343`) takes `sm` as an
unparameterized `Submodel` and reaches `sm.Δℓ_coeff` / `sm.A_sde` / `sm.Σ_sde`. The
`:: Float64` return annotation hides the instability from `@code_warntype` at the call site
while leaving the dynamic dispatch (and the box) in place for Enzyme.

### Consequence for the plan

**Do not start with PROFILING.md #1's prescription** (fixed-shape buffers sized to
`max_n_tel`/`max_n_star` with masking). That is an invasive rewrite of `downsize`,
`downsize_view`, `fill_OrderModel!`, and the EMPCA initialization, and if the first-stage JIT
cost is runtime-generic type inference rather than genuine per-candidate re-differentiation,
**the masking redesign buys nothing**. The current profile cannot distinguish the two. Step 3
below is an explicit gate that decides it with a measurement.

---

## Step 0 — Commit a reproducible driver (prerequisite, ~1h)

`profiling/PROFILING.md:6` claims the driver script is in this directory. **It is not.** Every
success criterion below is "re-profile and compare", so this has to exist first.

Write `profiling/profile_example.jl`:

- Same configuration as the original run: NEID order-81 simulated dataset, 50 observations,
  `max_n_tel = max_n_star = 2`, the Julia version recorded at the top of this file, `-t 4`,
  `OPENBLAS_NUM_THREADS=4`.
- `@timed_stage` wrapper appending to `timings.txt`, `Profile.@profile` per stage with
  `mincount=5` flat + tree dumps, `flush(stdout)` after each progress `println`.
- Take an output-directory argument so before/after runs don't overwrite each other.

Also add `profiling/bench_gradient.jl` — the micro-benchmark used by Steps 1–3:

```julia
# Build the smallest realistic workspace, then measure ONE gradient call.
using BenchmarkTools, Profile
import StellarSpectraObservationFitting as SSOF
# ... construct om, d, mws = SSOF.TotalWorkspace(om, d) ...
aws = mws.total
@btime SSOF.value_and_gradient!($aws.cache, $aws.l, $aws.θ)
println("allocs/call: ", @allocated SSOF.value_and_gradient!(aws.cache, aws.l, aws.θ))
# 200 Adam iterations, profiled, to check for runtime_generic_* frames
Profile.clear(); Profile.@profile for _ in 1:200; SSOF.update!(aws); end
Profile.print(; format=:flat, mincount=5)
```

**Record the baseline on the current `try_enzyme` HEAD (`2fa1302`), on the chosen Julia
version, before changing anything** — both the `bench_gradient.jl` numbers and a full
`profile_example.jl` run. Write the full-run baseline into this file:

Recorded on `try_enzyme` @ `8776cf0` (prep commit `98d38f2` + RNG-seed fix `8776cf0`, on top of
`2fa1302`; see note below on why the prep commit had to move ahead of `2fa1302`), Julia 1.12.7,
`-t 4`, `OPENBLAS_NUM_THREADS=4`, `profiling/profile_example.jl profiling/baseline_seeded` and
`profiling/bench_gradient.jl` (`profiling/bench_gradient_baseline.log`). Data simulation is
seeded (`seed=20260913` in `profiling/_setup.jl`) — an unseeded first attempt
(`profiling/baseline_98d38f2`, discarded) showed `calculate_initial_model`'s runtime_generic
share can otherwise vary run-to-run because the AIC search path itself varies; the seeded rerun
reproduced the same share to within 0.1pp, so the seed controls that confound adequately for
this comparison.

| Baseline (`8776cf0`, Julia 1.12.7) | value |
|---|---|
| `calculate_initial_model` wall | 464.5 s |
| `fit_regularization` wall | 623.6 s |
| `improve_model` wall | 98.8 s |
| `typeinf` self-samples (Overhead col, summed), `calculate_initial_model` | 3512 |
| `call_get_staged` self-samples, `calculate_initial_model` | 5876 |
| `EnzymeCreateAugmentedPrimal` self-samples, `calculate_initial_model` | 5748 |
| `EnzymeCreatePrimalAndGradient` self-samples, `calculate_initial_model` | 321 |
| `check_ir!` self-samples, `calculate_initial_model` | 1839 |
| `@allocated` per `value_and_gradient!` | 169,059,728 bytes (~161 MiB) |
| `@btime` per `value_and_gradient!` | median 98.6 ms (range 92.3–218.5 ms), 158.88 MiB / 764 allocs |
| `runtime_generic_{augfwd,rev}` / `enzyme_call`, `calculate_initial_model` | 3061/19617 = 15.6% |
| `runtime_generic_{augfwd,rev}` / `enzyme_call`, `fit_regularization` | 46371/49824 = 93.1% |
| `runtime_generic_{augfwd,rev}` / `enzyme_call`, `improve_model` | 47244/47356 = 99.8% |
| `runtime_generic_{augfwd,rev}` / `enzyme_call`, 200-iter Adam (`bench_gradient.jl`) | 16414/16456 = 99.7% |

Note on `calculate_initial_model`'s 15.6% (vs. PROFILING.md's original 85.2%): the prep commit
(`98d38f2`) includes a previously-uncommitted `loss_funcs_frozen_tel(o, om::OrderModelWobble, d)`
closure-specialization override (mirroring the already-committed `loss_funcs_total` Wobble
override from `3513464`) that was **not** present when PROFILING.md's numbers were taken. Since
`calculate_initial_model`'s AIC search evaluates several `n_tel=0` (no-tellurics ⇒
`FrozenTelWorkspace`) candidates, that fix alone already closed most of Implication #1 for this
stage before Step 1 started. `fit_regularization`/`improve_model` use `TotalWorkspace`
(tellurics present in the winning model) and were unaffected by that fix — they are the stages
Step 1 targets.

Step 3's gate compares against **this table**, not against PROFILING.md's.

---

## Step 1 — Concretize `Submodel` and the `OrderModel` types (the main fix, 1–2 days)

Do this in a worktree or a scratch branch; it is a wide but mechanical change.

### 1a. `Submodel` (`src/model_functions.jl:507–520`)

Add type parameters for `lm`, `A_sde`, `Σ_sde`:

```julia
mutable struct Submodel{T<:Number, AV1<:AbstractVector{T}, AV2<:AbstractVector{T},
                        AA<:AbstractArray{T}, LM<:LinearModel,
                        SM1<:StaticMatrix, SM2<:StaticMatrix}
	log_λ::AV1
	λ::AV2
	lm::LM
	A_sde::SM1
	Σ_sde::SM2
	Δℓ_coeff::AA
end
```

Keep `mutable` only if something actually reassigns a field — check with
`grep -n "\.lm = \|\.A_sde = \|\.Δℓ_coeff = " src/`. If nothing does, make it `struct`; that
alone helps Enzyme.

**Fix the latent bug at `src/model_functions.jl:554` while you're here:** the outer constructor
calls `Submodel{T, AV1, AV2}(...)` — only three of the four current type parameters. It works
today (Julia infers the rest from the UnionAll) but it silently defeats the point of
parameterizing. After this change it must name every parameter, or just call `new`/the default
constructor and let inference fill them in.

### 1b. `OrderModelWobble` (`src/model_functions.jl:690–711`)

```julia
struct OrderModelWobble{T<:Number, STel<:Submodel, SStar<:Submodel,
                        RV<:AbstractVector, BRV<:AbstractVector{<:Real},
                        T2O<:AbstractVector{<:SparseMatrixCSC}} <: OrderModel
	tel::STel
	star::SStar
	rv::RV
	reg_tel::Dict{Symbol, T}
	reg_star::Dict{Symbol, T}
	b2o::StellarInterpolationHelper   # already concrete after 2fa1302 — verify
	bary_rvs::BRV
	t2o::T2O
	metadata::Dict{Symbol, Any}
	n::Int
end
```

`rv` must stay a type parameter, not `Vector{Float64}`: `src/model_functions.jl:784` builds a
`view(om.rv, inds)` for the train/test split in `fit_regularization!`, so both `Vector` and
`SubArray` instantiations are needed. Same for `b2o`/`t2o` if any code path views them.

### 1c. `OrderModelDPCA` (`src/model_functions.jl:664–683`)

Same treatment: `tel`/`star`/`rv` parameterized on `Submodel`, `b2o`/`t2o` on the sparse vector
type. Lower priority — `calculate_initial_model` has a `# TODO: Make this work for
OrderModelDPCA` and the profiled path is Wobble — but leaving it abstract leaves the DPCA loss
bodies in the runtime-generic path, so do it in the same pass.

### 1d. Sweep the call sites

```bash
grep -rn "::Submodel\b\|::OrderModelWobble\b\|::OrderModelDPCA\b" src/ test/
```

Signatures written as `::Submodel` still dispatch fine on the parameterized UnionAll — nothing
breaks — but any that *construct* with explicit parameters need updating. The constructors at
`src/model_functions.jl:545, 547, 557, 558, 775, 782, 784, 922, 929, 954, 961` are the list.

### 1e. Check `model_prior`

Change `model_prior(lm, reg::Dict, sm::Submodel)` (`src/model_functions.jl:1343`) to
`reg::Dict{Symbol,<:Real}` and drop the `:: Float64` return annotation once inference gives
`Float64` on its own. If it still doesn't infer, that annotation is masking a real problem —
find it, don't re-add the annotation.

### Pass/fail for Step 1

Run `profiling/bench_gradient.jl` before and after.

| Metric | Threshold |
|---|---|
| `@allocated` per `value_and_gradient!` call | **≥10× reduction** |
| `runtime_generic_augfwd` / `runtime_generic_rev` in the 200-iteration Adam profile | **absent, or <5% of `enzyme_call` samples** |
| `@btime` per gradient call | expect ≥3×; anything <1.5× means the instability is elsewhere |
| `calculate_initial_model` stage wall time | see note below — **a regression here is tolerable** |
| `fit_regularization` + `improve_model` stage wall time | must improve |
| `Pkg.test()` | passes, and RVs/uncertainties match the pre-change run to round-off |

**Expect `calculate_initial_model` to behave differently from the other two, and do not read
that as failure.** Parameterizing `Submodel` on `LM` means `downsize(om, n_tel, n_star)` now
yields a *different concrete* `Submodel` — and hence a different concrete `OrderModelWobble` —
whenever it swaps `FullLinearModel` ↔ `TemplateModel` ↔ `BaseLinearModel`. This stage builds
one model per AIC candidate, so it absorbs the added type multiplicity hardest, and it is 37%
of wall time. Per-call cost drops while first-call compile cost per candidate rises; the net
for this one stage could go either way. **If `fit_regularization` and `improve_model` both
improve and `calculate_initial_model` regresses, Step 1 succeeded — do not revert.** Carry the
regression into Step 3, where it is precisely the Branch B signal.

**If allocations do not drop:** stop and bisect by annotation rather than proceeding. Add
`Enzyme.API.runtimeActivity!` diagnostics, or instrument with
`Enzyme.Compiler.enzyme_code_llvm` on the loss and grep the IR for `jl_apply_generic` to find
which call is still unresolved. `model_prior` and `gp_ℓ_precalc` are the first places to look.

### Cost to watch

Parameterizing on `lm` multiplies distinct concrete `Submodel` types: `{Full, Base, Template} ×
{owned, view}`. That is a bounded one-time compile cost traded against a per-call tax, and the
per-call tax is currently ~98% of AD time — but **measure compile time alongside run time**,
don't assume. `n_comp` changes array *sizes*, not types, so it adds no specializations.

---

## Step 2 — Re-run the full profile (½ day)

Run `profiling/profile_example.jl` into a fresh output directory. Produce the same wall-clock
table as `PROFILING.md` §"Wall-clock breakdown" and the same runtime-generic grep table as §0
above. Commit both alongside the originals.

**This is the measurement that PROFILING.md #3 asked for, and it is now done.** #3's specific
prescription (reuse buffers in `src/ad_backend.jl`) turned out to be aimed at a symptom; if
Step 1 hits its thresholds, #3 is closed. Note that explicitly in the updated PROFILING.md so
nobody re-opens it.

---

## Step 3 — GATE: does Implication #1 still need anything? (decide, don't assume)

Look at the new `profile_flat_calculate_initial_model.txt`:

```bash
grep -E "typeinf|call_get_staged|EnzymeCreateAugmentedPrimal|EnzymeCreatePrimalAndGradient|check_ir" \
  profiling/<new>/profile_flat_calculate_initial_model.txt
```

Compare against the **Step 0 baseline table**, not against PROFILING.md's counts (918 / 1553 /
1984 / 415 / 574) — those were taken on Julia 1.12.7 and are only valid if you stayed on that
version. Also compare `calculate_initial_model` stage wall time against the Step 0 baseline;
per Step 1's note, a regression there is the expected place for added type multiplicity to show
up, and it is itself a Branch B indicator.

**Branch A — compilation frames are gone or small (<5% of the stage).** Implication #1 is
resolved as a side effect of Step 1. **Do not build the masking redesign.** Close #1 with the
measurement as evidence. Next target becomes whatever now dominates — probably
`fit_regularization!` (PROFILING.md #5), where the natural follow-on is #2's parallelization of
the independent `eval_regularization` grid evaluations. Note that `value_and_gradient!` is
documented as **not thread-safe** (`src/ad_backend.jl:28`), so each parallel task needs its own
`prepare_gradient` cache — that is the actual design constraint for #2, and it is much cheaper
to satisfy once the caches are cheap to build.

**Branch B — compilation frames persist at meaningful self-time.** There is genuine
per-candidate re-differentiation, and #1's masking prescription is on the table. Before
committing to it, confirm the mechanism cheaply: instrument `TotalWorkspace`/`FrozenTelWorkspace`
construction (`src/optimization_functions.jl:787, 848`) to log `typeof(loss)`,
`typeof(build_θ(om))`, and `typeof(om)` per candidate `n_comp`, and run with
`max_n_tel = max_n_star = 2`. After Step 1, `typeof(om)` is the direct readout of the added
type multiplicity flagged in Step 1's pass/fail note. If the
tuple *types* differ across candidates (e.g. 2-tuple vs 3-tuple from the
`is_time_variable`-dependent `build_θ` at `src/optimization_functions.jl:797–812` and
`:855–876`, or `Template` vs `Full` vs `Base` linear models from `downsize`), that is the
re-specialization — and the cheap fix is to normalize the *closure* set, not to rewrite array
shapes. Only if the types are identical and Enzyme still recompiles is the fixed-shape/masking
rewrite justified.

Write the branch taken and the numbers behind it into `PROFILING.md` either way.

---

## Out of scope, but blocking one thing

`estimate_σ_curvature` (`src/error_estimation.jl:90`) still throws `DomainError` from
`sqrt(1 / (2*poly_f.w[3]))` on negative local curvature. It is unrelated to #1/#3, but it means
that stage can't be profiled — so it stays unmeasured after Step 2 as well. Eric needs to
decide whether negative curvature is a genuine ill-conditioning signal before anyone clamps or
`abs()`es it.

---

## Summary of sequencing

```
Step 0  driver script + baseline          (prerequisite)
Step 1  concretize Submodel/OrderModel*   (the fix — addresses #1 and #3 together)
Step 2  re-profile                        (the evidence)
Step 3  GATE on #1                        (A: close it.  B: then and only then, masking)
```

Steps 1–2 are ~2–3 days. The masking rewrite that PROFILING.md #1 proposes is weeks, and Step 3
Branch A likely makes it unnecessary.
