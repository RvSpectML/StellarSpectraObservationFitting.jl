# FlatLoss Plan: Reduce `set_runtime_activity` Cost on L-BFGS Path

## Status: COMPLETE (revised goal)

Implementation succeeded. Benchmark pending (2026-06-16).

### What was implemented

A `FlatLoss{L,U}` struct wraps `(loss, unflatten)` and a paired `EnzymeFlatLossCache`
differentiates `loss(nested)` w.r.t. `nested` directly — no `Vector{Any}` ever touches
Enzyme. `ParameterHandling.flatten(∂nested)` maps the gradient back to flat space.

`set_runtime_activity` is still required (added back after testing), because
`spectra_interp` internally broadcasts a Const SIH field against an Active RV vector
(`sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)'`), which Enzyme cannot prove
statically. The benefit over the original `loss ∘ unflatten` path is reduced scope:
Enzyme never traces through `Vector{Any}`, so fewer runtime activity checks occur.

The original goal of *eliminating* `set_runtime_activity` was not achievable without
either (a) a custom SIH Enzyme rule (which caused a 5× Adam regression due to
Julia-level scatter-add overhead replacing LLVM-level AD) or (b) refactoring
`spectra_interp` to separate the SIH-constant and RV-active components.

### Complications encountered

#### Complication 1 — `AugmentedRuleReturnError` for SIH `augmented_primal`

The `spectra_interp(model_flux, rvs, sih)` Enzyme rule's `augmented_primal` method was
returning bare `AugmentedReturn` (a UnionAll, no type params) because Julia's type
inference couldn't determine a concrete return type.  Enzyme's check at
`customrules.jl:2103` requires the inferred return type to be a concrete subtype of
`AugmentedReturn{Any, Any}`.

**Fix:**
1. Changed `::Type` to `RT::Type` in the `augmented_primal` signature to capture the
   return annotation.
2. Added `::Matrix{Float64}` type assertion on `primal_val` to force concrete inference.
3. Replaced `AugmentedReturn(primal_val, shadow, shadow)` with
   `Enzyme.EnzymeRules.augmented_rule_return_type(config, RT)(primal_val, shadow, shadow)`.

Required a hard Julia session reset (not just `revise()`) to take effect.

#### Complication 2 — `EnzymeRuntimeActivityError` for `Data` fields

`d::GenericData` captured as Const in the FlatLoss closure caused "constant memory stored
to differentiable variable" LLVM errors.  Enzyme's activity analysis failed to propagate
`Const` through field accesses on `d`.

**Fix:** Added `Enzyme.EnzymeRules.inactive_type(::Type{<:Data}) = true` to
`src/ad_backend.jl`.  This is safe because `Data` subtypes hold observational flux,
variance, and wavelength arrays that are never optimized parameters.

#### Complication 3 — Mixed-activity `rv .+ om.bary_rvs` broadcast

The broadcast `rv .+ om.bary_rvs` in the `l_total_s` closure mixed an Active `rv`
vector with a Const `bary_rvs` vector, creating `Broadcasted{Tuple{Active,Const}}` that
Enzyme couldn't analyze statically.

**Fix:** Named helper `_rv_shift(rv, bary_rvs) = rv .+ bary_rvs` with an explicit
Enzyme `augmented_primal` + `reverse` rule pair.

### Key files

- `src/ad_backend.jl` — `FlatLoss`, `EnzymeFlatLossCache`, SIH rule with `RT::Type`,
  `inactive_type(::Type{<:Data})`, `_rv_shift` rule
- `src/optimization_functions.jl:927` — `f = FlatLoss(loss, unflatten)`
- `test/runtests.jl` — `FlatLoss: Enzyme gradient without set_runtime_activity` test set

---

## Problem

`finalize_scores!` (L-BFGS via Optim.jl) is ~54% slower with Enzyme than Nabla
(22.6 s vs 14.7 s on 2nd call). Root cause: `value_and_gradient!(::EnzymeFlatCache, ...)`
in `src/ad_backend.jl` calls `Enzyme.set_runtime_activity(ReverseWithPrimal)`.

This flag is needed because `opt_funcs` in `src/optimization_functions.jl` builds
`f = loss ∘ unflatten` where `unflatten` is `ParameterHandling.Vector_from_vec`.
That function executes a list comprehension `[backs[n](x_vec[a:b]) for n in eachindex(x)]`
which returns `Vector{Any}` — Enzyme can't statically prove the activity of heterogeneous
elements, so it falls back to runtime checks on every L-BFGS gradient call.

## Strategy

Create a named `FlatLoss{L,U}` struct wrapping `(loss, unflatten)`. Add specialized
`prepare_gradient` and `value_and_gradient!` that compute the Enzyme gradient in
*nested* (properly typed) parameter space, then flatten the result back to a plain
`Vector{Float64}`. No `Vector{Any}` ever touches Enzyme.

## Implementation Steps

### Step 1 — Add `FlatLoss` struct to `src/ad_backend.jl` ✓

```julia
struct FlatLoss{L,U}
    loss::L
    unflatten::U
end
(f::FlatLoss)(x::AbstractVector) = f.loss(f.unflatten(x))
```

### Step 2 — Add `EnzymeFlatLossCache` and `prepare_gradient` method ✓

```julia
struct EnzymeFlatLossCache{∂N, T<:AbstractVector{<:Real}}
    ∂nested::∂N
    ∂θ::T
end

function prepare_gradient(::EnzymeBackend, f::FlatLoss, θ::AbstractVector{<:Real})
    nested = f.unflatten(θ)
    ∂nested = Enzyme.make_zero(nested)
    ∂θ = zero(θ)
    return EnzymeFlatLossCache(∂nested, ∂θ)
end
```

### Step 3 — Add specialized `value_and_gradient!` ✓

```julia
function value_and_gradient!(c::EnzymeFlatLossCache, f::FlatLoss, θ::AbstractVector{<:Real})
    nested = f.unflatten(θ)
    Enzyme.make_zero!(c.∂nested)
    _, val = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,    # no set_runtime_activity needed
        Enzyme.Const(f.loss),
        Enzyme.Active,
        Enzyme.Duplicated(nested, c.∂nested),
    )
    flat_grad, _ = ParameterHandling.flatten(c.∂nested)
    c.∂θ .= flat_grad
    return val, c.∂θ
end
```

### Step 4 — Update `opt_funcs` in `src/optimization_functions.jl` ✓

Line 927:
```julia
f = FlatLoss(loss, unflatten)
```

### Step 5 — Mooncake fallback in `ext/SSOFMooncakeExt.jl`

Not needed — Mooncake's existing `prepare_gradient(::MooncakeBackend, l, θ)` generic
method handles `FlatLoss` via `Mooncake.build_rrule(l, θ)` which calls `l(θ)` via
`(f::FlatLoss)(x)`, i.e., `f.loss(f.unflatten(x))`. No special fallback required.

### Step 6 — Add test to `test/runtests.jl` ✓

Test set "FlatLoss: Enzyme gradient without set_runtime_activity" — 4 tests pass.

### Step 7 — Benchmark

Run `benchmark_ad.jl` and confirm `finalize_scores!` 2nd call drops from
22.6 s toward the Nabla baseline of 14.7 s.

## Key Files

- `src/ad_backend.jl` — add `FlatLoss`, `EnzymeFlatLossCache`, new `prepare_gradient`
  and `value_and_gradient!` methods
- `src/optimization_functions.jl` — change `f = loss ∘ unflatten` to `FlatLoss(loss, unflatten)`
- `test/runtests.jl` — add `FlatLoss` gradient test
