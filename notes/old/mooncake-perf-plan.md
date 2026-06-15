# Mooncake performance regression plan

Baseline: Nabla Adam step median 286 ms. Mooncake: 4688 ms (16×). Target: within ~2× of
Nabla, or document why the remaining gap is acceptable.

Work the three causes in order — each has a "stop if fixed" gate before the next.

---

## Cause 1 — `l::Function` type instability (highest priority)

### What's wrong

`AdamSubWorkspace` declares its loss field as the abstract type `Function`:

```julia
# src/optimization_functions.jl:386
struct AdamSubWorkspace{T,C}
    l::Function   # ← abstract
```

When Julia compiles `update!`, `aws.l` is inferred as `Function`, not the concrete closure
type. Every call through `aws.l` inside `update!` goes through dynamic dispatch. More
critically, `value_and_gradient!(aws.cache, aws.l, aws.θ)` passes `l::Function` into
Mooncake's `value_and_gradient!!`. Mooncake compiled its rule for the concrete closure type
(at `build_rrule` time), but the caller now presents `l::Function`; this mismatch likely
forces interpreted fallback on every gradient call.

Nabla used a different tracing mechanism that is less sensitive to this, which is why the
regression didn't exist before.

### Diagnosis step

```julia
aws = mws.total
@code_warntype SSOF.update!(aws)
```

Look for `l::Function` in the output (red/yellow). If present, the hypothesis is confirmed.

### Fix

Add `L` as a concrete type parameter to `AdamSubWorkspace` and its inner constructor:

```julia
struct AdamSubWorkspace{T, C, L<:Function}
    θ::T
    opt
    as::AdamState
    l::L           # concrete closure type, captured at construction
    cache::C
    function AdamSubWorkspace(θ::T, opt, as, l::L, cache::C) where {T, C, L<:Function}
        @assert typeof(l(θ)) <: Real
        return new{T, C, L}(θ, opt, as, l, cache)
    end
end
```

The outer constructor (`AdamSubWorkspace(θ, l; backend=...)`) needs no change — `l` already
has a concrete type there.

### Verification

```julia
@code_warntype SSOF.update!(aws)   # l should no longer appear as Function
```

Re-run the benchmark. If median Adam step drops to within ~2× of Nabla (≤ 600 ms), stop
here and move to final verification. If the gap remains large, continue to Cause 3.

---

## Cause 3 — `@from_rrule` dispatch not firing

### What's wrong

The two `@from_rrule` registrations in `src/ad_backend.jl` use concrete types:

```julia
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), Matrix{Float64}, Vector{Float64}, StellarInterpolationHelper}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), Matrix{Float64}, Vector{SparseMatrixCSC{Float64,Int64}}}
```

If the actual argument types at the `spectra_interp` call site inside the loss differ even
slightly (e.g., a `SubArray` where `Matrix{Float64}` is expected, a `view` of a sparse
matrix, or a different element type), Mooncake won't match the rule and will differentiate
through `spectra_interp` from scratch — defeating the custom rrule entirely.

The same risk applies to `gp_ℓ_precalc` (registered in `prior_gp_functions.jl`).

### Diagnosis step

Time `spectra_interp` differentiation in isolation and compare to the ChainRulesCore rrule
execution time:

```julia
import StellarSpectraObservationFitting as SSOF
using Mooncake, BenchmarkTools

# Reconstruct the argument types that appear at the loss call site
# (pull from a live mws after calling ModelWorkspace):
# interp_args = (mws.om.b2o.M, mws.om.star.lm.s[1], mws.om.b2o)

# Check the concrete types:
# typeof.(interp_args)

# Time the gradient:
# rule = Mooncake.build_rrule(SSOF.spectra_interp, interp_args...)
# @benchmark Mooncake.value_and_gradient!!(rule, SSOF.spectra_interp, interp_args...)
```

If build time is long (minutes) and the rule is slow, the custom rrule is not being used.
If build time is short (seconds) and the rule is fast, it's firing correctly.

A simpler check: add a `println` to the ChainRulesCore rrule bodies and call
`Mooncake.build_rrule` — if the print fires during tracing, the rule is being picked up.

### Fix

Option A — widen the `@from_rrule` signatures to match the actual argument types:

```julia
# If the spectrum matrix arrives as a SubArray:
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), SubArray{Float64,...}, Vector{Float64}, StellarInterpolationHelper}
```

Option B — rewrite `@from_rrule` as native `Mooncake.@is_primitive` + `rrule!!` wrappers
that match via abstract supertypes (e.g., `AbstractMatrix{Float64}`). This is more robust
but requires translating the ChainRulesCore rrule into Mooncake's rrule!! convention.

Option B is the safer long-term choice if argument types are inconsistent.

### Verification

After fixing, `build_rrule` for the full loss should be noticeably faster (the custom rules
short-circuit tracing through the interpolation and GP likelihood bodies). Re-run the
benchmark.

---

## Cause 2 — `tangent_to_arrays` allocation (lowest priority)

### What's wrong

Every `value_and_gradient!` call ends with:

```julia
return val, tangent_to_arrays(∂θ)
```

For the Adam path, `∂θ` is a `Vector{Any}` containing `Vector{Any}` containing float
arrays (Mooncake type erasure on nested containers). `tangent_to_arrays` walks this
recursively and calls `collect` on every SubArray leaf, allocating fresh arrays each step.
With 60 observations and O(100k) parameters, this is O(n) allocations per gradient call.

This is secondary to causes 1 and 3 because it is post-gradient overhead, but it compounds
with them.

### Diagnosis step

Profile a single call:

```julia
using Profile
Profile.clear()
@profile for _ in 1:5; SSOF.value_and_gradient!(aws.cache, aws.l, aws.θ); end
Profile.print(mincount=5)
```

Look for `tangent_to_arrays` and `collect` in the hot path.

### Fix

Pre-allocate the output tangent buffer at `prepare_gradient` time and copy into it in
`value_and_gradient!`, eliminating per-call allocation:

```julia
struct MooncakeCache{R, G}
    rule::R
    grad_buf::G    # pre-allocated, same structure as θ
end

function prepare_gradient(::MooncakeBackend, l, θ)
    rule = Mooncake.build_rrule(l, θ)
    grad_buf = deepcopy(θ)   # plain arrays, same shape
    return MooncakeCache(rule, grad_buf)
end

function value_and_gradient!(c::MooncakeCache, l, θ)
    val, (_, ∂θ) = Mooncake.value_and_gradient!!(c.rule, l, θ)
    _copy_tangent!(c.grad_buf, ∂θ)   # in-place copy, no allocation
    return val, c.grad_buf
end
```

`_copy_tangent!` traverses the nested structure and does `copyto!` at each leaf. Callers
(`AdamState!`, `first_iterate!`) must not retain references to `∂θ` across calls, which
they currently don't.

### Verification

Profile again and confirm `tangent_to_arrays`/`collect` are gone from the hot path.
Re-run benchmark for final numbers.

---

## Execution order

1. Fix Cause 1. Re-benchmark. If Adam step ≤ 600 ms, continue to final validation.
2. If still slow, diagnose Cause 3. Fix whichever form of the rrule mismatch applies.
   Re-benchmark.
3. Profile for Cause 2. Fix only if allocation is measurably in the hot path.
4. Run full test suite (`julia --project=. -e 'using Pkg; Pkg.test()'`).
5. Update `benchmark_results_remove-nabla.txt` with final numbers.
6. Open PR.

---

## Acceptance criteria

- Adam step median ≤ 600 ms (within 2× of Nabla's 286 ms), OR
- Document the remaining gap with a clear explanation of why (e.g., Mooncake traces more
  aggressively through BLAS calls that Nabla skipped).
- All existing tests pass.
- Gradient values agree with finite differences to the same tolerance as before (already
  verified by the Phase 3 cross-check test).
