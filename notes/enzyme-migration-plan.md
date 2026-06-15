# Plan: Add Enzyme.jl as the default AD backend (Mooncake retained)

This plan replaces neither `nabla-removal-plan.md` (already executed) nor
`mooncake-perf-plan.md` (now superseded — the Mooncake regression is part of
the *motivation* for this work, not something to fix first). Read it in
conjunction with the existing `src/ad_backend.jl` comments and the "Future
EnzymeBackend requirements" block at the top of that file, which already
enumerates the structural obstacles. Work on a branch (suggested:
`enzyme-backend`, branched from the current `try_enzyme` head which is at the
same commit as `remove-nabla`). One phase per commit; do not start a phase
until the previous phase's verification step passes.

## Execution status (2026-06-13)

**Phase 1 done.** Enzyme added to `[deps]` + `[compat]`, `EnzymeBackend`
struct in `src/ad_backend.jl`, load test in `test/runtests.jl`. Both backends
subtype `ADBackend`.

**Phase 2 done with two caveats.** Flat-vector `EnzymeFlatCache` +
`prepare_gradient` + `value_and_gradient!` implemented. Two cross-check
testsets in `test/runtests.jl`:

- `EnzymeBackend flat-vector path (spectra_interp via SIH)` — exercises a
  closure capturing a `StellarInterpolationHelper`.
- `EnzymeBackend on loss ∘ unflatten composition` — exercises the exact
  closure shape `opt_funcs` builds (advisor-mandated; the SIH test does not
  cover ParameterHandling's unflatten path).

Each asserts:

- `val_mc ≈ val_en` to 1e-10
- `∂_mc ≈ ∂_en` to 1e-6
- `∂_en ≈ ∂_FD` to 1e-3

Two non-obvious implementation choices, documented inline in
`src/ad_backend.jl`:

- `Enzyme.remake_zero!(c.∂l)` (not `make_zero!`) — required because
  `StellarInterpolationHelper`'s `AbstractMatrix{Float64}` field is a
  differentiable value in an immutable position. `make_zero!` errors there;
  `remake_zero!` trusts that `make_zero` already zeroed those positions.
- `set_runtime_activity(ReverseWithPrimal)` (not plain `ReverseWithPrimal`)
  — required for the `loss ∘ unflatten` composition that production
  `opt_funcs` builds. ParameterHandling's `Vector_from_vec` closure stores
  constant memory into a differentiable variable that Enzyme's static
  analysis can't prove inactive (`EnzymeRuntimeActivityError`). The runtime
  check has a small per-call cost.

**Known blocker — Julia 1.12 + `@import_rrule`.** `Enzyme.@import_rrule`'s
generated `reverse` function uses the `japi3` calling convention which the
LLVM bridge in Enzyme 0.13 does not yet support on Julia 1.12
(`CallingConventionMismatchError`; tracking issue EnzymeAD/Enzyme.jl#2707).
The current implementation differentiates through `spectra_interp` /
`gp_ℓ_precalc` / `_eval_lm_inner` / `_eval_lm` natively via Enzyme — correct
but uses the generic AD path rather than the optimized rrules. Forward plan
recorded inline in `src/ad_backend.jl`'s comment block:

1. Reassess after upgrading Enzyme (#2707 fix).
2. Write native `Enzyme.EnzymeRules.augmented_primal`/`reverse` for the four
   hot rules — math already in the ChainRulesCore bodies, only the
   activity-shadow bookkeeping needs translation. **Budget: ~1 day per rule.**
3. Downgrade benchmarking to Julia 1.11 (baseline only — not a fix).

This blocker does not affect Phase 3 (nested-θ Adam path) correctness, but
**will dominate Phase 7's perf numbers — advisor estimate puts steady-state
Adam at 5–10× *slower* than Mooncake's 919 ms without native rules**, vs.
the original target of ~400 ms. Treat the native-rule fallback (Phase 7
option 2 in `src/ad_backend.jl`'s comment block) as a merge-blocker rather
than an "if needed" optimization.

## Decisions taken before writing this plan

- **Mooncake stays.** Add `EnzymeBackend` alongside `MooncakeBackend` as a
  long-lived sibling; switch the default to `EnzymeBackend()` once cross-checks
  pass. Mooncake remains as the in-process oracle for gradient agreement
  (cheaper than finite-differences for the full loss) and as a fallback for
  cases Enzyme rejects.
- **Dep wiring.** Add Enzyme to `[deps]` during development for fast iteration.
  Convert to `[weakdeps]` + `ext/SSOFEnzymeExt.jl` package extension before
  merge, so Mooncake-only users do not pay Enzyme's load/compile cost. The
  existing `MooncakeBackend` is **not** moved to an extension — it stays in the
  main module.
- **Aliasing strategy.** Try `Duplicated(l, ∂l)` with `∂l` built via
  `Enzyme.make_zero(l)` first. Only restructure individual losses to take
  `om`/`o`/`d` as explicit arguments if Enzyme rejects the Duplicated form for
  a specific loss. Track which losses needed restructuring in a "Phase 4
  follow-up" subsection as we go.

## Background — what we are replacing and why

Both predecessor states are documented in this repo. Pull from them
verbatim where possible:

- `nabla-removal-plan.md` — original migration off Nabla. Phases 1–3 done on
  `remove-nabla` (commits `3be0b5c`, `835a8bf`, `dcde871`, `e652240`). The
  backend seam (`src/ad_backend.jl` with `ADBackend`, `prepare_gradient`,
  `value_and_gradient!`) was built precisely so a second backend could be
  swapped in without touching the optimizers. We are now exercising that
  seam.
- `mooncake-perf-plan.md` — three diagnosed Mooncake performance regressions:
  (1) `l::Function` field type instability (already fixed —
  `AdamSubWorkspace{T,C,L<:Function}` in `src/optimization_functions.jl:396`),
  (2) `@from_rrule` dispatch missing, (3) `tangent_to_arrays` allocation.
  `benchmark_results_remove-nabla.txt` records the current Mooncake
  performance: Adam step median **919 ms**, first construction **89 s**,
  `finalize_scores!` first call **268 s**.
- The current Mooncake baseline is ~3.2× slower per Adam step than the Nabla
  baseline cited in the perf plan (286 ms). Enzyme's stated strengths
  (whole-program LLVM-level differentiation, native mutation support, no
  per-call tangent translation) are exactly the axes Mooncake is paying on,
  so this is the right time to try.

## What the seam guarantees we *don't* have to touch

Already engine-agnostic and untouched in any phase below:

- `src/model_functions.jl` rrules (lines 161–184, 978–986, 988–996, 998–1043)
  — `ChainRulesCore.rrule`s. Enzyme imports them via `@import_rrule`; no edits.
- `src/prior_gp_functions.jl` rrule (lines 288–296) — same.
- All three call sites of the seam:
  - `AdamSubWorkspace` in `optimization_functions.jl:396-415`
  - `opt_funcs` in `optimization_functions.jl:804-817`
  - `estimate_σ_curvature_helper` in `error_estimation.jl:9-72`
  All three already take `backend::ADBackend` keywords with default
  `MooncakeBackend()`. Phase 6 changes that default to `EnzymeBackend()`; no
  other call-site changes.

## Enzyme-specific obstacles to budget for

These are listed in priority order. The plan's phases address them in this
order.

### O1 — Closure aliasing with mutable captures

The Adam path's loss closures (`l_total`, `l_total_s`, `l_telstar`, etc.,
defined at `optimization_functions.jl:93-300`-ish) capture `om`, `o`, `d` and
operate on `θ` whose entries alias arrays inside `om`. Specifically:

- `vec(lm)` (returned for the Adam `θ`) returns the model's own arrays.
  Example: `θ[2][3] === om.star.lm.μ` is identity-true, not equality-true.
- `downsize_view` (`model_functions.jl:939-961`) returns `view`s into the
  model arrays as `lm.M`, `lm.s` — these views are then put into θ.

Enzyme tracks tangent storage by pointer identity, so to produce a correct
gradient w.r.t. θ we need:

- a shadow closure `∂l` whose captured `∂om` is a structurally-identical
  zero-copy of `om`, preserving any aliasing within `om`,
- a shadow `∂θ` whose entries alias `∂om`'s arrays in *exactly* the same way
  that `θ`'s entries alias `om`'s arrays.

`Enzyme.make_zero(l)` recurses through `l`'s captured fields and returns a
deepcopy with all `Float64` arrays zeroed. Crucially, Julia's `deepcopy`
preserves shared-structure identity, so if `om.tel.lm.M` and some other field
are the same array object, the shadow will share too. This is the property
that makes the Duplicated-closure path viable here — but it must be verified
empirically per loss (Phase 3 step 4 below).

The matching `∂θ` is *not* given to us by `make_zero` — it must be rebuilt
from `∂l`'s captured `∂om` using the same `vec(lm)` / `downsize_view` calls
that built the primal `θ`. The cleanest way is to factor that construction
into a builder function `build_θ(om)` that lives next to the workspace, so
that `prepare_gradient` can call it on `∂l.∂om` and get a matched `∂θ` for
free.

### O2 — DPCA in-loss mutation

`_loss_recalc_rv_basis` at `optimization_functions.jl:62-65` writes into
`om.rv.lm.M` from inside the differentiated function:

```julia
function _loss_recalc_rv_basis(o::Output, om::OrderModel, d::Data; kwargs...)
    om.rv.lm.M .= doppler_component_AD(om.star.λ, om.star.lm.μ)
    return _loss(o, om, d; kwargs...)
end
```

This is the DPCA path only (`l_total` for `OrderModelDPCA`, not Wobble). Under
Nabla this was a captured-array write so no gradient flowed through it; under
Mooncake the gradient does flow but `om.rv.lm.M` is not in θ for that
workspace. Under Enzyme, this combination of *mutating a captured array that
also aliases part of θ in some workspaces* is exactly the failure mode that
yields "Constant memory is stored to a differentiable variable" errors.

The hoist (nabla-removal-plan.md Phase 2 step 6) was deferred for Mooncake.
For Enzyme it is mandatory: refresh `om.rv.lm.M` once before each gradient
call, outside the differentiated region. Concretely, add a `pre!` callback to
`AdamSubWorkspace` and `OptimSubWorkspace`; default no-op; DPCA workspaces
install `pre!(om) = (om.rv.lm.M .= doppler_component_AD(om.star.λ, om.star.lm.μ))`.

### O3 — Custom rrule import

Enzyme's `Enzyme.@import_rrule` macro maps a ChainRulesCore.rrule onto its
internal custom-rule mechanism. All five `spectra_interp` rrules, the
four `_eval_lm_inner` rrules, the `_eval_lm` rrule, and the `gp_ℓ_precalc`
rrule must be imported.

**Budget reality check (per advisor review):** Enzyme's `@import_rrule` in
0.13.x is historically stricter than Mooncake's `@from_rrule` on abstract
signatures — it has typically required *concrete* `Tuple{...}` and rejected
`AbstractMatrix{Float64}` / `AbstractVector{<:SparseMatrixCSC}`. The current
`src/ad_backend.jl` has three abstract-typed `@from_rrule` registrations for
`spectra_interp`, plus a `SMatrix{3,3,Float64,9}` signature for `gp_ℓ_precalc`,
plus four `Val{...}`-dispatched `_eval_lm_inner` registrations. If
`@import_rrule` rejects any of them, the fallback is a hand-written
`Enzyme.EnzymeRules.augmented_primal` + `reverse` pair. **Budget ~1 day per
abstract-signature rule that needs hand-translation**, not half a day for all
of them combined. The math is already correct in the rrule body; the work is
mostly translating ChainRulesCore's `(Δargs...)`-return convention to
Enzyme's `Duplicated`-shadow-write convention.

Phase 1's smoke test (step 4 below) does the cheapest concrete `@import_rrule`
first so we learn whether to inflate this budget within hours, not days.

Note: Enzyme decided in 0.13+ that `@import_rrule` is the supported entry
point for ChainRules; older docs may show `Enzyme.Compiler.Interpreter`-level
hooks that no longer apply. Check the *installed* Enzyme version's
`@import_rrule` docstring before writing the import block.

### O4 — Nested-θ shadow structure

The Adam `θ` is `Vector{<:AbstractArray}` whose entries may themselves be
`Vector{<:AbstractArray}` (mixing `Matrix{Float64}` and `Vector{Float64}` and
`SubArray`s). Once `∂θ` is built (per O1's `build_θ(∂om)` recipe), each Adam
step needs to zero it before calling `Enzyme.autodiff`. Zeroing must traverse
the same nested structure — write a `_zero_θ!(∂θ)` helper that mirrors
`tangent_to_arrays` but in-place.

### O5 — Type stability through `EnzymeBackend`

The `AdamSubWorkspace{T, C, L<:Function}` parameterization (already in place)
makes `aws.l` concretely typed at the call site. Apply the same care to the
new `EnzymeCache` struct: parameterize on the rule/shadow types so
`value_and_gradient!(cache, l, θ)` is fully specialized. The Mooncake-era
`@code_warntype` recipe in `mooncake-perf-plan.md` Cause 1 is the same
diagnostic playbook here.

## Phase 0 — Capture a fixed gradient oracle (no code changes)

Before any Enzyme work, lock in the Mooncake gradients on a small reproducible
case so Phase 3 can diff against them with bitwise-stable comparison. Mooncake
gradients are deterministic given the same θ, so this is a one-shot.

1. Create `test/oracle_gradients.jl` (or a throwaway script in `$TMPDIR`)
   that:
   - Builds a tiny synthetic `OrderModelWobble` + `GenericData` (≤300 pixels,
     5 epochs). Follow the small-input construction style in
     `test/runtests.jl:54-71` rather than reusing `examples/data/`.
   - Builds an `AdamSubWorkspace` and an `OptimSubWorkspace` for it (Mooncake
     default).
   - Calls `value_and_gradient!` once per workspace and writes
     `(loss::Float64, ∂θ::Vector{...})` to a JLD2 file outside the repo.
   - Records the Julia/Mooncake versions used.
2. Repeat for `OrderModelDPCA` with the *non-hoisted* loss (i.e. current state),
   plus a separate oracle for the hoisted loss (call
   `om.rv.lm.M .= doppler_component_AD(...)` manually before each gradient
   call). The Enzyme path will match the hoisted oracle; the original
   non-hoisted DPCA oracle is recorded only for documenting the semantic
   change in the PR.
3. Run `julia --project=. examples/example.jl` and save the resulting RVs to
   `$TMPDIR/ssof_mooncake_baseline/`. The Enzyme run in Phase 7 must match
   these to near machine precision on the Wobble path.

**Verification:** Phase 0 produces oracle files; no code in the repo changes.

## Phase 1 — Dependency scaffolding and Enzyme load test

1. `Project.toml`:
   - Add `Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"` to `[deps]`.
   - Add `Enzyme = "0.13"` (or current major) to `[compat]`.
   - Leave Mooncake entries unchanged.
   - Leave `julia = "1.10"` unless Enzyme 0.13+ requires more (check; bump if
     needed, flag in PR).
2. In `src/ad_backend.jl`, add a new section after the Mooncake block:

   ```julia
   # ── Enzyme implementation ─────────────────────────────────────────────
   import Enzyme
   struct EnzymeBackend <: ADBackend end
   ```

   At this phase, do **not** implement `prepare_gradient` or
   `value_and_gradient!` for `EnzymeBackend`. We are only confirming the
   package loads cleanly alongside Mooncake.
3. Add a single load test to `test/runtests.jl`:

   ```julia
   @testset "backend module loads" begin
       @test SSOF.MooncakeBackend() isa SSOF.ADBackend
       @test SSOF.EnzymeBackend() isa SSOF.ADBackend
   end
   ```

**Verification:** `Pkg.test()` passes (Enzyme loads; nothing yet uses it).

## Phase 2 — Flat-vector `EnzymeBackend` (Optim and error_estimation paths)

The flat-`Vector{Float64}` path is the easy half: `θ` is a single dense
buffer, no aliasing into captured state, and Enzyme's API maps cleanly. Get
this working before tackling the Adam nested-θ path.

**Pre-step (30-min spike, optional):** `Enzyme.gradient` is a higher-level
wrapper that handles the `make_zero!`/`autodiff` plumbing for flat-vector
inputs. It may produce cleaner code for the flat-vector path *only*, but
does not generalize to the nested-θ path. If a quick experiment shows it
beats the explicit `Duplicated` plumbing on both readability and perf, use
it here and switch to explicit plumbing in Phase 3. If not, the explicit
form below already works for both phases and avoids the cognitive load of
two code paths.

1. In `src/ad_backend.jl`, implement for `θ::AbstractVector{<:Real}`:

   ```julia
   struct EnzymeFlatCache{L,T}
       l::L                    # remembered for shadow-closure freshness check
       ∂l::L                   # shadow closure built once via make_zero
       ∂θ::T                   # pre-allocated shadow vector
   end

   function prepare_gradient(::EnzymeBackend, l, θ::AbstractVector{<:Real})
       ∂l = Enzyme.make_zero(l)
       ∂θ = zero(θ)
       return EnzymeFlatCache(l, ∂l, ∂θ)
   end

   function value_and_gradient!(c::EnzymeFlatCache, l, θ::AbstractVector{<:Real})
       Enzyme.make_zero!(c.∂l)
       fill!(c.∂θ, 0)
       _, val = Enzyme.autodiff(
           Enzyme.ReverseWithPrimal,
           Enzyme.Duplicated(l, c.∂l),
           Enzyme.Active,
           Enzyme.Duplicated(θ, c.∂θ),
       )
       return val, c.∂θ
   end
   ```

   Notes:
   - `Active` is the return-value activity (loss is a scalar).
   - `Duplicated(l, c.∂l)` handles closures that capture mutable state. If a
     specific loss has no captures (rare here), `Const(l)` would be faster but
     the same code path works for both — keep `Duplicated` until profiling
     justifies a special case.
   - `make_zero!(c.∂l)` is the in-place reset. **Do not fall back to
     `deepcopy(make_zero(l))` per call** — that allocates a fresh shadow
     closure and breaks the alias between `c.∂l`'s captured `∂om` and the
     `∂om` that `c.∂θ` aliases into (the nested-θ path in Phase 3 relies on
     this alias). The correct fallback when `make_zero!` misbehaves on a
     specific type is a manual `_zero_in_place!(∂x)` helper that walks the
     struct and `fill!`s each array leaf — no allocation, no identity
     loss. Same applies to `c.∂om` in Phase 3.
2. Wire imports of the ChainRulesCore rrules into Enzyme. For each
   `Mooncake.@from_rrule` block already in `src/ad_backend.jl` (lines 52–76)
   and `src/prior_gp_functions.jl` (lines 297–303), add a matching
   `Enzyme.@import_rrule` call **in `src/ad_backend.jl`**. Keep the comment
   block that explains *why* each signature is shaped the way it is. If an
   `@import_rrule` rejects a signature (most likely the abstract-type
   fallbacks and the `SMatrix{3,3,Float64,9}` GP signature), translate that
   one rule by hand to an `EnzymeRules.augmented_primal`/`reverse` pair; the
   math is already in the ChainRulesCore body, just transcribed.
3. Verify each imported rule fires. Enzyme exposes a runtime check via
   `Enzyme.EnzymeRules.has_rrule_from_sig`; call it on each registered
   signature inside a testset that runs at load. If a rule doesn't register,
   the testset fails — better than silently re-deriving the math.
4. Extend `test/runtests.jl`:
   - Duplicate the existing "custom spectra_interp() sensitivity" testset
     (lines 52–71) with `EnzymeBackend()` substituted for `MooncakeBackend()`.
   - Add `EnzymeBackend()` to a `backends_to_test` list and loop the testset.
   - Add a tiny `OptimSubWorkspace` cross-check: build the workspace under
     each backend, compute `value_and_gradient!` on a fresh flat θ, compare
     values element-wise (relative tolerance 1e-8 for the loss and 1e-6 for
     the gradient — both backends carry rounding noise on a 5-epoch problem).

**Verification:** All `runtests.jl` testsets pass with both backends. The
oracle gradients from Phase 0 (Optim path only) match Enzyme's output within
the cross-check tolerance.

## Phase 3 — Nested-θ `EnzymeBackend` for the Adam path

This is the substantive engineering phase. The flat-vector implementation
from Phase 2 does not generalize directly because:

- `θ::Vector{<:AbstractArray}` is a heterogeneous container; `zero(θ)` is not
  defined for it.
- θ entries alias arrays in `om`, and `om` is captured by the loss closure.
  The shadow vector must alias the shadow closure's `om` in the same way.

The strategy is a builder-based cache: each Adam path passes a `build_θ`
closure to `prepare_gradient` that knows how to rebuild a θ-shaped container
from any `OrderModel` (the primal `om` for the primal `θ`, and the shadow
`∂om` for the shadow `∂θ`).

1. Introduce a struct + interface change at the workspace level. In
   `src/optimization_functions.jl`, every constructor of `AdamSubWorkspace`
   currently passes a pre-built `θ` and a `l = closure(θ)`. Change the contract
   to:

   ```julia
   AdamSubWorkspace(om, build_θ, build_loss; backend=EnzymeBackend()) = ...
   ```

   where:
   - `build_θ(om)::Vector{<:AbstractArray}` is the existing `θ`-construction
     code, lifted into a pure function.
   - `build_loss(om, o, d)::Function` is the closure factory currently
     inlined into each `loss_funcs_*` definition.

   Keep the old `AdamSubWorkspace(θ, l; backend=...)` constructor delegating
   to the new one for the Mooncake backend (which doesn't need the builders),
   so the diff stays localized.

   **Map the callers first:** before touching the constructor signature, run
   `grep -rn "AdamSubWorkspace(" src/ test/` and enumerate every call site.
   The delegating-old-constructor pattern covers test code, but
   `optimization_functions.jl` itself constructs workspaces internally (e.g.
   `TotalWorkspace`, `FrozenTelWorkspace`), and those callers need to know
   how to produce `build_θ` and `build_loss` for the Enzyme path. If a
   caller can't provide builders without itself accepting `om`/`o`/`d`, the
   change ripples upward; budget time for that.

   *Surgical-changes constraint:* if `loss_funcs_total` etc. resist being
   factored cleanly (e.g. closures over locals other than `om`/`o`/`d` that
   we cannot reconstruct from the model), bail on the builder approach for
   that loss only and use the deepcopy-of-closure trick described in step 3
   instead.

2. In `src/ad_backend.jl`, add:

   ```julia
   struct EnzymeNestedCache{L, T, OM}
       ∂l::L
       ∂θ::T
       ∂om::OM      # held alive; ∂l and ∂θ both alias into this
   end

   function prepare_gradient(
       ::EnzymeBackend,
       l,
       θ::AbstractVector{<:AbstractArray};
       om,
       build_θ,
       build_loss,
       o, d,
   )
       ∂om = Enzyme.make_zero(om)   # deepcopy-and-zero, preserves aliasing
       ∂l  = build_loss(∂om, o, d)
       ∂θ  = build_θ(∂om)
       return EnzymeNestedCache(∂l, ∂θ, ∂om)
   end

   function value_and_gradient!(c::EnzymeNestedCache, l, θ::AbstractVector{<:AbstractArray})
       _zero_θ!(c.∂θ)               # leaf-by-leaf fill!(_, 0)
       Enzyme.make_zero!(c.∂om)     # zero any non-θ-aliased buffers
       _, val = Enzyme.autodiff(
           Enzyme.ReverseWithPrimal,
           Enzyme.Duplicated(l, c.∂l),
           Enzyme.Active,
           Enzyme.Duplicated(θ, c.∂θ),
       )
       return val, c.∂θ
   end

   _zero_θ!(v::AbstractVector{<:AbstractArray}) = (foreach(_zero_θ!, v); v)
   _zero_θ!(a::AbstractArray{<:Real}) = fill!(a, 0)
   ```

3. Aliasing verification step (do this *before* trusting any gradient
   number). `Enzyme.make_zero` uses `IdDict`-based identity tracking under
   the hood, so shared-object identity within `om` is preserved across the
   deepcopy. Verify by building the workspace and asserting at construction
   time:

   ```julia
   @assert pointer(c.∂θ[k]) === pointer(<corresponding ∂om array field>)
   ```

   for every θ entry that aliases an `om` *array* field (i.e. `μ`, `M`,
   `s`). The exact assertion list is small (≤6 entries, see the Wobble and
   DPCA `l_total` definitions).

   **Do not assert pointer-identity on `Dict` fields** like `om.reg_tel` /
   `om.reg_star`. `Dict{Symbol, Float64}` entries are immutable `Float64`s
   that `make_zero` recreates per entry rather than aliasing — that's
   correct behavior (Floats are not tangent-bearing the way arrays are), but
   `pointer(::Float64)` is a category error. Aliasing assertions are
   array-only.

   If any array-pointer mismatches, the builder approach has failed for that
   case — restructure that specific loss instead (see Phase 4).

4. Closure-shadow fallback. If `Enzyme.make_zero(l)` for a captured closure
   misbehaves (Enzyme has known gaps on some closure types, particularly
   ones that capture `Dict`s — `reg_tel` and `reg_star` are `Dict`s), use the
   builder path instead: never `make_zero` the closure; always reconstruct it
   from `(∂om, o, d)` via `build_loss`. This is what step 2's signature
   already implies; `make_zero(l)` is not invoked anywhere in that code
   block. Document this choice in a one-line comment.

5. Update tests. Extend the `backends_to_test` loop from Phase 2 step 4 to
   include a `value_and_gradient!` cross-check on an `AdamSubWorkspace`:

   ```julia
   @testset "Adam gradient agrees: $(typeof(b))" for b in backends_to_test
       aws = SSOF.AdamSubWorkspace(om, build_θ, build_loss; backend=b)
       val, ∂θ = SSOF.value_and_gradient!(aws.cache, aws.l, aws.θ)
       # Compare to Mooncake oracle (Phase 0)
       @test isapprox(val, oracle.val; rtol=1e-10)
       @test all(isapprox.(_flatten(∂θ), _flatten(oracle.∂θ); rtol=1e-6))
   end
   ```

**Verification:** The Phase 0 Mooncake oracle matches Enzyme's nested-θ
gradient to 1e-6 relative across every loss in `loss_funcs_total`,
`loss_funcs_telstar`, `loss_funcs_frozen_tel`. *Both* `OrderModelWobble` and
`OrderModelDPCA` paths are exercised. **DPCA matches only after the Phase 4
hoist** — that's the gate for Phase 4.

**Also compare against finite differences, not just against Mooncake.** If
Mooncake has a subtle bug — e.g. on the DPCA `μ`-flow path — Enzyme would
match it and we'd ship two-wrongs-make-a-right. The existing `est_∇` helper
at `test/runtests.jl:9-19` is already configured for the test environment;
extend the Phase 3 cross-check loop to run `est_∇` on at least one loss per
workspace type (Wobble `l_total`, DPCA `l_total`, `l_telstar`,
`l_frozen_tel`) and assert all three of {Mooncake, Enzyme, FD} agree to ~1e-4
relative. This is the gold-standard correctness gate; Mooncake↔Enzyme
agreement alone is necessary but not sufficient.

## Phase 4 — DPCA mutation hoist (`_loss_recalc_rv_basis`)

Trigger: Phase 3 verification will show DPCA gradients differ from the
non-hoisted Mooncake oracle. The expected difference is exactly what
Mooncake plan's step 6 predicted (gradient flowing through `om.star.lm.μ`
into the basis update). Fix:

1. Add `pre!::F` field to `AdamSubWorkspace`:

   ```julia
   struct AdamSubWorkspace{T,C,L<:Function,F}
       θ::T
       opt
       as::AdamState
       l::L
       cache::C
       pre!::F      # default identity; DPCA installs basis refresh
   end
   ```

   Default constructor sets `pre! = _identity_pre!`. DPCA constructors set
   `pre! = om -> (om.rv.lm.M .= doppler_component_AD(om.star.λ, om.star.lm.μ))`.

2. `update!` (`optimization_functions.jl:423`) calls `aws.pre!(om)` *before*
   `value_and_gradient!`. The shadow closure's `∂om.rv.lm.M` does not need
   the basis refresh — that array is zeroed at the start of each call by
   `make_zero!(c.∂om)`.

3. **Same hoist for the Optim path, but per-closure-call, not per-step.**
   `opt_funcs` (`optimization_functions.jl:804-817`) returns `g!(G,θ)` and
   `fg_obj!(G,θ)` closures that L-BFGS line search invokes *many times per
   outer iteration*. The basis refresh must fire on every closure call, not
   on every L-BFGS step. Concretely:

   ```julia
   function g!(G, θ)
       pre!(om)                                 # refresh BEFORE gradient
       G .= value_and_gradient!(cache, f, θ)[2]
   end
   function fg_obj!(G, θ)
       pre!(om)
       val, ∂θ = value_and_gradient!(cache, f, θ)
       G .= ∂θ
       return val
   end
   ```

   Thread `pre!` into `opt_funcs` as a keyword argument, defaulting to a
   no-op; `OptimSubWorkspace` for the DPCA path passes the basis-refresh
   closure. The same applies anywhere `finalize_scores!` ends up calling
   `opt_funcs` for a DPCA workspace (grep to confirm).

4. Rebuild the Phase 0 DPCA oracle with the hoist applied (the "hoisted
   oracle" file). Phase 3's DPCA cross-check now compares against this
   oracle. Wobble path is unchanged.

5. **Document the semantic change** in a `# Migration note` block at the top
   of `loss_funcs_total(o::Output, om::OrderModelDPCA, d::Data)`: the
   gradient now flows through `om.star.lm.μ` into the Doppler basis. Worth
   discussing with Christian whether this is *more* correct (probably yes —
   it captures coupling Nabla silently dropped) or whether it should be
   recovered with a `stop_gradient`-equivalent. Default: keep the new
   semantics; if Christian objects, wrap the call to `doppler_component_AD`
   in something that zeroes the `μ` activity.

**Verification:** DPCA `l_total` Enzyme gradient matches the hoisted oracle
to 1e-6 relative on the synthetic test case. Wobble cross-check is
unchanged from Phase 3.

## Phase 5 — Switch the default backend

1. Change the default keyword on every seam call site from `MooncakeBackend()`
   to `EnzymeBackend()`:
   - `AdamSubWorkspace(θ, l::Function; backend::ADBackend=...)` —
     `optimization_functions.jl:412`
   - `opt_funcs(loss, pars; backend::ADBackend=...)` —
     `optimization_functions.jl:804`
   - `estimate_σ_curvature_helper(...; ...)` — `error_estimation.jl:19, 35`
     (calls `prepare_gradient(MooncakeBackend(), ...)` directly; replace
     with a keyword-threaded `backend::ADBackend=EnzymeBackend()`).
2. Sweep `grep -rn "MooncakeBackend()" src/` and replace with
   `EnzymeBackend()` everywhere the default is implicit. Keep
   `MooncakeBackend()` references in tests' `backends_to_test` loops.
3. Run the full `Pkg.test()` plus `julia examples/example.jl` end-to-end. The
   Wobble RVs from this run must match `$TMPDIR/ssof_mooncake_baseline/`
   to ~1e-10 relative (Adam is deterministic; the only source of difference
   is gradient agreement, which is bounded by the cross-check tolerance from
   Phase 3).

**Verification:** All tests pass. Example RVs match Mooncake baseline within
1e-10. `benchmark_ad.jl` (untracked at repo root) runs against the new
default and produces a new `benchmark_results_<branch>.txt`.

## Phase 6 — Convert Enzyme to a package extension

Now that the implementation is stable, isolate the Enzyme code so non-Enzyme
users don't pay for it on load.

1. Restructure `src/ad_backend.jl`:
   - Keep the `ADBackend` abstract type, `MooncakeBackend`, and Mooncake
     implementation in the main module file.
   - Define `struct EnzymeBackend <: ADBackend end` as an *empty placeholder*
     in the main module (so it's reachable for dispatch and tests can
     reference it without Enzyme loaded).
   - Move every Enzyme-using definition (cache structs, `prepare_gradient`,
     `value_and_gradient!`, `@import_rrule` block) into a new file
     `ext/SSOFEnzymeExt.jl` that imports `Enzyme` and extends
     `StellarSpectraObservationFitting`'s methods.
2. `Project.toml`:
   - Move `Enzyme` from `[deps]` to `[weakdeps]`.
   - Add `[extensions] SSOFEnzymeExt = "Enzyme"`.
   - Keep `[compat] Enzyme = "..."` (compat applies to both deps and
     weakdeps).
3. Update CI / test target. The test environment (`[targets].test`) must list
   Enzyme so tests can load it. The `using StellarSpectraObservationFitting`
   call in user code does *not* load Enzyme until `using Enzyme` is invoked.
4. Add a `prepare_gradient(::EnzymeBackend, ...)` fallback method in the main
   module that throws a clear error when Enzyme isn't loaded:

   ```julia
   prepare_gradient(::EnzymeBackend, l, θ) =
       error("EnzymeBackend requires `using Enzyme` to load the extension.")
   ```

5. Default backend question: with Enzyme behind an extension,
   `EnzymeBackend()` as the seam default means a user who follows the README
   `import SSOF` and calls `ModelWorkspace` will hit the error above unless
   they also `using Enzyme`. Two options:
   - **(a)** Document in the README that Enzyme must be loaded.
   - **(b)** Make the default fall back to Mooncake when Enzyme isn't loaded.
     Detect via `Base.get_extension(SSOF, :SSOFEnzymeExt) !== nothing`.

   Recommend (b) — silent graceful fallback, with a `@info` on first
   workspace construction telling the user how to get Enzyme back.

**Verification:** A fresh resolve with `julia --project=. -e 'using Pkg;
Pkg.instantiate(); Pkg.test()'` (Manifest deleted) passes. A separate
test in a minimal env that has SSOF but *not* Enzyme also passes (Mooncake
default fires).

## Phase 7 — Benchmark and acceptance

1. Run `julia benchmark_ad.jl` from the repo root (the script auto-detects
   the branch via `git rev-parse`; output goes to
   `benchmark_results_<branch>.txt`). Compare to the existing
   `benchmark_results_remove-nabla.txt` (Mooncake) and the Nabla baseline
   numbers cited in `mooncake-perf-plan.md`.
2. Acceptance targets:
   - **Steady-state Adam step**: median ≤ 400 ms (target: ~2.3× faster than
     current Mooncake's 919 ms; ~1.4× of Nabla's 286 ms). Hard floor: ≤ 600
     ms (any worse and we have a perf bug worth investigating before merge).
   - **ModelWorkspace construction (first call)**: ≤ 90 s. Enzyme's
     compile time is typically longer than Mooncake's `build_rrule`; this
     ceiling allows for that. If first-call construction exceeds 180 s,
     investigate before merge (likely cause: a custom rule failed to import
     and Enzyme is re-deriving math through the rrule body).
   - **`finalize_scores!` first call**: ≤ 300 s. Same rationale.
3. Compare the example pipeline's RVs against the Mooncake baseline from
   Phase 0 step 3. Acceptance: within 1e-10 relative on every order's RVs
   for the Wobble path. DPCA may diverge (different gradient semantics post
   hoist); document the size of the divergence rather than treating it as a
   regression.
4. PR description must include:
   - Cross-backend gradient agreement table (Mooncake vs. Enzyme on the
     synthetic test losses).
   - Timing table: Nabla baseline, current Mooncake, new Enzyme.
   - DPCA semantics note (Phase 4 step 5).
   - List of losses (if any) that needed restructuring beyond the builder
     factoring of Phase 3 step 1.
   - The aliasing-verification assertions from Phase 3 step 3, with their
     pass/fail status.

## Phase 8 (optional, post-merge) — Mooncake removal

Open as a separate issue if Enzyme proves robust over a few weeks of use.
Removal would entail:

- Delete the `MooncakeBackend` implementation block from `src/ad_backend.jl`.
- Drop `Mooncake` from `[deps]` and `[compat]`.
- Move every `@from_rrule` block out (the rrules in
  `src/model_functions.jl` and `src/prior_gp_functions.jl` stay — they are
  pure ChainRulesCore and Enzyme imports them directly).
- Update tests so `backends_to_test = [EnzymeBackend()]`.
- Delete `mooncake-perf-plan.md`.

Until then, both backends remain first-class and the seam acceptance gate
stays at "two-backend agreement on every cross-check testset".

## Known risks

- **Aliasing-preservation in `make_zero`.** Phase 3 step 3 catches this with
  pointer-identity asserts, but if `Enzyme.make_zero` does not preserve
  shared-object identity across a `Dict` field in `OrderModel` (e.g.
  `reg_tel`, `reg_star`), we'll need the builder fallback for that field.
- **`@import_rrule` signature rejection** for `SMatrix{3,3,Float64,9}` (the
  GP one), nested-abstract types, or `Val{...}` (the `_eval_lm_inner`
  signatures). Native-rule transcription is the fallback; budget half a day.
- **Compile time blow-up on closures with `Dict` captures.** `reg_tel` /
  `reg_star` are `Dict{Symbol, Float64}` (~10 entries). If Enzyme refuses to
  differentiate through them — or recompiles per loss — the priors can be
  hoisted out the same way the DPCA basis is. Defer until measured.
- **Thread safety in `error_estimation.jl`.** The Mooncake plan already
  noted gradient caches aren't thread-safe; Enzyme's are even more so
  because the shadow closure is mutable state. The current code already
  builds one cache per thread inside `Threads.@threads`; keep that pattern.
- **Extension reload friction.** Until Phase 6, Revise reloads the main
  module fine. After Phase 6, Revise's handling of package extensions has
  edge cases (especially around removing methods). If editing
  `ext/SSOFEnzymeExt.jl` interactively becomes painful, do hot iteration in
  `src/ad_backend.jl` and only move code once the design is final.
- **DPCA semantic change.** The Phase 4 hoist makes RV gradients *different*
  from the pre-port Nabla numbers on the DPCA path. The Wobble path is
  unaffected (no `_loss_recalc_rv_basis` there). Surface this in the PR.

## Open questions to raise with the user before Phase 3

1. **DPCA hoist confirmation.** The plan defaults to "let the gradient flow
   through `μ` is more correct, keep the new semantics". Christian should
   weigh in before we make this user-visible — but since the Wobble path is
   the default and the only one with a published RV pipeline, this is not
   blocking.
2. **`benchmark_ad.jl` checked in?** It's currently untracked at the repo
   root. If we want Phase 7's numbers to be reproducible from this branch,
   commit it (with the JLD2 file paths noted as a precondition). If not,
   keep it local. The plan does not assume either way.

