# Plan: Replace Nabla.jl with Mooncake.jl + ChainRulesCore.jl

This plan is written to be executed by another agent (or developer) without further
context. Read the cited files before editing them. Work on a branch
(e.g. `remove-nabla`), one phase per commit, and do not start a phase until the
previous phase's verification step passes.

## Background and rationale

SSOF currently uses Nabla.jl (a tape-based operator-overloading reverse-mode AD that
is no longer actively developed, hence the `Nabla = "< 0.13"` pin) to differentiate
its χ² + prior loss functions. Nabla is used in three ways:

1. As the gradient engine for both optimizers:
   - `src/optimization_functions.jl:400` — `gl = ∇(l; get_output=true)` inside
     `AdamSubWorkspace` (the custom Adam path). Here `θ` is a
     `Vector{<:AbstractArray}` whose entries can themselves be vectors of arrays
     (from `vec(lm::LinearModel)`) and can contain `SubArray`s (from
     `downsize_view`).
   - `src/optimization_functions.jl:778-794` — `opt_funcs` builds `∇(loss)` and
     `∇(loss; get_output=true)` for the Optim.jl L-BFGS path, where parameters are
     flattened to a `Vector{Float64}` with ParameterHandling's `flatten`
     (extended in `src/flatten.jl`).
   - `src/error_estimation.jl:19` — optional `g = ∇(ℓ)` when
     `estimate_σ_curvature_helper` is called with `use_gradient=true`; `ℓ` takes a
     plain `Vector{Float64}`.
2. Three custom sensitivities (analytic gradients, registered with
   `@explicit_intercepts` + `Nabla.∇` overloads):
   - `spectra_interp(model_flux::AbstractMatrix, rvs::AbstractVector, sih::StellarInterpolationHelper)`
     w.r.t. arg 1 — `src/model_functions.jl:146-181`.
   - `spectra_interp(model::AbstractMatrix, interp_helper::AbstractVector{<:SparseMatrixCSC})`
     w.r.t. arg 1 — `src/model_functions.jl:959-969`.
   - `gp_ℓ_precalc(Δℓ_coeff, x, A_k, Σ_k)` w.r.t. arg 2 —
     `src/prior_gp_functions.jl:263-286` (gradient is `Δℓ_precalc`; note the
     existing warning that the registered gradient is only valid for the default
     values of `H_k`, `P∞`, and `σ²_meas`).
3. Small accommodations for Nabla quirks:
   - `src/Nabla_extension.jl` (included from
     `src/StellarSpectraObservationFitting.jl:13`) — `Nabla.zerod_container`
     method so `Vector{<:SubArray}` parameters work.
   - Untyped fallback methods `spectra_interp(model_flux, rvs, sih) = spectra_interp_nabla(...)`
     at `src/model_functions.jl:152-161` (and the commented-out sparse analog at
     `model_functions.jl:963-966`) that exist so Nabla's boxed `Node` types hit an
     index-based (non-view) implementation.
   - `src/model_functions.jl:440` — a `muladd`-based `_eval_lm` method is bypassed
     because "Nabla doesn't handle it".
   - `gp_ℓ_nabla` (`src/prior_gp_functions.jl:130`) — a Nabla-friendly variant of
     `gp_ℓ`, now only referenced in comments and docstrings.

Target stack:

- **Mooncake.jl** as the single AD engine everywhere Nabla's `∇` is used. Chosen
  because the loss closures mutate captured state (`_loss_recalc_rv_basis` at
  `src/optimization_functions.jl:63-67` writes into `om.rv.lm.M`), the parameters
  contain views and nested vectors-of-arrays, and Mooncake supports mutation,
  `SubArray`s, and arbitrary nested structures. Zygote rules out on mutation;
  Enzyme would need careful activity annotations for the captured `OrderModel`.
- **ChainRulesCore.jl** to express the three custom rules as `rrule`s
  (backend-agnostic; a draft already exists in comments at
  `src/prior_gp_functions.jl:324-334`), imported into Mooncake with
  `Mooncake.@from_rrule`. If `@from_rrule` can't handle a signature (check the
  current Mooncake docs for kwarg and abstract-type restrictions in the installed
  version), fall back to a native `Mooncake.@is_primitive` + `Mooncake.rrule!!`
  for that function instead — the math stays the same.
- Test-only: **FiniteDifferences.jl** and **ChainRulesTestUtils.jl** (and
  `Mooncake.TestUtils.test_rule`).

Additionally, the engine call sites go behind a thin internal backend seam (see
the next section) so a user can later swap Mooncake for Enzyme without touching
the optimizers. Mooncake is the only backend implemented in this PR.

Compat consequence: Mooncake requires Julia ≥ 1.10, so `julia = "1"` in
`Project.toml` must become `julia = "1.10"`. The dev machine runs Julia 1.12.6.
Do NOT touch the `TemporalGPs = "0.5 - 0.6.7"` pin; it is unrelated (see commit
dd34be2).

## Backend seam design (implemented in Phase 2)

Goal: isolate every place the AD engine is invoked behind a deliberately tiny
internal interface, so swapping engines is a matter of implementing two
functions, not editing the optimizers. This is internal plumbing, not public
API; resist generalizing it beyond:

```julia
abstract type ADBackend end
struct MooncakeBackend <: ADBackend end

"""Build an opaque, backend-specific gradient cache for loss `l` at parameters
`θ`. `θ` is either a flat `Vector{Float64}` (Optim and error-estimation paths)
or the nested `Vector{<:AbstractArray}` used by the Adam path (entries may be
`SubArray`s or vectors of arrays)."""
prepare_gradient(b::ADBackend, l, θ)

"""Return `(val::Real, ∂θ)` where `∂θ` mirrors `θ` as plain nested Float64
arrays (callers index and broadcast over it elementwise)."""
value_and_gradient!(cache, l, θ)
```

Implementation notes:

- Put the interface and the Mooncake implementation together in one new file,
  `src/ad_backend.jl`, which replaces `src/Nabla_extension.jl` in the includes
  of `src/StellarSpectraObservationFitting.jl`. The Mooncake implementation
  owns the tangent-to-plain-arrays conversion helper (Phase 2 step 3) and the
  `Mooncake.@from_rrule` imports (Phase 2 step 1), so the rest of `src/` stays
  engine-free: the model and prior files define `ChainRulesCore.rrule`s only
  and never mention Mooncake.
- Thread a `backend::ADBackend=MooncakeBackend()` keyword through the workspace
  constructors (`TotalWorkspace`, `FrozenTelWorkspace`, `OptimSubWorkspace` via
  `opt_funcs`) and `estimate_σ_curvature_helper`. No global backend state.
- Caches are stateful and not thread-safe: one cache per (loss, θ-structure)
  pair, rebuilt if θ's structure changes. Each `AdamSubWorkspace` builds its
  own, matching how Nabla's compiled `gl` is handled today.
- DifferentiationInterface.jl was considered instead of a hand-rolled seam. Its
  `AutoMooncake`/`AutoEnzyme` types would fit the flat-vector paths, but its
  operator contract is built around `x::AbstractArray{<:Number}`, and the Adam
  path's θ (nested vectors of arrays and views) is outside what DI promises.
  The two-function seam above is ~30 lines and avoids contract gymnastics.
  Revisit DI if SSOF's parameters are ever flattened everywhere.

What a future `EnzymeBackend` would require — record this as a comment block at
the top of `src/ad_backend.jl`, since it is out of scope for this PR and the
reasons are easy to lose:

- De-aliasing work. The workspaces alias θ with the model arrays by design
  (`vec(lm)` at `model_functions.jl:417` returns the model's own arrays, so
  e.g. `θ[2][3] === om.star.lm.μ`), and the loss closures capture `om`, `o`,
  and `d`. Enzyme requires declaring captured data `Const` or `Duplicated`; a
  `Const` closure holding arrays that alias differentiated arguments is
  undefined behavior (typically "Constant memory is stored to a differentiable
  variable" errors). Supporting Enzyme means either annotating the closure
  `Duplicated` with a full shadow of the captured structures, or restructuring
  the losses so no differentiated array is reachable through captured state.
- The DPCA in-loss mutation (Phase 2 step 6); the hoist resolves it.
- Importing the three rrules via `Enzyme.@import_rrule` and validating them.
- Adding `EnzymeBackend()` to the `backends_to_test` list in the Phase 3
  gradient cross-check testset, which is the acceptance gate for any backend.
- Enzyme should arrive as a package extension (`ext/` + weakdep) so users who
  stay on Mooncake don't pay Enzyme's load and compile cost.

## Phase 0 — Baseline (no code changes)

1. Run the existing tests and confirm they pass:
   `julia --project=. -e 'using Pkg; Pkg.test()'`
2. Run the end-to-end example (`julia examples/example.jl`; the JLD2 data is
   present in `examples/data/`). Save the resulting RVs and final loss values
   somewhere outside the repo (e.g. `$TMPDIR/ssof_baseline/`) for later comparison.
3. Record baseline gradient values and timings: write a throwaway script that
   builds a `TotalWorkspace` from the example data (follow `examples/example.jl`
   up through `ModelWorkspace` creation), evaluates `mws.total.gl(mws.total.θ)`,
   and saves (a) the loss value, (b) the flattened gradient, and (c) a
   `@benchmark`/`@elapsed` timing of repeated gradient calls. Do the same for an
   `OptimTotalWorkspace`. Keep this script in `$TMPDIR`; it is for verification,
   not the repo.

## Phase 1 — Express the custom sensitivities as ChainRulesCore rrules

Add `ChainRulesCore` to `[deps]`/`[compat]`. Nabla stays in place during this
phase; the rrules are additions, not replacements.

1. In `src/model_functions.jl`, next to the existing `Nabla.∇` overload at
   line 163, add:

   ```julia
   function ChainRulesCore.rrule(::typeof(spectra_interp),
           model_flux::AbstractMatrix, rvs::AbstractVector, sih::StellarInterpolationHelper)
       y = spectra_interp(model_flux, rvs, sih)
       function spectra_interp_pullback(ȳ)
           # body of the existing Nabla.∇ method at model_functions.jl:163-181,
           # with `ȳ` in place of ȳ and unthunk(ȳ) if needed
           return NoTangent(), ȳnew, NoTangent(), NoTangent()
       end
       return y, spectra_interp_pullback
   end
   ```

   IMPORTANT — the rrule must return a real tangent for `rvs`, not `NoTangent()`.
   Under Nabla, the custom sensitivity (mask `[true, false, false]`) only fired
   when `model_flux` was tracked and `rvs` was a plain vector (the `l_telstar`
   loss). When `rvs` was itself a tracked parameter (Wobble `l_total` and `l_rv`,
   i.e. the default mode optimizing RVs), dispatch missed the typed method and
   fell through the untyped fallback at `model_functions.jl:160-161` to
   `spectra_interp_nabla`, where Nabla's tape differentiated w.r.t. `rvs` the
   slow way. A rule registered with Mooncake intercepts *every* call regardless
   of which arguments need gradients, so omitting the `rvs` tangent would
   silently zero the RV gradients in the default mode. Copy the `model_flux`
   pullback math verbatim from the `Nabla.∇` body, and add the `rvs` pullback:
   with `ratios = (c₀ .+ rv_to_D(rvs)') ./ step` and
   `rv_to_D(v) = log1p.(-v ./ light_speed_nu)` (`model_functions.jl:42`),

   ```
   r̄vs[j] = Σ_k ȳ[k,j] * (model_flux[lower_inds_p1] - model_flux[lower_inds])[k,j]
                       * D′(rvs[j]) / sih.model_log_λ_step
   D′(v) = -1 / (light_speed_nu * (1 - v / light_speed_nu))
   ```

   Verify both tangents against FiniteDifferences before relying on them.

2. Same treatment for the sparse-LSF method (`model_functions.jl:967-969`):
   pullback is `hcat([interp_helper[i]' * view(ȳ, :, i) for i in axes(model, 2)]...)`.

3. In `src/prior_gp_functions.jl`, replace the commented draft at lines 324-334
   with a working rrule for `gp_ℓ_precalc`: primal calls `gp_ℓ(x, A_k, Σ_k)`,
   pullback returns `ȳ .* Δℓ_precalc(Δℓ_coeff, x, A_k, Σ_k, H_k, P∞)` for arg 2
   and `NoTangent()` for the rest. Keep (move) the existing all-caps warning that
   this is only valid for the default `H_k`, `P∞`, `σ²_meas`.

4. Verify with ChainRulesTestUtils in a new testset (small random inputs, sizes
   like the existing test at `test/runtests.jl:53-65`):
   `test_rrule(spectra_interp, B, As ⊢ NoTangent(), ...)` — mark the `sih` and
   sparse-helper arguments non-differentiable, but let `test_rrule` exercise the
   `rvs` tangent of the `sih` method. For `gp_ℓ_precalc`,
   compare the pullback against `est_∇` finite differences (the helper already in
   `test/runtests.jl:10-20`) rather than `test_rrule`, since the registered
   gradient is only exact for default kwargs and `test_rrule` perturbs everything.

Verification: existing tests still pass (Nabla untouched), new rrule tests pass.

## Phase 2 — Swap the gradient engine to Mooncake

Add `Mooncake` to `[deps]`/`[compat]`.

1. Create `src/ad_backend.jl` per the "Backend seam design" section: the
   `ADBackend` interface, the `MooncakeBackend` implementation, and the rule
   imports. Import the three rules into Mooncake there (not in the model
   files), e.g.

   ```julia
   Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), Matrix{Float64}, Vector{Float64}, StellarInterpolationHelper}
   ```

   Check the installed Mooncake version's docs for the exact macro signature and
   how abstract argument types are handled; if a signature is rejected, write a
   native `Mooncake.@is_primitive`/`rrule!!` wrapper for that method. Confirm
   with `Mooncake.TestUtils.test_rule` that each rule is actually being hit
   (and not silently bypassed in favor of Mooncake deriving its own rule —
   e.g. by checking `Mooncake.is_primitive` on the signature).

2. `AdamSubWorkspace` (`src/optimization_functions.jl:383-403`): replace
   `gl = ∇(l; get_output=true)` with the seam:

   ```julia
   cache = prepare_gradient(backend, l, θ)
   gl = θ -> value_and_gradient!(cache, l, θ)
   ```

   where `backend` is the new keyword threaded down from the workspace
   constructors, defaulting to `MooncakeBackend()`. Inside the Mooncake
   implementation, `Mooncake.value_and_gradient!!` returns `(val, (∂l, ∂θ))`
   where `∂l` is the tangent of the closure itself — discard it and return
   `∂θ` (converted per step 3). In `update!`
   (`optimization_functions.jl:411-427`), `val.val` becomes plain `val`, and
   `Δ = only(Δ)` disappears because the seam already returns `∂θ` alone.

3. Tangent structure: the Adam code (`iterate!`, `first_iterate!`,
   `speed_up_iterate!`, `AdamState!`) expects `Δ` to mirror `θ` as plain nested
   arrays of `Float64`. For `Vector{Matrix{Float64}}` entries Mooncake's tangent
   is already the same plain structure, but for `SubArray` entries (θ built via
   `downsize_view`/`vec(lm)` on views) Mooncake may return wrapper tangent types.
   Write one small recursive helper, `tangent_to_arrays(Δ)`, in
   `src/ad_backend.jl`, that maps a Mooncake tangent for (vectors of)
   arrays/views to plain arrays, and apply it inside the `MooncakeBackend`
   `value_and_gradient!` so callers only ever see plain arrays (this is part of
   the seam's contract, and the functional replacement for
   `src/Nabla_extension.jl`). Unit-test it directly
   with a θ containing a `SubArray` (mimic `only_s=true` after `downsize_view`).
   Also confirm `Iterators.flatten(Iterators.flatten(Δ))` in `AdamState!`
   (`optimization_functions.jl:363`) still iterates numbers, not wrappers.

4. `opt_funcs` (`src/optimization_functions.jl:778-794`): simplify by
   differentiating the already-composed flat function instead of round-tripping
   through `flatten`:

   ```julia
   f = loss ∘ unflatten           # flat Vector{Float64} -> Real
   cache = prepare_gradient(backend, f, flat_initial_params)
   g!(G, θflat)      -> G .= value_and_gradient!(cache, f, θflat)[2]
   fg_obj!(G, θflat) -> (val, ∂θ) = value_and_gradient!(cache, f, θflat);
                        G .= ∂θ; return val
   ```

   Return signature of `opt_funcs` changes (no more `g_nabla`/`g_val_nabla`);
   grep for its callers (`OptimSubWorkspace` at line 815 discards them already)
   and adjust. `l.val` at line 791 becomes plain `val`. Note `unflatten` here
   relies on the custom `flatten` methods in `src/flatten.jl` — those are
   ParameterHandling extensions, independent of Nabla. Keep them.

5. `src/error_estimation.jl:19`: replace `g = ∇(ℓ)` with the seam
   (`prepare_gradient`/`value_and_gradient!` on the flat `Vector{Float64}`),
   and `only(g(x))[k]` (lines 41 and 58) with the corresponding `∂x[k]`. Watch
   the multithreaded branch: gradient caches are not thread-safe, so build one
   cache per thread inside the `Threads.@threads` loop (the loop already makes
   thread-local copies of `x`).

6. DPCA semantics decision (affects `OrderModelDPCA` only, not the default
   Wobble path): `_loss_recalc_rv_basis` (`optimization_functions.jl:63-67`)
   mutates `om.rv.lm.M` from `om.star.lm.μ` inside the loss. Under Nabla, `μ`
   entered that computation as a raw closure array, so no gradient flowed through
   it. Under Mooncake, `om.star.lm.μ` is typically the *same array object* as a
   θ entry, tangents are tracked by object identity, and gradient WILL flow
   through `doppler_component_AD` into `μ`. To keep this port faithful
   (identical gradients to Nabla), hoist the basis update out of the
   differentiated region: in `loss_funcs_total(o, om::OrderModelDPCA, d)` make
   `l_total` call a plain `_loss` and instead refresh
   `om.rv.lm.M .= doppler_component_AD(om.star.λ, om.star.lm.μ)` immediately
   before the gradient evaluation (e.g. via an optional `pre!` callback field on
   `AdamSubWorkspace` invoked at the top of `update!`, a no-op by default).
   Leave a comment noting that letting the gradient flow may be *more* correct
   and is worth revisiting with Christian.

7. Remove `using Nabla` from `src/optimization_functions.jl`,
   `src/model_functions.jl`, `src/prior_gp_functions.jl`; delete the
   `@explicit_intercepts`/`Nabla.∇` blocks (the rrules now carry the math);
   delete `src/Nabla_extension.jl` and its include at
   `src/StellarSpectraObservationFitting.jl:13`; delete the untyped
   `spectra_interp` fallback at `src/model_functions.jl:160-161`. Keep
   `spectra_interp_nabla` and `gp_ℓ_nabla` as plain reference implementations
   (tests use the former); just fix their docstrings to say "index-based
   reference implementation" rather than referencing Nabla. Do not rename them
   in this PR.

Verification before moving on: package loads, `Pkg.test()` may still fail on the
Nabla-using testset (fixed in Phase 3), but the Phase 0 throwaway script rerun
with the new engine must reproduce the baseline loss exactly and the baseline
gradient to ~1e-8 relative, for both the Adam and Optim workspaces.

## Phase 3 — Tests, deps, cleanup

1. `test/runtests.jl`: drop `using Nabla`. Rework the
   "custom spectra_interp() sensitivity" testset to compare
   `Mooncake.value_and_gradient!!` of `f_custom_sensitivity` against (a) the same
   for `f_nabla` (the reference implementation, differentiated by Mooncake
   without the custom rule) and (b) `est_∇` finite differences. Add the Phase 1
   rrule testset and the `tangent_to_arrays` test.
2. Add a new testset that cross-checks gradients of the real loss closures:
   build a tiny synthetic `GenericData` + `OrderModelWobble` (a few hundred
   pixels, a handful of epochs — see how `examples/example.jl` constructs data,
   or synthesize flat-spectrum data directly) and compare Mooncake gradients of
   `l_total`, `l_total_s`, `l_telstar`, `l_telstar_s`, `l_rv`, and
   `l_frozen_tel` against `est_∇`, looping over a flattened copy of θ. Wrap the
   whole testset in a loop over `backends_to_test = [MooncakeBackend()]`: this
   list is the acceptance gate for any future backend (e.g. an
   `EnzymeBackend`), which inherits the validation by being appended here. This
   is the test that catches aliasing/mutation surprises. Keep it small enough
   to stay in `runtests.jl`.
3. `Project.toml`: remove Nabla from `[deps]` and `[compat]`; add compat entries
   for Mooncake and ChainRulesCore (use the currently-resolved majors); set
   `julia = "1.10"`. Add FiniteDifferences/ChainRulesTestUtils under `[extras]` +
   `[targets]` test target if used (note the package currently lists `Test` in
   `[deps]` rather than using extras — follow the existing pattern rather than
   restructuring).
4. Sweep comments/docstrings mentioning Nabla (`grep -rn -i nabla src/ docs/`):
   reword `model_functions.jl:440` (and optionally switch `_eval_lm` to the
   `muladd` method now that Nabla is gone — only if the Phase 2 gradient checks
   still pass and it benchmarks no slower), `prior_gp_functions.jl:128,281-282`,
   and the dev-notes comment blocks at `prior_gp_functions.jl:289-334`.
5. Update the "Constraints worth knowing" bullet about Nabla in `CLAUDE.md`.

Verification: `julia --project=. -e 'using Pkg; Pkg.test()'` fully green on a
fresh resolve (delete `Manifest.toml` and re-instantiate to prove the compat
entries are right).

## Phase 4 — End-to-end verification and benchmarks

1. Rerun `julia examples/example.jl` and compare RVs and uncertainties to the
   Phase 0 baseline. Adam is deterministic given identical inits, so RVs should
   agree to near machine precision for the Wobble path; small drifts mean a
   gradient discrepancy — stop and bisect rather than loosening the comparison.
2. Rerun the timing script. Acceptance: Mooncake gradient evaluation within
   ~1.5× of the Nabla baseline (expect it to be faster once compiled). Report
   the first-call compile time separately; Mooncake's rule derivation can make
   the first gradient call noticeably slower than Nabla's tracing, which is
   acceptable but worth recording in the PR description.
3. PR description should include: baseline-vs-new gradient agreement numbers,
   timing table, the DPCA semantics note from Phase 2 step 6, and the Julia
   compat bump.

## Known risks

- The `spectra_interp` rrule replacing BOTH the Nabla custom sensitivity AND the
  tape-differentiated `spectra_interp_nabla` fallback path: it must supply the
  `rvs` tangent (see Phase 1 step 1) or RV gradients silently vanish in the
  default Wobble mode. The Phase 3 loss-closure finite-difference testset must
  include `l_total` (Wobble) and `l_rv` specifically to guard this.
- `Mooncake.@from_rrule` signature restrictions (abstract types, kwargs). The
  fallback is a native Mooncake rule; the analytic math is unchanged either way.
- SubArray-containing θ in the `only_s` workspaces — covered by the
  `tangent_to_arrays` unit test; do not skip it.
- Aliasing between θ entries and arrays reachable through the loss closures
  (e.g. `l_total_s` builds `[om.tel.lm.M, total_s[1], om.tel.lm.μ]` mixing
  closure arrays with parameters). Mooncake tracks tangents by object identity,
  so this should be correct, but it is exactly what the Phase 3 step 2 finite
  difference testset exists to confirm.
- Mooncake gradient caches assume stable types: a cache built for one θ must not
  be reused after θ's structure changes (fine here — each `AdamSubWorkspace`
  builds its own) and is not thread-safe (relevant only to
  `error_estimation.jl`).
- Julia compat bump to 1.10 drops support for older Julia; flag in the PR for
  Christian since this is his package and docs deploy from `master`.

## GPU outlook (future work, NOT this PR)

Recorded here so the rationale isn't lost; nothing in this section is to be
implemented during the Nabla removal.

Ordering: any GPU port must come after this migration, not before or alongside.
Nabla cannot trace GPU arrays or kernels, so GPU-ified code would be
undifferentiable until the swap lands, and combining the two changes would
break this plan's core verification strategy (strict agreement of new gradients
with the Nabla baseline — two moving pieces make discrepancies unattributable).

How the loss splits for GPU purposes:

- GPU-friendly: the dense `M * s` products, the elementwise χ² and `exp.`
  broadcasts (`_χ²_loss`, `_eval_lm`), the sparse `t2o`/`b2o`/LSF matvecs, and
  the forward pass of `spectra_interp` (a gather). The `spectra_interp`
  pullback is a scatter-add, currently a scalar loop
  (`model_functions.jl:163-181`); a GPU version needs a KernelAbstractions
  kernel or an NNlib-style `scatter!`.
- GPU-hostile: the GP prior. `gp_ℓ` and the `gp_Δℓ_helper_γ` pass inside the
  precalculated gradient are Kalman recursions, strictly sequential over ~10⁴
  pixels, evaluated for the template and every feature vector. Moving them to
  GPU requires reformulating the filter as an associative scan
  (Särkkä/García-Fernández-style parallel-scan Kalman) — a research-grade
  rewrite. Short of that, the GP prior stays on CPU, Amdahl's law caps the
  speedup, and host-device transfers are paid every loss evaluation.

Key architectural consequence of this plan that keeps the door open: because
every nontrivial operation in the loss is either standard linear algebra (which
has ChainRules coverage) or one of the three custom `rrule`s, a GPU loss would
not necessarily require a GPU-capable AD engine. If the custom rrules grow
GPU-array-dispatched methods (gather/scatter pullbacks), the engine never has
to look inside a kernel, so even Mooncake could drive a GPU loss. The
alternative route — differentiating hand-written KernelAbstractions kernels
directly — is Enzyme territory (proven in e.g. Oceananigans adjoints), which
would first require the `EnzymeBackend` de-aliasing work described in the
backend seam section.

Cheap habits to follow during THIS port so a GPU effort isn't penalized later:
keep the `rrule` signatures on abstract array types (concrete types belong
only in the `@from_rrule` import lines), prefer array/broadcast formulations
over scalar loops in any pullback code being touched anyway, and don't bake
`Vector{Float64}`/`Matrix{Float64}` into the seam's contract beyond what the
current optimizers require.

Decision input: whether GPU is worth pursuing at all should be decided from the
Phase 0 profiling data (expectation: sparse interpolation matvecs and the GP
prior dominate, and the latter does not move to GPU cheaply). Note also that
SSOF's natural parallelism is across spectral orders — fully independent
problems that saturate CPU cores or cluster nodes with zero code changes —
which may remain the better throughput axis than per-order GPU acceleration.
