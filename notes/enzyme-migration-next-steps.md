# Recommended next steps — Enzyme.jl migration

Companion to `enzyme-migration-review.md` (full review with citations).
Branch: `try_enzyme`. Sequence below is rough priority order.

## 1. Delete confirmed dead Nabla code

| File | Action |
|---|---|
| `src/Nabla_extension.jl` | Delete file (not in module includes) |
| `src/model_functions.jl:152–155` | Delete `spectra_interp_nabla` |
| `src/model_functions.jl:980–983` | Delete commented `spectra_interp_nabla` block |
| `src/prior_gp_functions.jl:113–155` | Delete `gp_ℓ_nabla` and commented siblings |
| `src/prior_gp_functions.jl:449–488` | Delete commented Nabla testing block |

Verify by `grep -n "Nabla\|nabla" src/ test/` returning empty (except
historical comments in plan files, if those are retained).

## 2. Fix or delete `doppler_component_log_AD`

`DPCA_functions.jl:76–77` is infinite-recursive. Either:

- Fix it: change RHS to `doppler_component_AD(λ, flux) ./ flux`, or
- Delete it (uncalled).

Add a unit test if fixed.

## 3. Add unit tests for previously untested Enzyme paths

Suggested additions to `test/runtests.jl`:

- A `@testset "EnzymeBackend through opt_funcs + Optim.optimize"` that
  runs a few L-BFGS iterations on `loss_funcs_telstar` and checks
  convergence.
- A `@testset "improve_model! end-to-end smoke test"` on a tiny
  synthetic Wobble dataset (10–20 model pixels, 3 observations,
  1 component each).
- A `@testset "LSFData Enzyme gradient"` that exercises the
  `spectra_interp(model, lsf::SparseMatrixCSC)` Enzyme rule with a
  real `LSFData` loss closure.
- (Optional) one short `@testset "DPCA improve_model! smoke test"`.

Gate on Mooncake↔Enzyme gradient agreement + FD where small enough.

## 4. Unify backend defaults

Pick one of two strategies and apply consistently:

**Option A (recommended): one default at the seam.**

- `AdamSubWorkspace(...; backend::ADBackend=EnzymeBackend())`
  (`optimization_functions.jl:445`)
- `opt_funcs(...; backend::ADBackend=EnzymeBackend())`
  (`optimization_functions.jl:862`)
- Remove the `backend=EnzymeBackend()` override from `TotalWorkspace`
  / `FrozenTelWorkspace` (lines 648, 709) — they inherit from the
  seam.
- Thread `backend` kwarg through `estimate_σ_curvature` /
  `estimate_σ_curvature_helper` (`error_estimation.jl:9, 19, 35`).
  Default: `EnzymeBackend()`.

**Option B: keep Optim on Mooncake but document it.** If `opt_funcs`
deliberately stays on Mooncake (because L-BFGS calls are infrequent
and Mooncake's per-call cost is lower for the small-rank Optim
problem), add a `# NOTE:` comment at the seam stating that
deliberately. Apply the consistency rule only to the workspaces.

Pick A unless benchmark numbers later show Optim+Enzyme regressing
badly.

## 5. Document DPCA semantics change

Add a comment block above
`loss_funcs_total(o::Output, om::OrderModelDPCA, d::Data)`
(`optimization_functions.jl:156`):

> Migration note: the Doppler basis is now recomputed inline from
> `om.star.lm.μ` inside `l_total`, so the gradient flows
> ∂loss/∂μ → ∂doppler_basis. This differs from the pre-port Nabla
> path, which treated the basis as a constant (no μ → basis term).
> The Wobble path is unaffected.

Also note this in `docs/src/opt.md` (the docstring page that mirrors
`optimization_functions.jl`).

## 6. Tighten `inactive_type` on Dict

In `src/ad_backend.jl:123`, change

```julia
Enzyme.EnzymeRules.inactive_type(::Type{<:AbstractDict}) = true
```

to

```julia
Enzyme.EnzymeRules.inactive_type(::Type{<:Dict{Symbol, <:Real}}) = true
```

This still covers `reg_tel` / `reg_star` (both
`Dict{Symbol, T} where T<:Real`) without globally piracy-declaring
every dict in downstream packages inactive.

## 7. Benchmark vs. Nabla baseline (Phase 7 of the plan)

Run `benchmark_ad.jl` from the repo root (currently untracked). The
expected reference points, per `enzyme-migration-plan.md` Phase 7 and
`mooncake-perf-plan.md`:

| Engine | Steady-state Adam | Notes |
|---|---|---|
| Nabla (master) | 286 ms | original baseline |
| Mooncake (`remove-nabla`) | 919 ms | pre-Enzyme; regression vs. Nabla |
| Enzyme (`try_enzyme`) | target ≤ 400 ms; hard floor 600 ms | this PR |

Also measure:
- ModelWorkspace first-call construction (target ≤ 90 s; investigate
  if > 180 s — likely a custom rule missed)
- `finalize_scores!` first-call (target ≤ 300 s)

Output the table into the PR description.

## 8. Convert Enzyme to a package extension (Phase 6)

Required before merge per the plan.

- Move Enzyme from `[deps]` to `[weakdeps]` in `Project.toml`.
- Add `[extensions] SSOFEnzymeExt = "Enzyme"`.
- Move Enzyme-using code from `src/ad_backend.jl` into
  `ext/SSOFEnzymeExt.jl`. Keep the `EnzymeBackend` struct as a
  placeholder in the main module so dispatch works without the
  extension loaded.
- Add a graceful-fallback `prepare_gradient(::EnzymeBackend, ...)`
  stub in the main module that either errors clearly or falls back
  to Mooncake with an `@info` (plan recommends the latter).
- Verify `Pkg.test()` works in a Manifest-less resolve.

## 9. Repository hygiene before PR

- Triage untracked files at repo root (`HeirarchicalGPQwenTest.jl`,
  `Manifest.toml.*.bak`, `enzyme-julia111-plan.md`,
  `examples/data/data.jld2.bak`).
- Decide whether planning docs (`enzyme-migration-plan.md`,
  `mooncake-perf-plan.md`, `nabla-removal-plan.md`, `resume.md`,
  this file) belong in repo (`notes/` dir) or are deleted with their
  content summarized in the PR description.
- Decide whether `benchmark_ad.jl` is committed (with JLD2 data
  preconditions documented) or kept local.

## 10. Plan Phase 8 (Mooncake removal) as a follow-up issue

After a few weeks of Enzyme stability, opening a separate issue/PR to
remove Mooncake unlocks deletion of:

- `Base.copy(::Base.TwicePrecision)` (`ad_backend.jl:36`) — type-piracy
- `tangent_to_arrays` recursive helper (`ad_backend.jl:78–84`)
- `_eval_lm_inner` + `Val{log_lm}` indirection
  (`model_functions.jl:886–889`) — revert to plain kwarg form
- All `@from_rrule` registrations (`ad_backend.jl:44–68`,
  `prior_gp_functions.jl:303–309`)
- Mooncake from `[deps]` and `[compat]`
- The dual concrete+abstract rrule signature pattern

ChainRulesCore rrule bodies stay (pure CRC, no Mooncake dependency).
Tests collapse to a single `backends_to_test = [EnzymeBackend()]`.

## Verification checklist before opening PR

- [ ] `Pkg.test()` passes on Julia 1.10 and 1.11.2
- [ ] `julia examples/example.jl` runs to completion and Wobble RVs
      match the pre-port Nabla baseline within 1e-10 relative
- [ ] DPCA divergence from Nabla baseline is measured and documented
      (expected due to μ → basis gradient)
- [ ] Benchmark table from step 7 included in PR description
- [ ] Manifest-less fresh resolve succeeds
- [ ] CI matrix updated for Julia versions matching `[compat] julia`
