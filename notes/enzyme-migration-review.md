# Review: Nabla.jl → Enzyme.jl migration on `try_enzyme`

Reviewer: Claude Opus 4.7, 2026-06-14
Branch reviewed: `try_enzyme` (HEAD `1ec3c56`)
Scope: identify remaining issues from the AD-backend switch, out-of-date
functions, untested code paths, and refactoring opportunities.

## TL;DR

The migration is functionally complete but carries "two-engine" complexity.
Every AD-critical loss has *two* gradient paths (a ChainRules rrule for
Mooncake's `@from_rrule` AND a hand-written Enzyme
`augmented_primal`/`reverse`). The duplication is intentional — Enzyme 0.13
can't import the rrules due to japi3 calling convention (EnzymeAD/Enzyme.jl
#2707) — but it's the single largest source of future maintenance risk.

The main blockers to merging:
- Inconsistent backend defaults across call sites
- Phase 6 (extension wiring) not done; Phase 7 (benchmarks) not landed
- DPCA semantics change is not user-visible documented
- Several functions never exercised under Enzyme

---

## 1. Confirmed dead code from the Nabla era

| Item | Location | Action |
|---|---|---|
| `src/Nabla_extension.jl` | repo file | **Delete** — not in module includes; only contains `Nabla.zerod_container` override. CLAUDE.md and memory both flag this. |
| `spectra_interp_nabla` | `model_functions.jl:152–155` | **Delete** — no callers; the only references are commented out at lines 980–983. |
| `gp_ℓ_nabla` | `prior_gp_functions.jl:130–155` | **Delete** — no callers; only commented references in test and `prior_gp_functions.jl:449–455`. |
| Commented Nabla blocks | `prior_gp_functions.jl:113–120`, `:449–455`, `:478–488`; `model_functions.jl:980–983` | **Delete** — kept "just in case", but git history preserves them. |

## 2. Latent bug in `DPCA_functions.jl`

`doppler_component_log_AD` at `DPCA_functions.jl:76–77` is defined
recursively against *itself*:

```julia
doppler_component_log_AD(λ::AbstractVector{T}, flux::Vector{T}) where {T<:Real} =
	doppler_component_log_AD(λ, flux) ./ flux   # ← infinite recursion
```

Surely meant `doppler_component_AD(λ, flux) ./ flux` (matching the
non-AD sibling on line 58). The function is never called, so the bug
hasn't fired — but it should be fixed or deleted.

## 3. The `_AD` suffix is now stale naming

`simple_derivative_AD` and `doppler_component_AD` were named for
Nabla-friendliness (array-comprehension form vs. mutation). Mooncake and
Enzyme both handle them. The "AD" suffix is now meaningless — consider
renaming (`doppler_component_immutable` or just folding into
`doppler_component`).

## 4. Functions not exercised against Enzyme

Test coverage on `try_enzyme` includes nice gradient cross-checks but
stops short of running real optimizations through Enzyme. The following
are **untested under the new AD**:

- **Optim path (`OptimTelStarWorkspace`, `OptimTotalWorkspace`)** —
  `opt_funcs` (`optimization_functions.jl:862`) still defaults to
  `MooncakeBackend()`. No test runs `Optim.optimize` through Enzyme.
- **`loss_funcs_telstar`** (`optimization_functions.jl:113–143`) — only
  called from `OptimTelStarWorkspace`, never gradient-checked under
  Enzyme.
- **`estimate_σ_curvature(...; use_gradient=true)`** — hardcoded
  `MooncakeBackend()` at `error_estimation.jl:19, 35` (not even a
  kwarg). Not exposed as user-tunable.
- **`improve_model!` / `improve_initial_model!` / `finalize_scores!` /
  `fit_regularization!` / `calculate_initial_model`** — end-to-end
  pipeline. Only `examples/example.jl` runs them, but `example.jl`
  isn't part of `Pkg.test()`. Should at least have a smoke-test pinging
  each on a tiny synthetic dataset.
- **`estimate_σ_bootstrap`** — re-fits via `improve_model!`, so it
  *does* hit Enzyme, but not unit-tested.
- **`OrderModelDPCA` end-to-end** — `calculate_initial_model` is gated
  to `OrderModelWobble` (`optimization_functions.jl:1325` types
  `oms::Array{OrderModelWobble}`; `:1307` has `# TODO: Make this work
  for OrderModelDPCA`). DPCA is only covered by the one new unit test.
- **`LSFData` + Enzyme** — `spectra_interp(model, lsf::SparseMatrixCSC)`
  has a dedicated Enzyme rule (`ad_backend.jl:266–291`), but no test
  specifically exercises an `LSFData` loss under Enzyme.
- **First iterate / speed-up paths** (`first_iterate!`,
  `speed_up_iterate!`) — only triggered in long Adam runs; not
  directly unit-tested.

## 5. Inconsistent backend defaults

The choice of default backend is *split* across the codebase:

| Call site | Default | File |
|---|---|---|
| `AdamSubWorkspace(θ, l; backend=…)` | `MooncakeBackend()` | `optimization_functions.jl:445` |
| `TotalWorkspace(...; backend=…)` | `EnzymeBackend()` | `optimization_functions.jl:648` |
| `FrozenTelWorkspace(...; backend=…)` | `EnzymeBackend()` | `optimization_functions.jl:709` |
| `opt_funcs(loss, pars; backend=…)` | `MooncakeBackend()` | `optimization_functions.jl:862` |
| `estimate_σ_curvature_helper` | `MooncakeBackend()` (hardcoded, no kwarg) | `error_estimation.jl:19, 35` |

The two workspace constructors override the lower-level default. This
works but is confusing. Two cleanup options:

1. **Pick one default** (Enzyme) at the seam (`AdamSubWorkspace`,
   `opt_funcs`) and remove the workspace-level overrides.
2. **Thread `backend` as a kwarg everywhere** so the user/example can
   switch with one knob, and remove all hardcoded references.

Either is better than today.

## 6. Outstanding migration-plan items (per `enzyme-migration-plan.md`)

The plan lists Phase 6, 7 as not done:

- **Phase 6: Convert Enzyme to a package extension.** Enzyme is
  currently in `[deps]`, not `[weakdeps]`/`ext/`. Mooncake-only users
  pay Enzyme's load + compile cost on every `using SSOF`. This was a
  stated pre-merge requirement.
- **Phase 7: Benchmark and acceptance.** The plan's hard target
  (steady-state Adam ≤ 400 ms, hard floor 600 ms) — has this been
  measured on `try_enzyme`? `benchmark_ad.jl` exists at the root but
  is untracked and not part of CI.
- **Phase 4 step 5: DPCA semantics doc.** The plan called for a
  `# Migration note` comment block in `loss_funcs_total(o,
  om::OrderModelDPCA, d)` explaining that the gradient now flows
  through `om.star.lm.μ → doppler_basis`, which differs from
  pre-port Nabla. There's a single-line `# Enzyme-safe:` comment but
  no semantics note.

## 7. Out-of-date / dormant code linked to the AD seam

- **`_loss_recalc_rv_basis`** at `optimization_functions.jl:81–86` is
  the *old* helper that recomputed the Doppler basis externally. The
  `l_total` for DPCA now inlines this logic. The helper has no
  callers (grep confirms). **Either delete or document it as
  kept-for-test-purposes.**
- **`Base.copy(x::Base.TwicePrecision)` at `ad_backend.jl:36`** is
  type-piracy needed by Mooncake. If/when Mooncake is removed
  (Phase 8), delete this.
- **`tangent_to_arrays`** recursive helper (`ad_backend.jl:78–84`)
  only exists to flatten Mooncake's `Any`-typed tangent containers.
  Pure Mooncake support cost.
- **`_eval_lm_inner` + `Val{log_lm}`** indirection
  (`model_functions.jl:886–889`) exists *only* because
  `Mooncake.@from_rrule` can't dispatch through `Core.kwcall`. Enzyme
  tracing handles `_eval_lm` natively. If Mooncake goes, revert to the
  original kwarg form.
- **The double-registration of rrules** (concrete + abstract type
  signatures, `ad_backend.jl:44–68`) is a Mooncake-specific
  workaround for abstract struct-field types. Each adds compile time.

## 8. Aliasing / activity concerns

- **`Enzyme.EnzymeRules.inactive_type(::Type{<:AbstractDict}) = true`**
  at `ad_backend.jl:123` is **global** — affects every `AbstractDict`
  in any package that loads SSOF, not just `reg_tel`/`reg_star`.
  Probably fine here, but worth a comment explaining the scope risk.
  Narrowing to `Dict{Symbol, <:Real}` would be safer.
- **`bary_rvs` and `t2o`** are *shared by reference* between `om` and
  `view`/`copy` constructions (`model_functions.jl:781, 783–784`).
  They're constants in the science problem, so this is correct — but
  no `inactive_type` declaration protects them, so Enzyme will still
  build shadow copies of them.
- **`StellarInterpolationHelper`** has `inactive_type`
  (`ad_backend.jl:117`) — good. The corresponding fields on the DPCA
  side (`b2o::AbstractVector{<:SparseMatrixCSC}`, `bary_rvs`) don't
  have analogous declarations.

## 9. Style / organization

- **`is_time_variable`** is *defined* at
  `optimization_functions.jl:1226–1227` but *used* much earlier (line
  117). Works only because of Julia late binding. Move declarations
  to the top of the file.
- **`TotalWorkspace` / `FrozenTelWorkspace`** constructors share ~30
  lines of nearly identical `build_θ`/`build_l` branching
  (`optimization_functions.jl:657–675` vs `716–738`). Factor out a
  `_build_total_signature(only_s, is_tel_tv, is_star_tv)` helper.
- **`_lm_tuple` vs `vec(lm)`** duality (Adam takes Tuple, Optim takes
  Vector) is intentional but undocumented at the seam. A short
  docstring above `_lm_tuple` explaining "why two representations"
  would prevent future contributors from collapsing them.
- **Five duplicated `if is_tel_time_variable / is_star_time_variable`
  branches** across `l_total_s`, `l_frozen_tel`, `l_frozen_tel_s`,
  `loss_funcs_total`, `loss_funcs_frozen_tel` could be condensed into
  a helper.

## 10. Repository hygiene

`git status` shows uncommitted artifacts that should be triaged before
PR:

- `HeirarchicalGPQwenTest.jl` — typo in filename; appears unrelated
  to SSOF. Delete or move out.
- `Manifest.toml.aside112`, `Manifest.toml.bak`,
  `Manifest.toml.julia111.bak`, `Manifest.toml.julia112.bak` — dev
  artifacts.
- `enzyme-julia111-plan.md`, `enzyme-migration-plan.md`,
  `mooncake-perf-plan.md`, `nabla-removal-plan.md`, `resume.md` —
  work notes at root level. Either move to a `notes/` directory in
  the repo or delete (git history preserves them).
- `benchmark_ad.jl` — useful but untracked; either commit (with its
  JLD2 deps documented) or delete.
- `examples/data/data.jld2.bak` — backup, presumably no longer
  needed.

## 11. `Project.toml` compat fields worth a look

- `Mooncake = "0.4, 0.5.31"` — odd point-pin on `0.5.31`. Was this
  deliberate?
- `julia = "1.10"` — memory says Mooncake 0.4.80 fails to precompile
  on Julia 1.12, so testing pinned to 1.11.2. Does CI matrix reflect
  this? Worth verifying.
