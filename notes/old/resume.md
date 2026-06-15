# Resuming the Nabla → Mooncake migration (`remove-nabla` branch)

## What's done

- **Phase 1** (`3be0b5c`): `ChainRulesCore.rrule` for `spectra_interp` (both variants) and `gp_ℓ_precalc` in `model_functions.jl` / `prior_gp_functions.jl`.
- **Phase 2–3** (`835a8bf`): `src/ad_backend.jl` created with the `ADBackend`/`MooncakeBackend` seam (`prepare_gradient` + `value_and_gradient!`); all three call sites wired (`AdamSubWorkspace`, `opt_funcs`, `error_estimation.jl`); tests updated; Nabla removed from `Project.toml`.

## What remains

### 1. Delete `src/Nabla_extension.jl`

It is no longer included in the module (see `CLAUDE.md`). Ask user before deleting.

### 2. Phase 4 — end-to-end verification

Run the full test suite on a fresh resolve:

```bash
rm Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

### 3. Phase 4 — benchmarks

`benchmark_ad.jl` is at the repo root (untracked). It needs JLD2 files at
`examples/data/results.jld2` and `examples/data/data.jld2`. Run:

```bash
julia benchmark_ad.jl
```

Results write to `benchmark_results_remove-nabla.txt`. Acceptance: Mooncake
steady-state gradient within ~1.5× of the Nabla baseline. First-call compile
overhead is expected and acceptable — report it separately.

Also rerun `julia examples/example.jl` and confirm RVs agree with the
master-branch baseline to near machine precision (Adam is deterministic).

### 4. Open PR

PR description must include:
- Gradient agreement numbers (new vs. baseline)
- Timing table from the benchmark
- DPCA semantics note: `_loss_recalc_rv_basis` hoist (Phase 2 step 6 in `nabla-removal-plan.md`)
- Julia compat bump to 1.10 (flag for Christian, since docs deploy from `master`)

## Key files

| File | Role |
|------|------|
| `src/ad_backend.jl` | AD seam: `ADBackend`, `MooncakeBackend`, `@from_rrule` imports |
| `src/Nabla_extension.jl` | Orphan — pending deletion |
| `nabla-removal-plan.md` | Full original plan with phase details and known risks |
| `benchmark_ad.jl` | Phase 4 benchmarking script (untracked, repo root) |
