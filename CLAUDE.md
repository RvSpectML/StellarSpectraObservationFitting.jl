# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

StellarSpectraObservationFitting.jl (SSOF, pronounced like "soufflé") measures radial velocities from Extremely Precise Radial Velocity (EPRV) spectrographs by building data-driven linear models for the time-variable telluric transmission and stellar spectrum, with fast GP-based regularization and optional LSF handling. The conventional import is `import StellarSpectraObservationFitting as SSOF`.

## Commands

```bash
# Run tests (all tests live in test/runtests.jl; there is no runtests_slow.jl here)
julia --project=. -e 'using Pkg; Pkg.test()'

# Build documentation (Documenter.jl)
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl

# Run the end-to-end example (activates examples/Project.toml itself; needs JLD2 data in examples/data/)
julia examples/example.jl
```

## Architecture

All code is one flat module: `src/StellarSpectraObservationFitting.jl` just includes the other files. Two files in `src/` are deliberately NOT included in the module: `_fit_SOAP_gp.jl` and `_fit_LSF_gp.jl` are standalone scripts that were used to derive the hard-coded GP hyperparameters (e.g. `SOAP_gp_params`) used in `prior_gp_functions.jl`.

The typical pipeline (see `examples/example.jl` and the docstrings it touches):

1. Wrap observations in a `Data` subtype (`LSFData` or `GenericData`, defined in `model_functions.jl`): flux, variance, `log_λ_obs` (observatory frame), `log_λ_star` (barycentric frame), optional LSF matrix.
2. `calculate_initial_model(data; ...)` (`optimization_functions.jl`) builds an `OrderModel` and does model selection over the number of telluric/stellar feature vectors (AIC-based, `model_selection_functions.jl`).
3. `ModelWorkspace(model, data)` creates an optimization workspace.
4. `fit_regularization!(mws)` (`regularization_functions.jl`) tunes regularization strengths via train/test split.
5. `improve_model!(mws)` optimizes the full model.
6. `estimate_σ_curvature(mws)` or `estimate_σ_bootstrap(mws)` (`error_estimation.jl`) gives RV and score uncertainties.

Key types:

- `OrderModel` (abstract, `model_functions.jl`) has two concrete variants: `OrderModelWobble` (default; RVs from linear interpolation of the stellar model, like wobble) and `OrderModelDPCA` (RVs from Doppler-constrained PCA, `DPCA_functions.jl`). Each holds telluric and stellar `Submodel`s, regularization coefficient `Dict`s, and interpolation helpers: `t2o` (sparse matrices, telluric → observed frame) and `b2o` (barycentric → observed; a `StellarInterpolationHelper` for Wobble, sparse matrices for DPCA).
- A `Submodel` wraps a `LinearModel`: `FullLinearModel` (template μ + feature vectors M × scores s), `BaseLinearModel` (no μ), or `TemplateModel` (μ only). Models can be in log or linear flux space (`lm.log`).
- `ModelWorkspace` (abstract, `optimization_functions.jl`) splits into `AdamWorkspace` variants (`TotalWorkspace`, `FrozenTelWorkspace`; custom Adam implementation in the same file) and `OptimWorkspace` variants (`OptimTotalWorkspace`, `OptimTelStarWorkspace`; L-BFGS via Optim.jl).

## Constraints worth knowing

- Automatic differentiation uses Mooncake.jl. The AD seam is `src/ad_backend.jl`: `prepare_gradient(b, l, θ)` + `value_and_gradient!(cache, l, θ)`. Custom sensitivities for `spectra_interp` and `gp_ℓ_precalc` are registered via `ChainRulesCore.rrule` (in `model_functions.jl` and `prior_gp_functions.jl` respectively) and imported into Mooncake with `@from_rrule`. `src/Nabla_extension.jl` is an orphaned file (no longer included in the module) pending deletion.
- `TemporalGPs = "0.5 - 0.6.7"` is pinned to avoid precompilation warnings (see commit dd34be2). `prior_gp_functions.jl` reimplements the state-space GP likelihood (and analytic gradients, optionally sparse/precalculated) rather than calling TemporalGPs directly, for speed.
- `continuum_functions.jl` and `rassine.jl` (a port of RASSINE) handle continuum normalization; `mask_functions.jl` handles bad-pixel/edge masking (e.g. `mask_bad_edges!`).
- Wavelengths are handled as log-wavelength almost everywhere (`log_λ`), and Doppler shifts as `rv_to_D` factors.
- This is Christian Gilbertson's package; docs deploy from `master` via GitHub Actions to christiangil.github.io. Docstring markdown pages live in `docs/src/` and are organized by source file (e.g. `opt.md` ↔ `optimization_functions.jl`).
