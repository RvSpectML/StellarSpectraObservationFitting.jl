# Handoff prompt — Enzyme migration cleanup (steps 2–6) + benchmarks

Paste the block below into a fresh Claude Code session in the
`StellarSpectraObservationFitting.jl` repo (branch `try_enzyme`). It
is self-contained — the agent has no prior conversation context.

---

## Prompt to paste

You are picking up work on `StellarSpectraObservationFitting.jl`
(SSOF), branch `try_enzyme`. The repo has migrated its AD backend
from **Nabla.jl → Mooncake.jl → Enzyme.jl** (current head: commit
`1ec3c56`). The migration is functionally complete but has accumulated
two-engine complexity and dead Nabla-era code. A previous review
produced two files at the repo root that you must read before
starting:

- `enzyme-migration-review.md` — the full review with file/line
  citations.
- `enzyme-migration-next-steps.md` — the prioritized work list.

Background context that matters:

- **Both AD backends coexist.** `MooncakeBackend` and `EnzymeBackend`
  both subtype `ADBackend` in `src/ad_backend.jl`. Mooncake is the
  in-process oracle for gradient agreement and a fallback for cases
  Enzyme rejects. Do **not** remove Mooncake during this task —
  Phase 8 (Mooncake removal) is a separate future PR.
- **Custom AD rules are duplicated by design.** Each AD-critical
  function has a `ChainRulesCore.rrule` (consumed by Mooncake via
  `@from_rrule`) AND a hand-written `Enzyme.EnzymeRules.augmented_primal`
  / `reverse` pair. This is forced by Enzyme 0.13 + japi3 calling
  convention (EnzymeAD/Enzyme.jl #2707). Keep both.
- **Julia version constraint.** Mooncake 0.4.80 fails to precompile on
  Julia 1.12. Use Julia 1.11.2 for testing. `calculate_initial_model`
  and Mooncake first-compile need ≥ 600s timeouts.
- **DPCA semantics changed.** The new DPCA `l_total` (`optimization_functions.jl:156`)
  inlines `doppler_component_AD(om.star.λ, star_μ)` so ∂loss/∂μ now flows
  through the Doppler basis. Pre-port Nabla treated the basis as constant.
  Wobble path is unaffected.

### Your tasks — in this order

#### Task 1. Skip Deleting confirmed dead Nabla code

#### Task 2. Fix the `doppler_component_log_AD` recursion bug

At `src/DPCA_functions.jl:76–77`, the function recurses against
itself (infinite loop). It is never called. Either:

- Fix: change RHS to `doppler_component_AD(λ, flux) ./ flux`
  (matching the non-AD sibling at line 58). Add a unit test that
  it agrees with FD.
- Or delete entirely.

Ask the user which they prefer before editing.

#### Task 3. Add Enzyme-coverage tests for previously untested paths

Add to `test/runtests.jl`. For each new testset, compare
Mooncake↔Enzyme gradients (rtol 1e-6) AND Enzyme↔FD (rtol 1e-3).
Use the existing `est_∇` helper.

a. **`opt_funcs + Optim.optimize` with EnzymeBackend.** Build a
   tiny `OrderModelWobble`, get `loss_funcs_telstar(o, om, d)`, drive
   `opt_funcs(loss_telstar, [vec(om.tel.lm), vec(om.star.lm)];
   backend=EnzymeBackend())` through 5 L-BFGS iterations. Assert
   it converges and that the final point matches the Mooncake-driven
   version to ~1e-8.

b. **End-to-end `improve_model!` smoke test.** Build a synthetic
   `GenericData` of size ~(30 model pixels × 4 epochs), construct
   `TotalWorkspace`, run `improve_model!(mws; iter=20)`, assert
   the loss decreased and RVs are finite.

c. **`LSFData` Enzyme gradient.** Build a tiny `LSFData` (use a
   small diagonal `SparseMatrixCSC` as the LSF). Construct
   `_loss(o, om, d)` and check the Enzyme gradient against Mooncake
   on the closure that captures `d.lsf` as a single sparse matrix
   (not a vector) — this exercises the rule at `ad_backend.jl:266–291`.

d. **(Optional)** A `DPCA improve_model!` smoke test, with a clear
   "expected to differ from Wobble" assertion.

#### Task 4. Unify backend defaults

Apply **Option A** from the next-steps doc unless the user says
otherwise:

- Change `AdamSubWorkspace(...; backend::ADBackend=EnzymeBackend())`
  at `optimization_functions.jl:445`.
- Change `opt_funcs(...; backend::ADBackend=EnzymeBackend())` at
  `optimization_functions.jl:862`.
- Remove the now-redundant `backend=EnzymeBackend()` overrides from
  `TotalWorkspace` (`:648`) and `FrozenTelWorkspace` (`:709`)
  signatures.
- Thread `backend` kwarg through `estimate_σ_curvature_helper` and
  `estimate_σ_curvature` (`error_estimation.jl:9, 19, 35`). Default
  the kwarg to `EnzymeBackend()`. Replace the two hardcoded
  `MooncakeBackend()` literals with the kwarg.

After: `grep -rn "MooncakeBackend()" src/` should only return
ChainRules rule files / dispatch-time references (not call-site
defaults).

#### Task 5. Document the DPCA semantics change

Add a comment block above
`loss_funcs_total(o::Output, om::OrderModelDPCA, d::Data)` at
`optimization_functions.jl:156`:

```julia
# Migration note (Nabla → Enzyme):
# The Doppler basis is recomputed inline from `om.star.lm.μ` inside
# `l_total`, so the gradient flows ∂loss/∂μ → ∂doppler_basis. This
# differs from the pre-port Nabla path, which treated the basis as a
# constant. The Wobble path is unaffected.
```

Also reflect this in `docs/src/opt.md` (the docstring page that
mirrors `optimization_functions.jl`) — one paragraph under the
existing `loss_funcs_total` entry.

#### Task 6. Tighten `inactive_type` declaration

In `src/ad_backend.jl:123`, change

```julia
Enzyme.EnzymeRules.inactive_type(::Type{<:AbstractDict}) = true
```

to

```julia
Enzyme.EnzymeRules.inactive_type(::Type{<:Dict{Symbol, <:Real}}) = true
```

This still covers `reg_tel` / `reg_star` (both `Dict{Symbol, T}
where T<:Real`) without globally declaring every dict in
downstream packages inactive. Verify `Pkg.test()` still passes.

#### Task 7. Benchmark vs. Nabla baseline

The repo root has an untracked `benchmark_ad.jl`. It needs
`examples/data/results.jld2` and `examples/data/data.jld2` (the
latter is present; `data.jld2.bak` exists but use `data.jld2`).
If `results.jld2` is missing, ask the user where it lives.

Run benchmarks for THREE engines, report a table:

| Engine | How to get it |
|---|---|
| Nabla baseline | `git stash`, `git checkout master`, run `benchmark_ad.jl`, save numbers, then `git checkout try_enzyme && git stash pop`. (Master is `try_enzyme`'s base; Nabla is what master used.) |
| Mooncake | On `try_enzyme`, build workspace with `backend=MooncakeBackend()` explicitly |
| Enzyme | On `try_enzyme`, default backend (Enzyme after task 4) |

Measure for each:

- **Steady-state Adam step** (median of 50 iterations after a warmup of 5)
- **`ModelWorkspace` construction (first call)** — includes compile time
- **`finalize_scores!` first call**
- **`improve_model!` total time** on the example dataset
- **End-to-end Wobble RVs** — compare bit-for-bit Mooncake vs Enzyme on
  `try_enzyme`; compare Nabla vs Enzyme to characterize the
  expected divergence (DPCA only; Wobble should match to ~1e-10)

Acceptance targets (per `enzyme-migration-plan.md` Phase 7):

- Steady-state Adam: ≤ 400 ms (target), hard floor ≤ 600 ms
- Workspace first call: ≤ 90 s (investigate if > 180 s)
- `finalize_scores!` first call: ≤ 300 s

Write the results into `enzyme-migration-benchmark-results.md` at
the repo root, formatted as markdown tables. Include:

- Hardware (CPU model, RAM, OS, Julia version)
- Git commit hash for each engine
- All raw timings, not just medians

If Enzyme misses the hard floor, **stop and report** before
proceeding — a perf bug is worth investigating before more cleanup.

### What NOT to do

- Do not delete code that called Nabla.jl.  We still might need it for regression tests or benchmark tests.jl
- Do **not** start Phase 6 (extension wiring) or Phase 8 (Mooncake
  removal). Those are separate PRs.
- Do **not** delete the planning docs at the repo root
  (`enzyme-migration-plan.md`, `mooncake-perf-plan.md`,
  `nabla-removal-plan.md`, `resume.md`,
  `enzyme-migration-review.md`, `enzyme-migration-next-steps.md`,
  this file). The user will decide their fate before the PR.
- Do **not** touch the untracked files (`HeirarchicalGPQwenTest.jl`,
  `Manifest.toml.*.bak`, `examples/data/data.jld2.bak`,
  `enzyme-julia111-plan.md`) — they belong to the user.
- Do **not** skip pre-commit hooks or amend existing commits. Make a
  new commit per logical task (one for dead-code removal, one for
  the recursion fix, etc.) so the PR history reads cleanly.
- Do **not** add backwards-compat shims. Just change the code.

### Verification at the end

- `Pkg.test()` passes on Julia 1.11.2
- `julia examples/example.jl` runs to completion and produces RVs
  matching the Mooncake baseline within 1e-10 (Wobble path)
- `grep -rn "Nabla\|nabla" src/ test/` returns empty (or only
  intentional historical comments)
- The benchmark results file exists with all three engines tabulated
- One report message at the end summarizing what was done, what was
  found, and the benchmark headline numbers

### Style and conventions

The repo follows BlueStyle Julia. Read `CLAUDE.md` (project) and
`~/.claude-profiles/julia/CLAUDE.md` (user global) before editing.
Key points:

- Surgical changes only — don't refactor adjacent code
- Prefer generic argument types (`AbstractVector`, `Real`)
- Prefer views over allocating new arrays in hot paths
- Use docstrings for *why*, not *what*
- Ask before deleting files

Ask the user before any destructive operation, before adding
features beyond the listed tasks, and when ambiguity arises.
