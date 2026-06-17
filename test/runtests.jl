using Test
import TemporalGPs; TGP = TemporalGPs
using SparseArrays
import StellarSpectraObservationFitting as SSOF
using LinearAlgebra
using Statistics

println("Testing...")

# Set SSOF_SKIP_SLOW_TESTS=true to skip testsets that take >60 s each.
const SKIP_SLOW = get(ENV, "SSOF_SKIP_SLOW_TESTS", "false") == "true"
SKIP_SLOW && println("Skipping slow tests (SSOF_SKIP_SLOW_TESTS=true)")

# Shared finite-difference gradient estimator used across multiple test files.
function est_∇(f::Function, inputs; dif::Real=1e-7, inds::AbstractUnitRange=eachindex(inputs))
    val = f(inputs)
    grad = Array{Float64}(undef, length(inds))
    for i in inds
        hold = inputs[i]
        inputs[i] += dif
        grad[i] =  (f(inputs) - val) / dif
        inputs[i] = hold
    end
    return grad
end

# Shared tiny Wobble dataset used by several test files.
function _make_tiny_wobble_data(n_obs, n_epochs)
    log_λ_obs = collect(LinRange(8.78535, 8.78602, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    log_λ_star = log_λ_obs
    flux = ones(n_obs, n_epochs) .+ 0.1 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    return SSOF.GenericData(flux, var, var, log_λ_obs, log_λ_star)
end

include("general_functions.jl")
include("model_selection_functions.jl")
include("prior_gp_functions.jl")
include("model_functions.jl")
include("DPCA_functions.jl")
include("EMPCA.jl")
include("flatten.jl")
include("mask_functions.jl")
include("continuum_functions.jl")
include("ad_backend.jl")
include("optimization_functions.jl")
include("error_estimation.jl")
include("regularization_functions.jl")
