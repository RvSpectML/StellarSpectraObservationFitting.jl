using Test
import TemporalGPs; TGP = TemporalGPs
using SparseArrays
import StellarSpectraObservationFitting as SSOF
using LinearAlgebra

println("Testing...")

@testset "AD backends load" begin
    @test SSOF.MooncakeBackend() isa SSOF.ADBackend
    @test SSOF.EnzymeBackend() isa SSOF.ADBackend
end

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

@testset "fast GP prior likelihoods (and their gradients)" begin
    x = 8.78535917650598:6.616545829861497e-7:8.786020831088965
    fx = SSOF.SOAP_gp(x)
    y = rand(fx)

    # are my likelihood calculations the same as TemporalGPs
    # @test isapprox(TGP.logpdf(fx, y), SSOF.SOAP_gp_ℓ(y, step(x)))
    # @test isapprox(TGP.logpdf(fx, y), SOAP_gp_ℓ_nabla(y, step(x)))

    # setting up constants and precalcuating gradient coefficients
    H_k, P∞, σ²_meas = SSOF.H_k, SSOF.P∞, SSOF._σ²_meas_def
    A_k, Σ_k = SSOF.SOAP_gp_sde_prediction_matrices(step(x))
    sparsity = Int(round(0.5 / (step(x) * SSOF.SOAP_gp_params.λ)))
    Δℓ_coe = SSOF.gp_Δℓ_coefficients(length(y), A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas)
    Δℓ_coe_s = SSOF.gp_Δℓ_coefficients(length(y), A_k, Σ_k; H_k=H_k, P∞=P∞, σ²_meas=σ²_meas, sparsity=sparsity)

    f(y) = SSOF.gp_ℓ(y, A_k, Σ_k; σ²_meas=σ²_meas)
    numer = est_∇(f, y; dif=1e-9)
    anal = SSOF.gp_Δℓ(y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
    anal_p = SSOF.Δℓ_precalc(Δℓ_coe, y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
    anal_p_s = SSOF.Δℓ_precalc(Δℓ_coe_s, y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)

    @test isapprox(numer, anal; rtol=1e-4)
    @test isapprox(numer, anal_p; rtol=1e-4)
    @test isapprox(numer, anal_p_s; rtol=1e-4)

    # f(y) = SSOF.gp_ℓ_nabla(y, A_k, Σ_k; σ²_meas=σ²_meas)
    # nabla = only(∇(f)(y_test))
    println()
end

@testset "custom spectra_interp() sensitivity" begin

    B = rand(3,5)
    As = [sparse(rand(2,3)) for i in axes(B, 2)]
    C = rand(5,6)

    f_custom_sensitivity(x) = sum(SSOF.spectra_interp(x .^ 2, As) * C)

    # Numerical gradient via finite differences
    B_vec = copy(vec(B))
    numer = est_∇(xv -> f_custom_sensitivity(reshape(xv, size(B))), B_vec; dif=1e-7)

    # Analytic gradient via Mooncake (exercises the ChainRulesCore rrule)
    cache = SSOF.prepare_gradient(SSOF.MooncakeBackend(), f_custom_sensitivity, copy(B))
    _, ∂B = SSOF.value_and_gradient!(cache, f_custom_sensitivity, copy(B))

    @test isapprox(vec(∂B), numer; rtol=1e-4)

    println()
end

@testset "EnzymeBackend flat-vector path (spectra_interp via SIH)" begin
    # Exercises the flat-Vector{Float64} EnzymeBackend path against a loss that
    # captures a StellarInterpolationHelper — representative of the closures
    # built by the optimizer paths. FD is the gold-standard correctness gate;
    # Mooncake↔Enzyme agreement is the secondary check.
    n_model, n_obs, n_epochs = 20, 8, 3
    log_λ_obs = collect(LinRange(1.0, 2.0, n_obs)) .+ 0.0 .* (1:n_epochs)'
    log_λ_obs = collect(reshape(log_λ_obs, n_obs, n_epochs))
    model_log_λ = range(0.5, 2.5; length=n_model)
    rvs = randn(n_epochs) .* 0.01
    sih = SSOF.StellarInterpolationHelper(model_log_λ, rvs, log_λ_obs)

    # Flat-vector loss: model_flux flattened to a Vector{Float64}.
    n_flux = n_model * n_epochs
    function ℓ(x::AbstractVector{Float64})
        M = reshape(x, n_model, n_epochs)
        return sum(SSOF.spectra_interp(M, rvs, sih) .^ 2)
    end

    x0 = collect(LinRange(0.5, 1.5, n_flux))

    # Finite-difference reference
    fd = est_∇(ℓ, copy(x0); dif=1e-6)

    # Mooncake
    mcache = SSOF.prepare_gradient(SSOF.MooncakeBackend(), ℓ, copy(x0))
    val_mc, ∂_mc = SSOF.value_and_gradient!(mcache, ℓ, copy(x0))

    # Enzyme (flat-vector path)
    ecache = SSOF.prepare_gradient(SSOF.EnzymeBackend(), ℓ, copy(x0))
    val_en, ∂_en = SSOF.value_and_gradient!(ecache, ℓ, copy(x0))

    @test isapprox(val_mc, val_en; rtol=1e-10)
    @test isapprox(∂_mc, ∂_en; rtol=1e-6)
    @test isapprox(∂_en, fd; rtol=1e-3)

    println()
end

@testset "EnzymeBackend on loss ∘ unflatten composition" begin
    # opt_funcs (optimization_functions.jl:804) differentiates `f = loss ∘ unflatten`
    # where unflatten rebuilds a nested parameter structure from a flat vector.
    # Phase 5 will switch the default backend to Enzyme via this exact composition,
    # so Phase 2's done-gate must exercise it — a hand-written ℓ on a flat input
    # does not capture the closure shape (the unflatten closure carries its own
    # captured shape info) that production code uses.
    pars0 = [rand(3, 2), rand(2, 3), rand(5)]
    loss(pars) = sum(pars[1]) + sum(pars[2] .^ 2) + sum(pars[3] .^ 3)

    p0, unflatten = SSOF.flatten(pars0)
    f = loss ∘ unflatten

    cache_mc = SSOF.prepare_gradient(SSOF.MooncakeBackend(), f, copy(p0))
    val_mc, ∂_mc = SSOF.value_and_gradient!(cache_mc, f, copy(p0))

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), f, copy(p0))
    val_en, ∂_en = SSOF.value_and_gradient!(cache_en, f, copy(p0))

    fd = est_∇(f, copy(p0); dif=1e-6)

    @test isapprox(val_mc, val_en; rtol=1e-10)
    @test isapprox(∂_mc, ∂_en; rtol=1e-6)
    @test isapprox(∂_en, fd; rtol=1e-3)

    println()
end


@testset "weighted project_doppler_comp!()" begin

    flux_star = rand(100, 20)
    weights = sqrt.(flux_star)
    μ = SSOF.make_template(flux_star, weights; min=0, max=1.2, use_mean=true)
    doppler_comp = SSOF.doppler_component(LinRange(5000,6000,100), μ)
    M = zeros(100, 3)
    s = zeros(20)

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs1 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, ones(size(data_tmp)))
    s1 = copy(s)
    M1 = copy(M[:, 1])

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs2 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, ones(size(data_tmp)))
    s2 = copy(s)
    M2 = copy(M[:, 1])

    data_tmp = copy(flux_star)
    data_tmp .-= μ
    rvs3 = SSOF.project_doppler_comp!(M, s, data_tmp, doppler_comp, weights)
    s3 = copy(s)
    M3 = copy(M[:, 1])

    @test M1 == M2 == M3
    @test rvs1 == rvs2
    @test s1 == s2
    @test rvs2 != rvs3
    @test s2 != s3

    println()
end
