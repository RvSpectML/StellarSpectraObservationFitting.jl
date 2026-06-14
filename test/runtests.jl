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

@testset "EnzymeBackend nested-tuple θ with aliasing into captured om" begin
    # Phase 3: the Adam path. Production calls look like:
    #   AdamSubWorkspace((om.tel.lm.s, om.star.lm.s, om.rv), l_total)
    # where each θ entry is identity-equal to an array inside `om`, and the
    # loss closure captures `om`. The shadow ∂θ must alias ∂l's captured ∂om
    # in the same way. Tested here with a minimal synthetic struct that mimics
    # the relevant aspects.
    mutable struct MiniOM
        M::Matrix{Float64}
        s::Vector{Float64}
        μ::Vector{Float64}
        bias::Matrix{Float64}    # captured-only context (not in θ)
    end
    om = MiniOM(rand(4, 3), rand(3), rand(4), rand(4, 5))

    # Aliasing invariant: θ[1] === om.M, θ[2] === om.s, θ[3] === om.μ.
    # The loss reads BOTH θ AND om.bias (captured); correctness requires
    # gradients to flow through θ even though the same arrays are reachable
    # via captured om.
    build_θ_minimini(om) = (om.M, om.s, om.μ)
    build_l_minimini(om, _, _) = function (θ)
        # Reads θ AND captured om.bias; computes loss = ||θ[1]*θ[2]+θ[3] − col_means(om.bias)||²
        pred = θ[1] * θ[2] .+ θ[3]
        target = vec(sum(om.bias; dims=2)) ./ size(om.bias, 2)
        return sum((pred .- target) .^ 2)
    end

    θ = build_θ_minimini(om)
    l = build_l_minimini(om, nothing, nothing)

    # Aliasing assertions on the primal (sanity check that the test setup is right)
    @test pointer(θ[1]) === pointer(om.M)
    @test pointer(θ[2]) === pointer(om.s)
    @test pointer(θ[3]) === pointer(om.μ)

    # Enzyme cache
    cache = SSOF.prepare_gradient(
        SSOF.EnzymeBackend(), l, θ;
        om=om, build_θ=build_θ_minimini, build_l=build_l_minimini,
        o=nothing, d=nothing,
    )

    val_en, ∂θ_en = SSOF.value_and_gradient!(cache, l, θ)

    # FD reference: differentiate the closure w.r.t. a flat parameterization
    function ℓ_flat(x)
        M = reshape(view(x, 1:12),   4, 3)
        s = view(x, 13:15)
        μ = view(x, 16:19)
        pred = M * s .+ μ
        target = vec(sum(om.bias; dims=2)) ./ size(om.bias, 2)
        return sum((pred .- target) .^ 2)
    end
    x_flat = vcat(vec(om.M), om.s, om.μ)
    fd = est_∇(ℓ_flat, copy(x_flat); dif=1e-6)

    ∂_flat = vcat(vec(∂θ_en[1]), ∂θ_en[2], ∂θ_en[3])
    @test isapprox(val_en, ℓ_flat(x_flat); rtol=1e-10)
    @test isapprox(∂_flat, fd; rtol=1e-3)

    # Second call uses the same cache; verify remake_zero! correctly resets
    val_en2, ∂θ_en2 = SSOF.value_and_gradient!(cache, l, θ)
    @test isapprox(val_en2, val_en; rtol=1e-12)
    @test isapprox(∂θ_en2[1], ∂θ_en[1]; rtol=1e-12)

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

@testset "EnzymeBackend nested-tuple θ for OrderModelDPCA (l_total)" begin
    # Build a minimal DPCA model. l_total for DPCA inlines the spectral computation
    # (tel_o, star_o, rv_o via positional args) so Enzyme sees all active variables
    # through positional arguments rather than kwargs, ensuring correct activity tracking.
    n_obs = 20
    n_epochs = 4
    log_λ_obs = collect(LinRange(8.78535, 8.78602, n_obs)) .+ 1e-5 .* (0:n_epochs-1)'
    log_λ_star = log_λ_obs
    flux = ones(n_obs, n_epochs) .+ 0.1 .* randn(n_obs, n_epochs)
    flux = max.(flux, 1e-6)
    var = fill(1e-4, n_obs, n_epochs)
    d = SSOF.GenericData(flux, var, var, log_λ_obs, log_λ_star)
    om = SSOF.OrderModel(d; dpca=true, n_comp_tel=1, n_comp_star=1, oversamp=false)
    # initialize templates on the model grid (length differs from n_obs)
    om.star.lm.μ .= 1 .+ 0.1 .* sin.(om.star.λ)
    om.tel.lm.μ .= one(eltype(om.tel.lm.μ))
    om.rv.lm.M[:, 1] .= SSOF.doppler_component(om.star.λ, om.star.lm.μ)
    o = SSOF.Output(om, d)

    l_total, _ = SSOF.loss_funcs_total(o, om, d)
    θ = (SSOF._lm_tuple(om.tel.lm), SSOF._lm_tuple(om.star.lm), om.rv.lm.s)

    cache_mc = SSOF.prepare_gradient(SSOF.MooncakeBackend(), l_total, θ)
    val_mc, ∂θ_mc = SSOF.value_and_gradient!(cache_mc, l_total, θ)

    cache_en = SSOF.prepare_gradient(SSOF.EnzymeBackend(), l_total, θ)
    val_en, ∂θ_en = SSOF.value_and_gradient!(cache_en, l_total, θ)

    @test isapprox(val_mc, val_en; rtol=1e-8)

    # gradient w.r.t. star μ must include the doppler-basis path
    ∂μ_mc = ∂θ_mc[2][3]
    ∂μ_en = ∂θ_en[2][3]
    @test isapprox(∂μ_mc, ∂μ_en; rtol=1e-4)

    # finite-difference sanity on star μ gradient
    function l_total_flat_μ(μ_flat)
        tel_t, star_t, rv_s = θ
        return l_total((tel_t, (star_t[1], star_t[2], μ_flat), rv_s))
    end
    fd_μ = est_∇(l_total_flat_μ, copy(om.star.lm.μ); dif=1e-6)

    @test isapprox(∂μ_en, fd_μ; rtol=1e-3)
    @test isapprox(∂μ_mc, fd_μ; rtol=1e-3)

    ∂rv_en = ∂θ_en[3]
    ∂rv_mc = ∂θ_mc[3]
    @test isapprox(∂rv_mc, ∂rv_en; rtol=1e-4)

    println()
end
