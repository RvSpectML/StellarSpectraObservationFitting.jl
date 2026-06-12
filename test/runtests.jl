using Test
import TemporalGPs; TGP = TemporalGPs
using SparseArrays
import StellarSpectraObservationFitting as SSOF
using LinearAlgebra

println("Testing...")

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
