## Importing packages
using Pkg
    Pkg.activate("EXPRES")

    import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
    using JLD2
    using Statistics
    import StatsBase

    ## Setting up necessary variables

    stars = ["10700", "26965", "34411"]
    star = stars[SSOF.parse_args(1, Int, 2)]
    include("data_locs.jl")  # defines expres_data_path and expres_save_path
    desired_order = SSOF.parse_args(2, Int, 68)  # 68 has a bunch of tels, 47 has very few

    ## Loading in data and initializing model
    save_path = expres_save_path * star * "/$(desired_order)/"
    @load save_path * "data.jld2" n_obs data times_nu airmasses

    if isfile(save_path*"results.jld2")
        @load save_path*"results.jld2" model rvs_naive rvs_notel
        if model.metadata[:todo][:err_estimated]
            @load save_path*"results.jld2" rv_errors
        end
        if model.metadata[:todo][:downsized]
            @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
        end
    else
        model_upscale = 2 * sqrt(2)
        @time model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8, upscale=model_upscale)
        @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
        if !use_reg
            SSOF.rm_regularization(model)
            model.metadata[:todo][:reg_improved] = true
        end
        @save save_path*"results.jld2" model rvs_naive rvs_notel
    end

    using AbstractGPs, KernelFunctions, TemporalGPs
    import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
    using LinearAlgebra
    using SparseArrays
    using BenchmarkTools

    ## Looking into using sparse inverse of covariance
    x = model.tel.log_λ
    x2 = x[1:1000]
    y = model.tel.lm.μ
    y .-= 1
    y2 = y[1:1000]

## Looking into using a gutted version of TemporalGPs attempt 2
# LTI SDE : Linear Time Invariant Stochastic differential equation
# LGGSM : linear-Gaussian state-space model
# LGC : Linear-Gaussian Conditional
import TemporalGPs; TGP = TemporalGPs
using Zygote

fx = ft = SSOF.SOAP_gp(x2, 8e-5)
@time TGP._logpdf(fx, y2)

n = length(x2)
@time Σ = cholesky(cov(fx))
    ℓ = -(n * log(2*π) + logdet(Σ) + y2' * (Σ \ y2)) / 2

## my version from scratch
using StaticArrays

# we have the same F, H, & m. A and Q (i.e. ΣΔt) are calced as expected
# q off by a factor of 2 but P∞ somehow the same?
λ = sqrt(5)
F = [0. 1 0;0 0 1;-λ^3 -3λ^2 -3λ]

# using SpecialFunctions
# σ^2*2sqrt(π)*gamma(3)/gamma(5/2)*λ^5  # eq 9 https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf
# D = 3; q = (factorial(D-1))^2/(factorial(2D-2))*((2λ)^(2D-1))  # eq 12.35 in ASDE
# q = 16/3*λ^5
# L = [0;0;1]
# Q = L* q *L'
# using MatrixEquations
# @time P∞ = lyapc(F, Q)
P∞ = SMatrix{3,3}([1 0 -5/3; 0 5/3 0; -5/3 0 25])
# F*P∞ + P∞*F' + Q

A_k = A_km1 = SMatrix{3,3}(exp(F * step(x) * fx.f.f.kernel.transform.s[1]))  # State transition matrix
Σ_k = Σ_km1 = SMatrix{3,3}(Symmetric(P∞) - A_k * Symmetric(P∞) * A_k')  # eq. 6.71, the process noise
σ²_kernel = ft.f.f.kernel.kernel.σ²[1]
H = [1 0 0]
H_k = SMatrix{1,3}(H .* sqrt(σ²_kernel))
σ²_meas = fx.Σy[1,1]

## kalman filter update (alg 10.18 in ASDE) for constant Ak and Qk

function ℓ_basic(y, A_k, Σ_k, H_k, P∞; σ²_meas::Real=1e-12)

    n = length(y)
    n_state = 3
    ℓ = 0
    m_k = @MVector zeros(n_state)
    P_k = MMatrix{3,3}(P∞)
    m_kbar = @MVector zeros(n_state)
    P_kbar = @MMatrix zeros(n_state, n_state)
    K_k = @MMatrix zeros(n_state, 1)
    for k in 1:n
        # prediction step
        m_kbar .= A_k * m_k  # state prediction
        P_kbar .= A_k * P_k * A_k' + Σ_k  # covariance of state prediction, all of the allocations are here

        # update step
        v_k = y[k] - only(H_k * m_kbar)  # difference btw meas and pred, scalar
        S_k = only(H_k * P_kbar * H_k') + σ²_meas  # P_kbar[1,1] * σ²_kernel + σ²_meas, scalar
        K_k .= P_kbar * H_k' / S_k  # 3x1
        m_k .= m_kbar + SVector{3}(K_k * v_k)
        P_k .= P_kbar - K_k * S_k * K_k'
        ℓ -= log(S_k) + v_k^2/S_k  # 2*ℓ without normalization
    end
    return (ℓ - n*log(2π))/2
end

@time ℓ_basic(y2, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
@time SSOF.SOAP_gp_ℓ(y2, step(x2); σ²_meas=σ²_meas)
A_k = SMatrix{3,3}(exp(SSOF.F * step(x) * SSOF.SOAP_gp.f.kernel.transform.s[1]))  # State transition matrix
Σ_k = SMatrix{3,3}(Symmetric(P∞) - A_k * Symmetric(P∞) * A_k')  # eq. 6.71, the process noise
@time SSOF.SOAP_gp_ℓ(y2, A_k, Σ_k; σ²_meas=σ²_meas)

