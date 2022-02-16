# [1]: Jouni Hartikainen and Simo Särkkä 2015? https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf
# [2]: Arno Solin and Simo Särkkä 2019 https://users.aalto.fi/~asolin/sde-book/sde-book.pdf
using StaticArrays
import TemporalGPs; TGP = TemporalGPs

@assert typeof(SOAP_gp.f.kernel.kernel.kernel) <: Matern52Kernel

## Defining necessary params

const σ²_kernel = SOAP_gp_params.var_kernel  # = SOAP_gp.f.kernel.kernel.σ²[1]

# p = 2
# ν = 2 * p + 1 / 2
const λ = sqrt(5)  # ==sqrt(2 ν) / l, l assumed to be 1, dealt with later by scaling in A_k
const F = SMatrix{3,3}([0. 1 0;0 0 1;-λ^3 -3λ^2 -3λ])  # eq 9+ of [1]

# using SpecialFunctions
# σ^2*2sqrt(π)*gamma(3)/gamma(5/2)*λ^5  # eq 9 of [1]
# D = 3; q = (factorial(D-1))^2/(factorial(2D-2))*((2λ)^(2D-1))  # eq 12.35 in [2]
# q = 16/3*λ^5
# L = [0;0;1]
# Q = L* q *L'
# using MatrixEquations
# F*P∞ + P∞*F' + Q == 0 # eq 6.69 in [2] process is in steady state before we start
# P∞ = lyapc(F, Q)
const P∞ = SMatrix{3,3}([1 0 -5/3; 0 5/3 0; -5/3 0 25])  # steady-state state covariance, derived with the above commented lines

H = [1 0 0]  # matrix for extracting only the measured part of the state
const H_k = SMatrix{1,3}(H .* sqrt(σ²_kernel))  # this is how we deal with kernel amplitude

_σ²_meas_def = 1e-12

# Need one of these per timestep
# They are constant if we have constant timestep
# σ²_meas and H_k only inlcuded as kwargs to prevent errors with passing kwargs...
# in SOAP_gp_ℓ(y, Δx::Real; kwargs...)
function SOAP_gp_sde_prediction_matrices(Δx; Δx_scaler::Real=SOAP_gp_params.λ, P∞::AbstractMatrix=P∞, F::AbstractMatrix=F, σ²_meas::Real=_σ²_meas_def, H_k::AbstractMatrix=H_k)
    A_k = SMatrix{3,3}(exp(F * Δx * Δx_scaler))  # State transition matrix eq 6.23 in [2]?
    Σ_k = SMatrix{3,3}(Symmetric(P∞) - A_k * Symmetric(P∞) * A_k')  # eq. 6.71 in [2], the process noise
    return A_k, Σ_k
end
LSF_gp_sde_prediction_matrices(Δx; Δx_scaler::Real=LSF_gp_params.λ, kwargs...) =
    SOAP_gp_sde_prediction_matrices(Δx; Δx_scaler=Δx_scaler, kwargs...)

function predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)
    m_kbar .= A_k * m_k  # state prediction
    P_kbar .= A_k * P_k * A_k' + Σ_k  # covariance of state prediction
end
function update_sde!(K_k, m_k, P_k, y, H_k, m_kbar, P_kbar, σ²_meas)
    v_k = y - only(H_k * m_kbar)  # difference btw meas and pred, scalar
    S_k = only(H_k * P_kbar * H_k') + σ²_meas  # P_kbar[1,1] * σ²_kernel + σ²_meas, scalar
    K_k .= P_kbar * H_k' / S_k  # 3x1
    m_k .= m_kbar + SVector{3}(K_k * v_k)
    P_k .= P_kbar - K_k * S_k * K_k'
    return v_k, S_k
end
function init_states(n_state)
    m_k = @MVector zeros(n_state)
    P_k = MMatrix{3,3}(P∞)
    m_kbar = @MVector zeros(n_state)
    P_kbar = @MMatrix zeros(n_state, n_state)
    K_k = @MMatrix zeros(n_state, 1)
    return m_k, P_k, m_kbar, P_kbar, K_k
end

function SOAP_gp_ℓ(y, Δx::Real; kwargs...)
    A_k, Σ_k = SOAP_gp_sde_prediction_matrices(Δx; kwargs...)
    return gp_ℓ(y, A_k, Σ_k; kwargs...)
end
function LSF_gp_ℓ(y, Δx::Real; kwargs...)
    A_k, Σ_k = LSF_gp_sde_prediction_matrices(Δx; kwargs...)
    return gp_ℓ(y, A_k, Σ_k; kwargs...)
end

# Based on Kalman filter update (alg 10.18 in ASDE) for constant Ak and Qk
# changing y only changes m_kbar, v_k, and m_k. Could be faster if
# P_kbar, S_k, K_k, and P_k were saved?
function gp_ℓ(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix; σ²_meas::Real=_σ²_meas_def, H_k::AbstractMatrix=H_k, P∞::AbstractMatrix=P∞)

    n = length(y)
    n_state = 3
    ℓ = 0
    m_k, P_k, m_kbar, P_kbar, K_k = init_states(n_state)

    for k in 1:n
        # prediction step
        predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)

        # update step
        v_k, S_k = update_sde!(K_k, m_k, P_k, y[k], H_k, m_kbar, P_kbar, σ²_meas)

        ℓ -= log(S_k) + v_k^2/S_k  # 2*ℓ without normalization
    end
    return (ℓ - n*log(2π))/2
end


function SOAP_gp_ℓ_nabla(y, Δx::Real; kwargs...)
    A_k, Σ_k = SOAP_gp_sde_prediction_matrices(Δx; kwargs...)
    return gp_ℓ_nabla(y, A_k, Σ_k; kwargs...)
end
function LSF_gp_ℓ_nabla(y, Δx::Real; kwargs...)
    A_k, Σ_k = LSF_gp_sde_prediction_matrices(Δx; kwargs...)
    return gp_ℓ_nabla(y, A_k, Σ_k; kwargs...)
end


# removing things that Nabla doesn't like from SOAP_gp_ℓ
function gp_ℓ_nabla(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix; σ²_meas::Real=_σ²_meas_def, H_k::AbstractMatrix=H_k, P∞::AbstractMatrix=P∞)

    n = length(y)
    ℓ = 0
    n_state = 3
    m_k = @MMatrix zeros(n_state, 1)
    P_k = MMatrix{3,3}(P∞)
    # m_kbar = @MVector zeros(n_state)
    P_kbar = @MMatrix zeros(n_state, n_state)
    K_k = @MMatrix zeros(n_state, 1)
    for k in 1:n
        # prediction step
        m_kbar = A_k * m_k  # state prediction
        P_kbar .= A_k * P_k * A_k' + Σ_k  # covariance of state prediction, all of the allocations are here

        # update step
        v_k = y[k] - (H_k * m_kbar)[1]  # difference btw meas and pred, scalar
        S_k = only(H_k * P_kbar * H_k') + σ²_meas  # P_kbar[1,1] * σ²_kernel + σ²_meas, scalar
        K_k .= P_kbar * H_k' / S_k  # 3x1
        # m_k .= m_kbar + SVector{3}(K_k * v_k)
        m_k = m_kbar + (K_k * v_k)
        P_k .= P_kbar - K_k * S_k * K_k'
        ℓ -= log(S_k) + v_k^2/S_k  # 2*ℓ without normalization
    end
    return (ℓ - n*log(2π))/2
end

x_test = 8.78535917650598:6.616545829861497e-7:8.798522794434488
fx = SOAP_gp(x_test)
y = rand(fx)

@assert isapprox(TGP.logpdf(fx, y), SOAP_gp_ℓ(y, step(x_test)))
@assert isapprox(TGP.logpdf(fx, y), SOAP_gp_ℓ_nabla(y, step(x_test)))

# for calculating gradients w.r.t. y
function gp_Δℓ_helper_K(n::Int, A_k::AbstractMatrix, Σ_k::AbstractMatrix, H_k::AbstractMatrix, P∞::AbstractMatrix; σ²_meas::Real=_σ²_meas_def)

    n_state = 3
    _, P_k, _, P_kbar, _ = init_states(n_state)
    K = [MMatrix{3,1}(zeros(3,1)) for i in 1:n]
    for k in 1:n
        # prediction step
        P_kbar .= A_k * P_k * A_k' + Σ_k

        # update step
        S_k = only(H_k * P_kbar * H_k') + σ²_meas  # P_kbar[1,1] * σ²_kernel + σ²_meas, scalar
        K[k] .= P_kbar * H_k' / S_k  # 3x1
        P_k .= P_kbar - K[k] * S_k * K[k]'
    end
    return K
end

function gp_Δℓ_helper_γ(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix, H_k::AbstractMatrix, P∞::AbstractMatrix; σ²_meas::Real=_σ²_meas_def)
    n_state = 3
    n = length(y)
    m_k, P_k, m_kbar, P_kbar, K_k = init_states(n_state)
    γ = zeros(n)
    for k in 1:n
        # prediction step
        predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)

        # update step
        v_k, S_k = update_sde!(K_k, m_k, P_k, y[k], H_k, m_kbar, P_kbar, σ²_meas)

        γ[k] = v_k / S_k
    end
    return γ
end

function gp_Δℓ(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix, H_k::AbstractMatrix, P∞::AbstractMatrix; kwargs...)
    n = length(y)
    K = gp_Δℓ_helper_K(n, A_k, Σ_k, H_k, P∞; kwargs...)  # O(n)
    γ = gp_Δℓ_helper_γ(y, A_k, Σ_k, H_k, P∞; kwargs...)  # O(n)
    # now that we have K and γ
    α = H_k * A_k
    dLdy = copy(γ)
    δLδyk_inter = @MMatrix zeros(3, 1)
    for i in 1:(n-1)
        δLδyk_inter .= K[i]
        dLdy[i] -= γ[i+1] * only(α * δLδyk_inter)
        for j in (i+1):(n-1)
            δLδyk_inter .= (A_k - K[j] * α) * δLδyk_inter
            dLdy[i] -= γ[j+1] * only(α * δLδyk_inter)
        end
    end
    return -dLdy
end


using SparseArrays
function gp_Δℓ_coefficients(n::Int, A_k::AbstractMatrix, Σ_k::AbstractMatrix; H_k::AbstractMatrix=H_k, P∞::AbstractMatrix=P∞, sparsity::Int=0, kwargs...)
    @assert 0 <= sparsity <= n/10
    use_sparse = sparsity != 0

    K = gp_Δℓ_helper_K(n, A_k, Σ_k, H_k, P∞; kwargs...)  # O(n)

    α = H_k * A_k
    # dLdy_coeffs = spdiagm(-ones(n))  # it's faster to start as dense and convert to sparse after
    dLdy_coeffs = diagm(-ones(n))
    δLδyk_inter = @MMatrix zeros(3, 1)
    for i in 1:(n-1)
        δLδyk_inter .= K[i]
        dLdy_coeffs[i, i+1] = only(α * δLδyk_inter)
        use_sparse ? ceiling = min(i+1+sparsity, n-1) : ceiling = n-1
        for j in (i+1):ceiling
            δLδyk_inter .= (A_k - K[j] * α) * δLδyk_inter
            dLdy_coeffs[i, j+1] = only(α * δLδyk_inter)
        end
    end
    if use_sparse
        dLdy_coeffs = sparse(dLdy_coeffs)
        dropzeros!(dLdy_coeffs)
    end
    return dLdy_coeffs
end

# Δℓ_coe = Δℓ_coefficients(y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas)
# Δℓ_coe_s = Δℓ_coefficients(y, A_k, Σ_k, H_k, P∞; σ²_meas=σ²_meas, sparsity=100)

# only to allow Nabla to know that we should use the faster gradient
# calculations using the precalcuated coefficients
gp_ℓ_precalc(Δℓ_coeff::AbstractMatrix, y::AbstractVector, A_k::AbstractMatrix, Σ_k::AbstractMatrix; kwargs...) =
    gp_ℓ(y, A_k, Σ_k; kwargs...)

Δℓ_precalc(Δℓ_coeff::AbstractMatrix, y::AbstractVector, A_k::AbstractMatrix, Σ_k::AbstractMatrix, H_k::AbstractMatrix, P∞::AbstractMatrix; kwargs...) =
    Δℓ_coeff * gp_Δℓ_helper_γ(y, A_k, Σ_k, H_k, P∞; kwargs...)


using Nabla
# BE EXTREMELY CAREFUL! AS WE CANT PASS kwargs... THIS WILL ONLY WORK FOR THE DEFAULT VALUES OF H_k, P∞, F, AND σ²_meas
@explicit_intercepts gp_ℓ_precalc Tuple{AbstractMatrix, AbstractVector, AbstractMatrix, AbstractMatrix}
Nabla.∇(::typeof(gp_ℓ_precalc), ::Type{Arg{2}}, _, y, ȳ, Δℓ_coeff, x, A_k, Σ_k) =
    ȳ .* Δℓ_precalc(Δℓ_coeff, x, A_k, Σ_k, H_k, P∞)


# sm = mws.om.tel
# μ_mod = sm.lm.μ .- 1
# SSOF.SOAP_gp_ℓ_precalc(sm.Δℓ_coeff, μ_mod, sm.A_sde, sm.Σ_sde)
# import TemporalGPs; TGP = TemporalGPs
# TGP._logpdf(SSOF.SOAP_gp(sm.log_λ), μ_mod)

# n_test=1000
# using Nabla
# f1(y) = SSOF.SOAP_gp_ℓ_precalc(sm.Δℓ_coeff[1:length(y), 1:length(y)], y, sm.A_sde, sm.Σ_sde)
# f2(y) = SSOF.SOAP_gp_ℓ_nabla(y, sm.A_sde, sm.Σ_sde)
# only(∇(f1)(μ_mod[1:n_test]))
# only(∇(f2)(μ_mod[1:n_test]))
# est_∇(f1, μ_mod[1:n_test])
# SSOF.Δℓ_precalc(sm.Δℓ_coeff[1:n_test, 1:n_test], μ_mod[1:n_test], sm.A_sde, sm.Σ_sde, SSOF.H_k, SSOF.P∞)


# function SOAP_gp_Δℓ_helper_ℓγ(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix, H_k::AbstractMatrix, P∞::AbstractMatrix; σ²_meas::Real=_σ²_meas_def)
#     n_state = 3
#     n = length(y)
#     m_k, P_k, m_kbar, P_kbar, K_k = init_states(n_state)
#     γ = zeros(n)
#     ℓ = 0
#     for k in 1:n
#         # prediction step
#         predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)
#
#         # update step
#         v_k, S_k = update_sde!(K_k, m_k, P_k, y[k], H_k, m_kbar, P_kbar, σ²_meas)
#
#         γ[k] = v_k / S_k
#         ℓ -= log(S_k) + v_k^2/S_k  # 2*ℓ without normalization
#     end
#     return (ℓ - n*log(2π))/2, γ
# end

# using ChainRulesCore
# function ChainRulesCore.rrule(::typeof(SOAP_gp_ℓ_precalc), Δℓ_coeff::AbstractMatrix, yy::Vector, A_k::AbstractMatrix, Σ_k::AbstractMatrix; kwargs...)
#     y, γ = SOAP_gp_Δℓ_helper_ℓγ(yy, A_k, Σ_k; kwargs...)
#     function SOAP_gp_ℓ_pullback(ȳ)
#         f̄ = NoTangent()
#         ȳy = Δℓ_precalc(Δℓ_coeff, yy, A_k, Σ_k; kwargs...)
#         Ā_k = NoTangent()  # this is wrong but shouldn't be needed
#         Σ̄_k = NoTangent()  # this is wrong but shouldn't be needed
#         return f̄, ȳy, Ā_k, Σ̄_k
#     end
#     return y, foo_mul_pullback
# end
