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

# Need one of these per timestep
# They are constant if we have constant timestep
# σ²_meas and H_k only inlcuded as kwargs to prevent errors with passing kwargs...
# in SOAP_gp_ℓ(y, Δx::Real; kwargs...)
function SOAP_gp_sde_prediction_matrices(Δx; Δx_scaler::Real=SOAP_gp_params.λ, P∞::AbstractMatrix=P∞, F::AbstractMatrix=F, σ²_meas::Real=1e-12, H_k::AbstractMatrix=H_k)
    A_k = SMatrix{3,3}(exp(F * Δx * Δx_scaler))  # State transition matrix eq 6.23 in [2]?
    Σ_k = SMatrix{3,3}(Symmetric(P∞) - A_k * Symmetric(P∞) * A_k')  # eq. 6.71 in [2], the process noise
    return A_k, Σ_k
end

function predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)
    m_kbar .= A_k * m_k  # state prediction
    P_kbar .= A_k * P_k * A_k' + Σ_k  # covariance of state prediction
end


function SOAP_gp_ℓ(y, Δx::Real; kwargs...)
    A_k, Σ_k = SOAP_gp_sde_prediction_matrices(Δx; kwargs...)
    return SOAP_gp_ℓ(y, A_k, Σ_k; kwargs...)
end

# Based on Kalman filter update (alg 10.18 in ASDE) for constant Ak and Qk
function SOAP_gp_ℓ(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix; σ²_meas::Real=1e-12, H_k::AbstractMatrix=H_k, P∞::AbstractMatrix=P∞, F::AbstractMatrix=F)

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
        predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)

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


function SOAP_gp_ℓ_nabla(y, Δx::Real; kwargs...)
    A_k, Σ_k = SOAP_gp_sde_prediction_matrices(Δx; kwargs...)
    return SOAP_gp_ℓ_nabla(y, A_k, Σ_k; kwargs...)
end


# removing things that Nabla doesn't like from SOAP_gp_ℓ
function SOAP_gp_ℓ_nabla(y, A_k::AbstractMatrix, Σ_k::AbstractMatrix; σ²_meas::Real=1e-12, H_k::AbstractMatrix=H_k, P∞::AbstractMatrix=P∞, F::AbstractMatrix=F)

    n = length(y)
    ℓ = 0
    n_state = 3
    m_k = @MMatrix zeros(n_state, 1)
    P_k = MMatrix{3,3}(P∞)
    P_kbar = @MMatrix zeros(n_state, n_state)
    K_k = @MMatrix zeros(n_state, 1)
    for k in 1:n
        # prediction step
        predict!(m_kbar, P_kbar, A_k, m_k, P_k, Σ_k)

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
