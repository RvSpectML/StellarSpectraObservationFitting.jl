import TemporalGPs; TGP = TemporalGPs
using Zygote
using LinearAlgebra

my_logpdf2(ft::TGP.FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}}) =
    my_logpdf2(TGP.build_lgssm(ft), y)
my_logpdf2(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(TGP.scan_emit(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])
my_logpdf3(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(my_scan_emit2(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])

function my_scan_emit2(f::Function, xs::Iterators.Zip, state::TGP.Gaussian, idx::UnitRange)
    (yy, state) = f(state, TGP._getindex(xs, idx[1]))
    ys = Vector{typeof(yy)}(undef, length(idx))
    ys[idx[1]] = yy

    for t in idx[2:end]
        (yy, state) = f(state, TGP._getindex(xs, t))
        ys[t] = yy
    end

    return (ys, state)
end



my_logpdf(ft::TGP.FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}}) =
    TGP.logpdf(TGP.build_lgssm(ft), y)
my_logpdf(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(TGP.scan_emit(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])
function my_scan_emit(f, xs, state, idx)
    (lml, state) = f(state, TGP._getindex(xs, idx[1]))
    lmls = Vector{typeof(yy)}(undef, length(idx))
    lmls[idx[1]] = lml
    for t in idx[2:end]
        (lml, state) = f(state, TGP._getindex(xs, t))
        lmls[t] = lml
    end
    return lmls
end
function my_step_logpdf(x::TGP.Gaussian, (model_i, y_i))
    xp = my_predict(x, model_i.transition)
    xf, lml = my_posterior_and_lml(xp, model_i.emission, y_i)
    return lml, xf
end
my_predict(x::TGP.Gaussian, f::TGP.AbstractLGC) = TGP.Gaussian(f.A * x.m + f.a, f.A * my_symmetric(x.P) * f.A' + f.Q)
my_symmetric(X::AbstractMatrix) = Symmetric(X)
my_symmetric(X::Diagonal) = X
function my_posterior_and_lml(x::TGP.Gaussian, f::TGP.ScalarOutputLGC, y::T) where {T<:Real}
    m, P = x.m, x.P
    A, a, Q = f.A, f.a, f.Q
    V = A * P
    sqrtS = sqrt(V * A' + Q)  # std of obs at time
    B = sqrtS \ V
    α = sqrtS \ (y - (A * m + a))
    lml = -(log(2π) + 2 * log(sqrtS) + α^2) / 2
    return TGP.Gaussian(m + B'α, P - B'B), lml
end
