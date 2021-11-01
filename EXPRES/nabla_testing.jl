include("testing_header.jl")
model = reset_model()

## running the optimization outside the functions
using Optim
workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)
ow = workspace; osw = ow.telstar
optim_cb_local(x::OptimizationState) = SSOF.optim_cb(x; print_stuff=true)
options = Optim.Options(;iterations=100, f_tol=SSOF._f_tol_def, g_tol=SSOF._g_tol_def*sqrt(length(osw.p0)), callback=optim_cb_local)
@time result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)

## ADAM
using Nabla

""" Implementation of the Adam optimiser. """
mutable struct Adam{T<:AbstractArray}
    α::Float64
    β1::Float64
    β2::Float64
    m::T
    v::T
    β1_acc::Float64
    β2_acc::Float64
    ϵ::Float64
end
Adam(θ0::AbstractArray, α::Float64, β1::Float64, β2::Float64, ϵ::Float64) =
    Adam(α, β1, β2, custom_zeros(θ0), custom_zeros(θ0), β1, β2, ϵ)
custom_zeros(θ::VecOrMat) = zero(θ)
custom_zeros(θ::Vector{<:Array}) = [custom_zeros(i) for i in θ]

function _iterate_helper!(θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam; m=opt.m, v=opt.v, α=opt.α, β1=opt.β1, β2=opt.β2, ϵ=opt.ϵ, β1_acc=opt.β1_acc, β2_acc=opt.β2_acc)
    # the matrix and dotted version is slower
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + (1.0 - β1) * ∇θ[n]
        v[n] = β2 * v[n] + (1.0 - β2) * ∇θ[n]^2
        m̂ = m[n] / (1 - β1_acc)
        v̂ = v[n] / (1 - β2_acc)
        θ[n] -= α * m̂ / (sqrt(v̂) + ϵ)
    end
end
function iterate!(θs::Vector{<:Vector{<:AbstractArray{<:Real}}}, ∇θs::Vector{<:Vector{<:AbstractArray{<:Real}}}, opts::Vector{Vector{Adam}})
    for i in 1:length(θs)
        for j in 1:length(θs[i])
            _iterate_helper!(θs[i][j], ∇θs[i][j], opts[i][j])
            opt.β1_acc *= opt.β1
            opt.β2_acc *= opt.β2
        end
    end
end
function iterate!(θs::Vector{<:Vector{<:AbstractArray{<:Real}}}, ∇θs::Vector{<:Vector{<:AbstractArray{<:Real}}}, opt::Adam)
    for i in 1:length(θs)
        for j in 1:length(θs[i])
            _iterate_helper!(θs[i][j], ∇θs[i][j], opt; m=opt.m[i][j], v=opt.v[i][j])
        end
    end
    opt.β1_acc *= opt.β1
    opt.β2_acc *= opt.β2
end
α, β1, β2, ϵ = 1e-3, 0.9, 0.999, 1e-8

# model = reset_model()
# om = model; d = data; o = SSOF.Output(om, d)
# θ = [vec(om.tel.lm), vec(om.star.lm)]
# _, l, _, _ = SSOF.loss_funcs_telstar(o, om, d)
#
# l(ts.θ)
# Δ = only(∇(l)(ts.θ))
#
# opts = [Adam.(θ, α, β1, β2, ϵ) for θ in ts.θ]
# # 0.001816 seconds (61 allocations: 1008 bytes)
# @time iterate!(ts.θ, Δ, opts)
#
# opt = Adam(ts.θ, α, β1, β2, ϵ)
# #  0.001803 seconds (14 allocations: 448 bytes)
# iterate!(ts.θ, Δ, opt)


## using ADAM to optimize

model = reset_model()
om = model; d = data; o = SSOF.Output(om, d)
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
plt = status_plot(o, d)
png(plt, "before")
θ = [vec(om.tel.lm), vec(om.star.lm)]
_, l, _, _ = SSOF.loss_funcs_telstar(o, om, d)
opt = Adam(θ, α, β1, β2, ϵ)

mutable struct AdamState
    iter::Int
    ℓ::Float64
    L1_Δ::Float64
    L2_Δ::Float64
    L∞_Δ::Float64
end
AdamState() = AdamState(0, 0., 0., 0., 0.)
function AdamState!(as::AdamState, ℓ, Δ)

as = AdamState()

L∞_cust(Δ) = maximum([maximum([maximum(i) for i in j]) for j in Δ])
# function L∞_cust2(Δ)  # 4x slower
#     it = Iterators.flatten(Iterators.flatten(Δ))
#     return maximum(it)
# end
function update!()
    val, Δ = ∇(l; get_output=true)(θ)
    Δ = only(Δ)
    as.ℓ = val.val
    flat_Δ = Iterators.flatten(Iterators.flatten(Δ))
    as.L1_Δ = sum(abs, flat_Δ)
    as.L2_Δ = sum(abs2, flat_Δ)
    as.L∞_Δ = L∞_cust(Δ)
    iterate!(θ, Δ, opt)
    # println("Iter:  ", x.iteration)
    # println("Time:  ", x.metadata["time"], " s")
    println("ℓ:     ", as.ℓ)
    # println("l2(Δ): ", sum(abs2, Iterators.flatten(Iterators.flatten(Δ))))
    println("L∞(Δ): ", as.L∞_Δ)
    println()
end


@btime L∞_cust(Δ)
it = Iterators.flatten(Iterators.flatten(Δ))
@btime L∞_cust2(it)
val, Δ = ∇(l; get_output=true)(θ)
Δ = only(Δ)
@time sum(abs2, Iterators.flatten(Iterators.flatten(Δ)))
@time L∞(Δ)
@time for i in 1:100
    update!()
end

SSOF.Output!(o, om, d)
plt = status_plot(o, d)
png(plt, "after")

plt = plot_stellar_model_bases(model; display_plt=interactive);
png(plt, save_path * "model_star_basis.png")
plt = plot_stellar_model_scores(model; display_plt=interactive);
png(plt, save_path * "model_star_weights.png")
plt = plot_telluric_model_bases(model; display_plt=interactive);
png(plt, save_path * "model_tel_basis.png")
plt = plot_telluric_model_scores(model; display_plt=interactive);
png(plt, save_path * "model_tel_weights.png")
## looking at Nabla gradients (they seem correct)

function est_∇(f::Function, inputs::Vector{<:Real}; dif::Real=1e-7, inds::UnitRange=1:length(inputs))
    val = f(inputs)
    grad = Array{Float64}(undef, length(inputs))
    for i in inds
        hold = inputs[i]
        inputs[i] += dif
        grad[i] =  (f(inputs) - val) / dif
        inputs[i] = hold
    end
    return grad
end

ts = workspace.telstar
gn = ones(length(ts.p0))

ts.obj.f(ts.p0)
using BenchmarkTools
ts.obj.df(gn, ts.p0);
# @btime Zygote.gradient(ts.obj.f, ts.p0)
gn


ge = est_∇(ts.obj.f, ts.p0; inds = 1:10)
ge = est_∇(ts.obj.f, ts.p0; inds = 362290:362301)
