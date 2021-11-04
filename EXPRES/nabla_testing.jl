include("testing_header.jl")
# model = reset_model(; overrule=true)

## running the optimization outside the functions
using Optim
workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)
ow = workspace; osw = ow.telstar
optim_cb_local(x::OptimizationState) = SSOF.optim_cb(x; print_stuff=true)
options = Optim.Options(;iterations=100, f_tol=SSOF._f_tol_def, g_tol=SSOF._g_tol_def*sqrt(length(osw.p0)), callback=optim_cb_local)
@time result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)

## using ADAM to optimize

model = reset_model()
om = model; d = data; o = SSOF.Output(om, d)
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
plt = status_plot(o, d)
png(plt, "before")

mws = SSOF.TotalWorkspace(o, om, d)
train_OrderModel!(mws; print_stuff=true)
plt = status_plot(o, d)
png(plt, "after")

## looking at Nabla gradients (they seem correct)

function est_∇(f::Function, inputs; dif::Real=1e-7, inds::UnitRange=1:length(inputs))
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

using Nabla
mws = SSOF.TotalWorkspace(o, om, d)
f = mws.total.l
θ = mws.total.θ
Δ = only(∇(f)(θ))

val = f(θ)

dif = 1e-8
for i in 1:2
    for j in 1:3
        hold = θ[i][j][100]
        θ[i][j][100] += dif
        println((f(θ) - val) / dif, " ", Δ[i][j][100])
        θ[i][j][100] = hold
    end
end
