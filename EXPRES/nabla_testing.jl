include("testing_header.jl")
# model = reset_model(; overrule=true)

## Optim Version

ows = SSOF.OptimWorkspace(model, data)
res = SSOF.train_OrderModel!(ows; print_stuff=true)
res

## using ADAM to optimize

model2 = reset_model()
# om = model; d = data; o = SSOF.Output(om, d)
# SSOF_path = dirname(dirname(pathof(SSOF)))
# include(SSOF_path * "/src/_plot_functions.jl")
# status_plot(o, d)
# png(plt, "before")

mws = SSOF.TelStarWorkspace(model2, data)
@time SSOF.train_OrderModel!(mws; print_stuff=true)
# status_plot(o, d)
# png(plt, "after")

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

## Can we use views in Nabla gradients
using Nabla
smol = SSOF.TotalWorkspace(mws.om, mws.d, 1:10)
Δ = only(∇(smol.total.l)(smol.total.θ))

smol = SSOF.TotalWorkspace(mws.om, mws.d, 1:10; only_s=true)
Δ = only(∇(smol.total.l)(smol.total.θ))

## Looking at different optimization paths

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
function sanity(mws::SSOF.ModelWorkspace)
    status_plot(mws)
    plot_stellar_model_bases(mws.om)
    plot_stellar_model_scores(mws.om)
    plot_telluric_model_bases(mws.om)
    plot_telluric_model_scores(mws.om)
    println("mean(abs(rv)): ", mean(abs, SSOF.rvs(mws.om)))
end
function new_model()
    om = reset_model()
    # SSOF.copy_dict!(om.reg_star, SSOF.default_reg_star)
    return om
end
function current_loss(mws::SSOF.ModelWorkspace)
    l = SSOF.loss_func(mws; include_priors=true)
    l()
end

d = data
d = SSOF.GenericData(data)

model1 = new_model()
mws1 = SSOF.OptimWorkspace(model1, d)


# sanity(mws1)
@time res = SSOF.train_OrderModel!(mws1; print_stuff=true)
current_loss(mws1)
sanity(mws1)

model2 = new_model()
mws2 = SSOF.TelStarWorkspace(model2, d)
@time SSOF.train_OrderModel!(mws2; print_stuff=true)
current_loss(mws2)
sanity(mws2)

model3 = new_model()
mws3 = SSOF.TotalWorkspace(model3, d)
@time SSOF.train_OrderModel!(mws3; print_stuff=true)
current_loss(mws3)
sanity(mws3)
