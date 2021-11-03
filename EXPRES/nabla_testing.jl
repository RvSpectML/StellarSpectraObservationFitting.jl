include("testing_header.jl")
model = reset_model()

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

mws = SSOF.ModelWorkspace(o, om, d);
as = mws.telstar.as;
callback() = println(as)
SSOF.train_OrderModel!(mws; print_stuff=true)
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
