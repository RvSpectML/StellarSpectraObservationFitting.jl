## Some helpful analysis functions
import StellarSpectraObservationFitting as SSOF
using JLD2
using Statistics
import StatsBase

valid_optimizers = ["adam", "l-bfgs", "frozen-tel"]

function create_model(
	data_fn::String,
	desired_order::Int,
	instrument::String,
	star::String;
	max_components::Int=5,
	oversamp::Bool=true,
	use_reg::Bool=true,
	save_fn::String="",
	recalc::Bool=false,
	dpca::Bool=true,
	seed::Union{SSOF.OrderModel, Nothing}=nothing
	)

	save = save_fn!=""
	seeded = !isnothing(seed)

	# save_path = save_path_base * star * "/$(desired_order)/"
	@load data_fn n_obs data times_nu airmasses
	data.var[data.var .== 0] .= Inf

	# takes a couple mins now
	if isfile(save_fn) && !recalc
		println("using saved model at $save_fn")
	    @load save_fn model
	    if model.metadata[:todo][:err_estimated]
	        @load save_fn rv_errors
	    end
	else
	    model = SSOF.OrderModel(data, instrument, desired_order, star; n_comp_tel=max_components, n_comp_star=max_components, oversamp=oversamp, dpca=dpca)
	    model, _, _, _ = SSOF.initialize!(model, data; seed=seed)
		if !use_reg
			SSOF.rm_regularization!(model)
			model.metadata[:todo][:reg_improved] = true
		end
		if save; @save save_fn model end
	end
	return model, data, times_nu, airmasses
end

function create_workspace(model, data, opt::String)
	@assert opt in valid_optimizers
	if opt == "l-bfgs"
		mws = SSOF.OptimWorkspace(model, data)
	elseif opt == "frozen-tel"
		mws = SSOF.FrozenTelWorkspace(model, data)
	else
		mws = SSOF.TotalWorkspace(model, data)
	end
	return mws
end

function improve_regularization!(mws::SSOF.ModelWorkspace; redo::Bool=false, print_stuff::Bool=true, testing_ratio::Real=0.25, save_fn::String="")

	save = save_fn!=""

	model = mws.om
	if redo || !model.metadata[:todo][:reg_improved]  # 27 mins
		@assert 0 < testing_ratio < 1
		n_obs = size(mws.d.flux, 2)

	    SSOF.train_OrderModel!(mws; print_stuff=print_stuff, ignore_regularization=true)  # 45s
	    n_obs_test = Int(round(testing_ratio * n_obs))
	    test_start_ind = max(1, Int(round(rand() * (n_obs - n_obs_test))))
	    testing_inds = test_start_ind:test_start_ind+n_obs_test-1
	    SSOF.fit_regularization!(mws, testing_inds)
	    model.metadata[:todo][:reg_improved] = true
	    model.metadata[:todo][:optimized] = false
	    if save; @save save_fn model end
	end
end

function improve_model!(mws::SSOF.ModelWorkspace; print_stuff::Bool=true, show_plot::Bool=false, save_fn::String="", kwargs...)
	save = save_fn!=""
	model = mws.om
	if !model.metadata[:todo][:optimized]
	    SSOF.train_OrderModel!(mws; print_stuff=print_stuff, kwargs...)  # 120s
		SSOF.finalize_scores!(mws)
	    if show_plot; status_plot(mws) end
	    model.metadata[:todo][:optimized] = true
	    if save; @save save_fn model end
	end
end
function improve_model!(mws::SSOF.ModelWorkspace, airmasses::AbstractVector, times::AbstractVector; show_plot::Bool=false, kwargs...)
	improve_model!(mws; show_plot=show_plot, kwargs...)
	if show_plot; plot_model(mws, airmasses, times) end
end

function downsize_model(mws::SSOF.ModelWorkspace, times; save_fn::String="", decision_fn::String="", print_stuff::Bool=true, plots_fn::String="", kwargs...)
	save = save_fn!=""
	save_md = decision_fn!=""
	save_plots = plots_fn!=""
	model = mws.om

	if !model.metadata[:todo][:downsized]  # 1.5 hrs (for 9x9)
	    test_n_comp_tel = 0:size(model.tel.lm.M, 2)
	    test_n_comp_star = 0:size(model.star.lm.M, 2)
	    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
	    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_intra_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    for (i, n_tel) in enumerate(test_n_comp_tel)
	        for (j, n_star) in enumerate(test_n_comp_star)
	            comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], mws, times)
	        end
	    end
	    n_comps_best, ℓ, aics, bics = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, mws.d.var; return_inters=true, kwargs...)
	    if save_md; @save decision_fn comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds end

	    model_large = copy(model)
		mws_smol = _downsize_model(mws, n_comps_best[1], n_comps_best[2]; print_stuff=print_stuff)
	    model = mws_smol.om
	    model.metadata[:todo][:downsized] = true
	    model.metadata[:todo][:reg_improved] = true
	    model.metadata[:todo][:optimized] = true
	    if save; @save save_fn model model_large end

		if save_plots
			diagnostics = [ℓ, aics, bics, comp_stds, comp_intra_stds]
			diagnostics_labels = ["ℓ", "AIC", "BIC", "RV std", "Intra-night RV std"]
			diagnostics_fn = ["l", "aic", "bic", "rv", "rv_intra"]
			for i in 1:length(diagnostics)
				plt = component_test_plot(diagnostics[i], test_n_comp_tel, test_n_comp_star, ylabel=diagnostics_labels[i]);
				png(plt, plots_fn * diagnostics_fn[i] * "_choice.png")
			end
		end
		return mws_smol, ℓ, aics, bics, comp_stds, comp_intra_stds
	end
end
function _downsize_model(mws::SSOF.ModelWorkspace, n_comps_tel::Int, n_comps_star::Int; print_stuff::Bool=true)
	model = SSOF.downsize(mws.om, n_comps_tel, n_comps_star)
	mws_smol = typeof(mws)(model, mws.d)
	SSOF.train_OrderModel!(mws_smol; print_stuff=print_stuff)  # 120s
	SSOF.finalize_scores!(mws_smol)
	return mws_smol
end

function estimate_errors(mws::SSOF.ModelWorkspace; save_fn="")
	save = save_fn!=""
	model = mws.om
	data = mws.d
	if !model.metadata[:todo][:err_estimated] # 25 mins
	    data.var[data.var.==Inf] .= 0
	    data_noise = sqrt.(data.var)
	    data.var[data.var.==0] .= Inf

		rvs = SSOF.rvs(model)
	    n = 50
	    typeof(mws.om) <: SSOF.OrderModelWobble ?
		 	rv_holder = Array{Float64}(undef, n, length(model.rv)) :
			rv_holder = Array{Float64}(undef, n, length(model.rv.lm.s))

	    _mws = typeof(mws)(copy(model), copy(data))
	    _mws_score_finalizer() = SSOF.finalize_scores_setup(_mws)
	    for i in 1:n
	        _mws.d.flux .= data.flux .+ (data_noise .* randn(size(data.var)))
	        SSOF.train_OrderModel!(_mws; iter=50)
	        _mws_score_finalizer()
	        rv_holder[i, :] = SSOF.rvs(_mws.om)
	    end
	    rv_errors = vec(std(rv_holder; dims=1))
	    model.metadata[:todo][:err_estimated] = true
	    if save; @save save_fn model rvs rv_errors end
		return rvs, rv_errors
	else
		println("loading rvs")
		if save; @load save_fn rvs rv_errors end
		return rvs, rv_errors
	end
end
