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
	kwargs...
	)

	save = save_fn!=""

	# save_path = save_path_base * star * "/$(desired_order)/"
	@load data_fn n_obs data times_nu airmasses
	data.var[data.var .== 0] .= Inf

	# takes a couple mins now
	if isfile(save_fn) && !recalc
		println("using saved model at $save_fn")
	    @load save_fn model lm_tel lm_star
	    if model.metadata[:todo][:err_estimated]
	        @load save_fn rv_errors
	    end
	else
	    model = SSOF.OrderModel(data, instrument, desired_order, star; n_comp_tel=max_components, n_comp_star=max_components, oversamp=oversamp, dpca=dpca)
	    lm_tel, lm_star = SSOF.initializations!(model, data; kwargs...)
		if !use_reg
			SSOF.rm_regularization!(model)
			model.metadata[:todo][:reg_improved] = true
		end
		if save; @save save_fn model lm_tel lm_star end
	end
	return model, data, times_nu, airmasses, lm_tel, lm_star
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
	    if save; @save save_fn model end
	end
end

function improve_model!(mws::SSOF.ModelWorkspace; print_stuff::Bool=true, show_plot::Bool=false, save_fn::String="", kwargs...)
	save = save_fn!=""
	model = mws.om
    SSOF.train_OrderModel!(mws; print_stuff=print_stuff, kwargs...)  # 120s
	SSOF.finalize_scores!(mws)
    if show_plot; status_plot(mws) end
    if save; @save save_fn model end
end
function improve_model!(mws::SSOF.ModelWorkspace, airmasses::AbstractVector, times::AbstractVector; show_plot::Bool=false, kwargs...)
	improve_model!(mws; show_plot=show_plot, kwargs...)
	if show_plot; plot_model(mws, airmasses, times) end
end

function downsize_model(mws::SSOF.ModelWorkspace, times::AbstractVector, lm_tel::Vector{<:SSOF.LinearModel}, lm_star::Vector{<:SSOF.LinearModel}; save_fn::String="", decision_fn::String="", print_stuff::Bool=true, plots_fn::String="", iter::Int=50, ignore_regularization::Bool=true, kwargs...)
	model = mws.om

	if !model.metadata[:todo][:downsized]  # 1.5 hrs (for 9x9)
		save = save_fn!=""
		save_md = decision_fn!=""
		save_plots = plots_fn!=""

	    test_n_comp_tel = -1:size(model.tel.lm.M, 2)
	    test_n_comp_star = 0:size(model.star.lm.M, 2)
	    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
	    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
	    comp_intra_stds = zeros(length(test_n_comp_tel), length(test_n_comp_star))
		better_models = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
	    for (i, n_tel) in enumerate(test_n_comp_tel)
	        for (j, n_star) in enumerate(test_n_comp_star)
	            comp_ls[i, j], ks[i, j], comp_stds[i, j], comp_intra_stds[i, j], better_models[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], mws, times, lm_tel, lm_star; iter=iter, ignore_regularization=ignore_regularization)
	        end
	    end
	    n_comps_best, ℓ, aics, bics, best_ind = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, mws.d.var; return_inters=true, kwargs...)
	    if save_md; @save decision_fn comp_ls ℓ aics bics best_ind ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds better_models end

	    model_large = copy(model)
		mws_smol = _downsize_model(mws, n_comps_best, better_models[best_ind], lm_tel, lm_star; print_stuff=print_stuff, iter=iter, ignore_regularization=ignore_regularization)

		if save_plots
			diagnostics = [ℓ, aics, bics, comp_stds, comp_intra_stds]
			diagnostics_labels = ["ℓ", "AIC", "BIC", "RV std", "Intra-night RV std"]
			diagnostics_fn = ["l", "aic", "bic", "rv", "rv_intra"]
			mask = ℓ .< 0
			for i in 1:length(diagnostics)
				diagnostics[i][mask] .= Inf
				plt = component_test_plot(diagnostics[i], test_n_comp_tel, test_n_comp_star, ylabel=diagnostics_labels[i]);
				png(plt, plots_fn * diagnostics_fn[i] * "_choice.png")
			end
		end

		if save; @save save_fn model model_large end

		return mws_smol#, ℓ, aics, bics, comp_stds, comp_intra_stds
	end
	return mws
end
function _finish_downsizing(mws::SSOF.ModelWorkspace, model::SSOF.OrderModel; no_tels::Bool=false, kwargs...)
	if no_tels
		mws_smol = SSOF.FrozenTelWorkspace(model, mws.d)
		model.tel.lm.μ .= 1
	else
		mws_smol = typeof(mws)(model, mws.d)
	end
	SSOF.train_OrderModel!(mws_smol; kwargs...)  # 120s
	SSOF.finalize_scores!(mws_smol)
	return mws_smol
end
function _downsize_model(mws::SSOF.ModelWorkspace, n_comps::Vector{<:Int}, better_model::Int, lm_tel::Vector{<:SSOF.LinearModel}, lm_star::Vector{<:SSOF.LinearModel}; kwargs...)
	model = SSOF.downsize(mws.om, n_comps[1], n_comps[2])
	model.metadata[:todo][:downsized] = true
	if all(n_comps .> 0)
		SSOF._fill_model!(model, n_comps, better_model, lm_tel, lm_star)
		better_model==1 ?
			println("downsizing: used the $(n_comps[1]) telluric basis -> $(n_comps[2]) stellar basis initialization") :
			println("downsizing: used the $(n_comps[2]) stellar basis -> $(n_comps[1]) telluric basis initialization")
	end
	return _finish_downsizing(mws, model; no_tels=n_comps[1]<0, kwargs...)
end
function _downsize_model(mws::SSOF.ModelWorkspace, n_comps::Vector{<:Int}; kwargs...)
	model = SSOF.downsize(mws.om, n_comps[1], n_comps[2])
	model.metadata[:todo][:downsized] = true
	return _finish_downsizing(mws, model; no_tels=n_comps[1]<0, kwargs...)
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
