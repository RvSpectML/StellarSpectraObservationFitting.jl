using Statistics  # mean function
using Base.Threads

# getting error bars
function estimate_σ_curvature_helper(x::AbstractVecOrMat, ℓ::Function; n::Int=7, use_gradient::Bool=false, multithread::Bool=nthreads() > 3, print_every::Int=10, kwargs...)
	σs = Array{Float64}(undef, length(x))
	if !multithread
		x_test = Array{Float64}(undef, n)
		ℓs = Array{Float64}(undef, n)
	end
	if use_gradient; g = ∇(ℓ) end
	_std = std(x)
	if multithread
		nchains = nthreads()
		schedule = collect(Iterators.partition(eachindex(x), Int(ceil(length(x)/nchains))))
		Threads.@threads for i in 1:nchains
		# ThreadsX.foreach(1:nchains) do i
			local _todo = copy(schedule[i])
			local _σs = Array{Float64}(undef, length(_todo))
			local _x = copy(x)
			local _x_test = Array{Float64}(undef, n)
			local _ℓs = Array{Float64}(undef, n)
			for ii in eachindex(_todo)
				k = _todo[ii]
				_x_test .= _x[k] .+ LinRange(-_std, _std, n)
				for j in 1:n
					_x[k] = _x_test[j]
					if use_gradient
						_ℓs[j] = only(g(_x))[k]
					else
						_ℓs[j] = ℓ(_x)
					end
				end
				estimate_σ_curvature_helper_finalizer!(_σs, _ℓs, _x_test, ii; use_gradient=use_gradient, print_every=100000, kwargs...)
			end
			σs[_todo] .= _σs
		end
	else
		for i in eachindex(x)
			hold = x[i]
			# x_test[:] = x[i] .+ LinRange(-_std/1e3, _std/1e3, n)
			x_test[:] = x[i] .+ LinRange(-_std, _std, n)
			for j in 1:n
				x[i] = x_test[j]
				if use_gradient
					ℓs[j] = only(g(x))[i]
				else
					ℓs[j] = ℓ(x)
				end
			end
			x[i] = hold
			# println("$i: ", ℓs .- ℓs[Int(round(n//2))])
			estimate_σ_curvature_helper_finalizer!(σs, ℓs, x_test, i; use_gradient=use_gradient, print_every=print_every, kwargs...)
		end
	end
	return reshape(σs, size(x))
end

function estimate_σ_curvature_helper_finalizer!(σs::AbstractVecOrMat, _ℓs::AbstractVector, x_test::AbstractVector, i::Int; use_gradient::Bool=false, param_str::String="", print_every::Int=10, verbose::Bool=false, show_plots::Bool=false, )
	if use_gradient
		poly_f = ordinary_lst_sq_f(_ℓs, 1; x=x_test)
		σs[i] = sqrt(1 / poly_f.w[2])
		max_dif = maximum(abs.((poly_f.(x_test)./_ℓs) .- 1))
		if verbose; println("∇_$i: $(poly_f.w[1] + poly_f.w[2] * x[i])") end
	else
		poly_f = ordinary_lst_sq_f(_ℓs, 2; x=x_test)
		σs[i] = sqrt(1 / (2 * poly_f.w[3]))
		max_dif = maximum(abs.((poly_f.(x_test)./_ℓs) .- 1))
		if verbose; println("∇_$i: $(poly_f.w[2] + 2 * poly_f.w[3] * x[i])") end
	end
	if show_plots
		plt = scatter(x_test, _ℓs; label="ℓ")
		plot!(x_test, poly_f.(x_test); label="polynomial fit")
		display(plt)
	end
	if max_dif > 1e-2; @warn param_str * "_σ[$i] misfit at $(round(100*max_dif; digits=2))% level" end
	if i%print_every==0; println("done with $i/$(length(σs)) " * param_str * "_σ estimates") end
	# println("done with $i/$(length(σs)) " * param_str * "_σ estimates")
end

function estimate_σ_curvature(mws::ModelWorkspace; kwargs...)

	model = mws.om
	time_var_tel = is_time_variable(model.tel)
	time_var_star = is_time_variable(model.star)

	typeof(model) <: OrderModelDPCA ? rvs = copy(model.rv.lm.s) : rvs = copy(model.rv)
	ℓ_rv(x) = _loss(mws.o, model, mws.d; rv=x) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
	rvs_σ = estimate_σ_curvature_helper(rvs, ℓ_rv; param_str="rv", kwargs...)
	if typeof(model) <: OrderModelDPCA
		rvs = vec(rvs)
		rvs .*= -light_speed_nu
		rvs_σ .*= light_speed_nu
	end

	if time_var_tel
		ℓ_tel(x) = (_loss(mws.o, model, mws.d; tel=vec(model.tel.lm)) + model_s_prior(model.tel.lm.s, model.reg_tel)) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
		tel_s_σ = estimate_σ_curvature_helper(model.tel.lm.s, ℓ_tel; param_str="tel_s", kwargs...)
	else
		tel_s_σ = nothing
	end

	if time_var_star
		ℓ_star(x) = (_loss(mws.o, model, mws.d; star=vec(model.star.lm)) + model_s_prior(model.star.lm.s, model.reg_star)) / 2  # factor of 2 makes curvature estimates correct (χ² -> data fit part of multivariate Gaussian)
		star_s_σ = estimate_σ_curvature_helper(model.star.lm.s, ℓ_star; param_str="star_s", kwargs...)
	else
		star_s_σ = nothing
	end

	model.metadata[:todo][:err_estimated] = true

	return rvs, rvs_σ, tel_s_σ, star_s_σ

end


function estimate_σ_bootstrap_reducer(shaper::AbstractArray, holder::AbstractArray, reducer::Function)
	result = Array{Float64}(undef, size(shaper, 1), size(shaper, 2))
	for i in axes(shaper, 1)
		result[i, :] .= vec(reducer(view(holder, :, i, :); dims=1))
	end
	return result
end

function estimate_σ_bootstrap_helper!(rv_holder::AbstractMatrix, tel_holder, star_holder, i::Int, mws::ModelWorkspace, data_noise::AbstractMatrix, n::Int; verbose::Bool=true)
	time_var_tel = is_time_variable(mws.om.tel)
	time_var_star = is_time_variable(mws.om.star)
	_mws = typeof(mws)(copy(mws.om), copy(mws.d))
	_mws.d.flux .= mws.d.flux .+ (data_noise .* randn(size(mws.d.var)))
	improve_model!(_mws; iter=50, verbose=false)
	rv_holder[i, :] = rvs(_mws.om)
	if time_var_tel
		tel_holder[i, :, :] .= _mws.om.tel.lm.s
	end
	if time_var_star
		star_holder[i, :, :] .= _mws.om.star.lm.s
	end
	if (verbose && i%10==0); println("done with $i/$n bootstraps") end
end

function estimate_σ_bootstrap(mws::ModelWorkspace; n::Int=50, return_holders::Bool=false, recalc_mean::Bool=false, multithread::Bool=nthreads() > 3, verbose::Bool=true)
	mws.d.var[mws.d.var.==Inf] .= 0
	data_noise = sqrt.(mws.d.var)
	mws.d.var[mws.d.var.==0] .= Inf

	typeof(mws.om) <: OrderModelWobble ?
		rv_holder = Array{Float64}(undef, n, length(mws.om.rv)) :
		rv_holder = Array{Float64}(undef, n, length(mws.om.rv.lm.s))

	time_var_tel = is_time_variable(mws.om.tel)
	time_var_star = is_time_variable(mws.om.star)
	time_var_tel ?
		tel_holder = Array{Float64}(undef, n, size(mws.om.tel.lm.s, 1), size(mws.om.tel.lm.s, 2)) :
		tel_holder = nothing
	time_var_star ?
		star_holder = Array{Float64}(undef, n, size(mws.om.star.lm.s, 1), size(mws.om.star.lm.s, 2)) :
		star_holder = nothing
	if multithread
		@threads for i in 1:n
		# # using Polyester  # same performance
		# @batch per=core for i in 1:n
		# using ThreadsX  # tiny bit better performance
		# ThreadsX.foreach(1:n) do i
			estimate_σ_bootstrap_helper!(rv_holder, tel_holder, star_holder, i, mws, data_noise, n; verbose=false)
		end
	else
		for i in 1:n
			estimate_σ_bootstrap_helper!(rv_holder, tel_holder, star_holder, i, mws, data_noise, n; verbose=verbose)
		end
	end
	recalc_mean ? _rvs = vec(mean(rv_holder; dims=1)) : _rvs = rvs(mws.om)
	rvs_σ = vec(std(rv_holder; dims=1))

	if time_var_tel
		recalc_mean ?
			tel_s = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, mean) :
			tel_s = mws.om.tel.lm.s
		tel_s_σ = estimate_σ_bootstrap_reducer(mws.om.tel.lm.s, tel_holder, std)
	else
		tel_s = nothing
		tel_s_σ = nothing
	end
	if time_var_star
		recalc_mean ?
			star_s = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, mean) :
			star_s = mws.om.star.lm.s
		star_s_σ = estimate_σ_bootstrap_reducer(mws.om.star.lm.s, star_holder, std)
	else
		star_s = nothing
		star_s_σ = nothing
	end
	mws.om.metadata[:todo][:err_estimated] = true

	if return_holders
		return _rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ, rv_holder, tel_holder, star_holder
	else
		return _rvs, rvs_σ, tel_s, tel_s_σ, star_s, star_s_σ
	end

end

