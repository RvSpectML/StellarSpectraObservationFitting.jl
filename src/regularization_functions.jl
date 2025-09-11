_reg_fields = [:reg_tel, :reg_star]


"""
    _eval_regularization(om, mws, training_inds, testing_inds; kwargs...)

Training `om` on the training data, evaluating `_loss()` on the testing data after optimizing the RVs and scores
"""
function _eval_regularization(om::OrderModel, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat; kwargs...)
    train = typeof(mws)(om, mws.d, training_inds)
    test = typeof(mws)(om, mws.d, testing_inds; only_s=true)
    train_OrderModel!(train; kwargs...) # trains feature vectors and scores at training times
    train_OrderModel!(test; shift_scores=false, kwargs...)  # trains scores at testing times
    return _loss(test)
end


"""
    eval_regularization(reg_fields, reg_key, reg_val, mws, training_inds, testing_inds; kwargs...)

Setting regularizaiton values for a copy of `mws.om` then training it on the training data and evaluating `_loss()` on the testing data after optimizing the RVs and scores
"""
function eval_regularization(reg_fields::Vector{Symbol}, reg_key::Symbol, reg_val::Real, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat; kwargs...)
    om = copy(mws.om)
    for field in reg_fields
        getfield(om, field)[reg_key] = reg_val
    end
    return _eval_regularization(om, mws, training_inds, testing_inds; kwargs...)
end


"""
    fit_regularization_helper!(reg_fields, reg_key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=10e3, cullable=Symbol[], robust_start=true, thres=8, kwargs...)

Setting `reg_key` values in each Dict in mws.om.x (where x is each symbol in ``reg_fields``) for a copy of `mws.om` then training it on the training data and evaluating `_loss()` on the testing data after optimizing the RVs and scores
"""
function fit_regularization_helper!(reg_fields::Vector{Symbol}, reg_key::Symbol, before_ℓ::Real, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat, test_factor::Real, reg_min::Real, reg_max::Real; start::Real=10e3, cullable::Vector{Symbol}=Symbol[], robust_start::Bool=true, thres::Real=8, kwargs...)
    
    # only do anything if 
    if haskey(getfield(mws.om, reg_fields[1]), reg_key)

        om = mws.om
        @assert 0 < reg_min < reg_max < Inf
        ℓs = Array{Float64}(undef, 2)

        # check starting from `reg_min`, `start`, and `reg_max`
        if robust_start
            starting_ℓs =
                [eval_regularization(reg_fields, reg_key, reg_min, mws, training_inds, testing_inds; kwargs...),
                eval_regularization(reg_fields, reg_key, start, mws, training_inds, testing_inds; kwargs...),
                eval_regularization(reg_fields, reg_key, reg_max, mws, training_inds, testing_inds; kwargs...)]
            start_ind = argmin(starting_ℓs)
            if reg_key in cullable
                if starting_ℓs[start_ind] > before_ℓ
                    for field in reg_fields
                        getfield(mws.om, field)[reg_key] = 0.
                    end
                    println("a course search suggests $(reg_fields[1])[:$reg_key] isn't useful, so setting it to 0")
                    return before_ℓ
                end
            end
            if start_ind==1
                reg_hold = [reg_min, reg_min*test_factor]
                start_ℓ = ℓs[1] = starting_ℓs[1]
                start = reg_min
                ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds; kwargs...)
            elseif start_ind==2
                reg_hold = [start, start*test_factor]
                start_ℓ = ℓs[1] = starting_ℓs[2]
                ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds; kwargs...)
            else
                reg_hold = [reg_max/test_factor, reg_max]
                start_ℓ = ℓs[2] = starting_ℓs[3]
                start = reg_max
                ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds; kwargs...)
            end
        else
            reg_hold = [start, start*test_factor]
            start_ℓ = ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds; kwargs...)
            ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds; kwargs...)
        end

        # need to try decreasing regularization
        if ℓs[2] > ℓs[1]
            off_edge = reg_min < reg_hold[1]
            while (ℓs[2] > ℓs[1]) && (reg_min < reg_hold[1] < reg_max)
                # println("trying a lower regularization")
                ℓs[2] = ℓs[1]
                reg_hold ./= test_factor
                ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds; kwargs...)
            end
            if off_edge
                last_checked_ℓ, end_ℓ = choose_reg_and_ℓ(reg_fields, om, reg_key, reg_hold, ℓs, 2)
            else
                last_checked_ℓ, end_ℓ = choose_reg_and_ℓ(reg_fields, om, reg_key, reg_hold, ℓs, 1)
            end
        # need to try increasing regularization
        else
            off_edge = reg_hold[2] < reg_max
            while (ℓs[1] > ℓs[2]) && (reg_min < reg_hold[2] < reg_max)
                # println("trying a higher regularization")
                ℓs[1] = ℓs[2]
                reg_hold .*= test_factor
                ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds; kwargs...)
            end
            if off_edge
                last_checked_ℓ, end_ℓ = choose_reg_and_ℓ(reg_fields, om, reg_key, reg_hold, ℓs, 1)
            else
                last_checked_ℓ, end_ℓ = choose_reg_and_ℓ(reg_fields, om, reg_key, reg_hold, ℓs, 2)
            end
        end

        println("$(reg_fields[1])[:$reg_key] : $start -> $(getfield(mws.om, reg_fields[1])[reg_key])")
        if isapprox(end_ℓ, last_checked_ℓ; rtol=1e-6)
            @warn "weak local minimum $end_ℓ vs. $last_checked_ℓ"
        end

        println("$(reg_fields[1])[:$reg_key] χ²: $start_ℓ -> $end_ℓ (" * ratio_clarifier_string(end_ℓ/start_ℓ) * ")")
        println("overall χ² change: $before_ℓ -> $end_ℓ (" * ratio_clarifier_string(end_ℓ/before_ℓ) * ")")

        # removing the regularization term if it is significantly bad
        if end_ℓ > ((1 + thres/100) * before_ℓ)
            for field in reg_fields
                getfield(mws.om, field)[reg_key] = 0.
            end
            println("$(reg_fields[1])[:$reg_key] significantly increased the χ² (by more than $thres%), so setting it to 0")
            return before_ℓ
        end
        return end_ℓ
    end
    return before_ℓ
end


"""
    choose_reg_and_ℓ(reg_fields, om, reg_key, reg_hold, ℓs, j)

Set the `reg_key` regularization for `om` once the local minimum is found
"""
function choose_reg_and_ℓ(reg_fields::Vector{Symbol}, om::OrderModel, reg_key::Symbol, reg_hold::Vector{<:Real}, ℓs::Vector{<:Real}, j::Int)
    jis1 = j == 1
    @assert jis1 || j == 2
    for field in reg_fields
        getfield(om, field)[reg_key] = reg_hold[j]
    end
    jis1 ? (return ℓs[2], ℓs[1]) : (return ℓs[1], ℓs[2])
end


"""
    ratio_clarifier_string(ratio)

Convert `ratio` to a nice 3 digit-rounded string
"""
function ratio_clarifier_string(ratio::Real)
    x = round(ratio; digits=3)
    if x == 1.
        if ratio == 1; return "=1.0" end
        ratio > 1 ? (return ">1.0") : (return "<1.0")
    else
        return string(x)
    end
end


_key_list = [:GP_μ, :L2_μ, :L1_μ, :L1_μ₊_factor, :GP_M, :L2_M, :L1_M, :shared_M]
_key_list_μ = [:GP_μ, :L2_μ, :L1_μ, :L1_μ₊_factor]
_key_list_fit = [:GP_μ, :L2_μ, :L1_μ, :GP_M, :L2_M, :L1_M]
_key_list_bases = [:GP_M, :L2_M, :L1_M, :shared_M]
_key_list_model = [:GP_μ, :L2_μ, :L1_μ, :L2_h2o, :L1_h2o, :L2_oxy, :L1_oxy]


"""
    check_for_valid_regularization(reg)

Make sure all the keys in `reg` are in SSOF._key_list
"""
function check_for_valid_regularization(reg::Dict{Symbol, <:Real})
    for i in keys(reg)
        @assert i in _key_list "The requested regularization isn't valid"
    end
end

min_reg = 1e-1 #1e-3
max_reg = 1e7 #1e12
min_reg_model = 1e0
max_reg_model = 1e12


"""
    fit_regularization!(mws, testing_inds; key_list=_key_list_fit, share_regs=false, kwargs...)

Fit all of the regularization values in `key_list` for the model in `mws`
"""
function fit_regularization!(mws::ModelWorkspace, testing_inds::AbstractVecOrMat; key_list::Vector{Symbol}=_key_list_fit, share_regs::Bool=false, ignore_μ_tel::Bool=false, kwargs...)
    om = mws.om
    n_obs = size(mws.d.flux, 2)
    training_inds = [i for i in 1:n_obs if !(i in testing_inds)]
    check_for_valid_regularization(om.reg_tel)
    check_for_valid_regularization(om.reg_star)
    if share_regs; @assert keys(om.reg_tel) == keys(om.reg_star) end
    if share_regs; @assert !ignore_μ_tel end
    hold_tel = copy(default_reg_star_full)
    hold_star = copy(default_reg_tel_full)
    initial_tel = copy(om.reg_tel)
    copy_dict!(hold_tel, om.reg_tel)
    copy_dict!(hold_star, om.reg_star)
    zero_regularization(om)
    if ignore_μ_tel
        for key in _key_list_μ
            if key in keys(initial_tel)
                om.reg_tel[key] = initial_tel[key]
            end
        end
    end
    println("starting regularization searches")
    before_ℓ = _eval_regularization(copy(mws.om), mws, training_inds, testing_inds)
    println("initial training χ²: $before_ℓ")
    for key in key_list
        if key == :L1_μ₊_factor
            test_factor, reg_min, reg_max = 1.2, 1e-1, 1e1
        else
            test_factor, reg_min, reg_max = 10, min_reg, max_reg
        end
        if share_regs
            before_ℓ = fit_regularization_helper!(_reg_fields, key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_tel[key], kwargs...)
        else
            if (!(key in _key_list_bases)) || is_time_variable(om.star)
                before_ℓ = fit_regularization_helper!([:reg_star], key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_star[key], kwargs...)
            end
            if ((!(key in _key_list_bases)) || is_time_variable(om.tel)) && !((key in _key_list_μ) && ignore_μ_tel)
                before_ℓ = fit_regularization_helper!([:reg_tel], key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_tel[key], kwargs...)
            end
        end
    end
end


"""
    fit_regularization_model!(mws, testing_inds; key_list=_key_list_model, kwargs...)

Fit all of the regularization values in `key_list` for `mws.om.reg_model`
"""
function fit_regularization_model!(mws::ModelWorkspace, testing_inds::AbstractVecOrMat; key_list::Vector{Symbol}=_key_list_model, kwargs...)
    om = mws.om
    n_obs = size(mws.d.flux, 2)
    training_inds = [i for i in 1:n_obs if !(i in testing_inds)]
    hold_reg = copy(om.reg_model)
    for (key, value) in om.reg_model
        om.reg_model[key] = 0
    end
    println("starting regularization searches for mws.om.reg_model")
    before_ℓ = _eval_regularization(copy(mws.om), mws, training_inds, testing_inds)
    println("initial training χ²: $before_ℓ")
    test_factor, reg_min, reg_max = 10, min_reg_model, max_reg_model
    for key in key_list
        if haskey(hold_reg, key)
            before_ℓ = fit_regularization_helper!([:reg_model], key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_reg[key], kwargs...)
        end
    end
end



"""
    fit_regularization!(mws; verbose=true, testing_ratio=0.33, careful_first_step=true, speed_up=false, kwargs...)

Find the best fit model without regularization then fit all of the regularization values in `key_list` for the model in `mws`
"""
function fit_regularization!(mws::ModelWorkspace; verbose::Bool=true, testing_ratio::Real=0.33, careful_first_step::Bool=true, speed_up::Bool=false,
                             fit_reg_model::Bool=true, kwargs...)
	# if mws.om.metadata[:todo][:reg_improved]
    n_obs = size(mws.d.flux, 2)
    train_OrderModel!(mws; verbose=verbose, ignore_regularization=true, careful_first_step=careful_first_step, speed_up=speed_up)
    n_obs_test = Int(round(testing_ratio * n_obs))
    test_start_ind = max(1, Int(round(rand() * (n_obs - n_obs_test))))
    testing_inds = test_start_ind:test_start_ind+n_obs_test-1
    fit_regularization!(mws, testing_inds; kwargs...)
    if mws.om.metadata[:tel_prior] && fit_reg_model
        fit_regularization_model!(mws, testing_inds; kwargs...)
    end
    mws.om.metadata[:todo][:reg_improved] = true
	# end
end