_reg_fields = [:reg_tel, :reg_star]


function _eval_regularization(om::OrderModel, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat; kwargs...)
    train = typeof(mws)(om, mws.d, training_inds)
    test = typeof(mws)(om, mws.d, testing_inds; only_s=true)
    # train_OrderModel!(train; iter=50, kwargs...) # trains basis vectors and (scores at training time)
    # train_OrderModel!(test; shift_scores=false, iter=50, kwargs...)  # trains scores at testing times
    train_OrderModel!(train; kwargs...) # trains basis vectors and (scores at training time)
    train_OrderModel!(test; shift_scores=false, kwargs...)  # trains scores at testing times
    return _loss(test)
end
function eval_regularization(reg_fields::Vector{Symbol}, reg_key::Symbol, reg_val::Real, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat; kwargs...)
    om = copy(mws.om)
    for field in reg_fields
        getfield(om, field)[reg_key] = reg_val
    end
    return _eval_regularization(om, mws, training_inds, testing_inds; kwargs...)
end

function fit_regularization_helper!(reg_fields::Vector{Symbol}, reg_key::Symbol, before_ℓ::Real, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat, test_factor::Real, reg_min::Real, reg_max::Real; start::Real=10e3, try_to_cull::Bool=true, kwargs...)
    if haskey(getfield(mws.om, reg_fields[1]), reg_key)

        om = mws.om
        @assert 0 < reg_min < reg_max < Inf
        ℓs = Array{Float64}(undef, 2)
        if try_to_cull
            starting_ℓs =
                [eval_regularization(reg_fields, reg_key, reg_min, mws, training_inds, testing_inds; kwargs...),
                eval_regularization(reg_fields, reg_key, start, mws, training_inds, testing_inds; kwargs...),
                eval_regularization(reg_fields, reg_key, reg_max, mws, training_inds, testing_inds; kwargs...)]
            start_ind = argmin(starting_ℓs)
            if starting_ℓs[start_ind] > before_ℓ && reg_key!=:GP_μ
                println("a course search suggests $(reg_fields[1])[:$reg_key] isn't useful, so setting it to 0")
                return before_ℓ
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
            while (ℓs[2] > ℓs[1]) && (reg_min < reg_hold[1] < reg_max)
                # println("trying a lower regularization")
                ℓs[2] = ℓs[1]
                reg_hold ./= test_factor
                ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds; kwargs...)
            end
            for field in reg_fields
                getfield(om, field)[reg_key] = reg_hold[2]
            end
            last_checked_ℓ, end_ℓ = ℓs
        # need to try increasing regularization
        else
            while (ℓs[1] > ℓs[2]) && (reg_min < reg_hold[2] < reg_max)
                # println("trying a higher regularization")
                ℓs[1] = ℓs[2]
                reg_hold .*= test_factor
                ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds; kwargs...)
            end
            for field in reg_fields
                getfield(om, field)[reg_key] = reg_hold[1]
            end
            end_ℓ, last_checked_ℓ = ℓs
        end

        println("$(reg_fields[1])[:$reg_key] : $start -> $(getfield(mws.om, reg_fields[1])[reg_key])")
        if isapprox(end_ℓ, last_checked_ℓ; rtol=1e-6)
            @warn "weak local minimum $end_ℓ vs. $last_checked_ℓ"
        end
        println("$(reg_fields[1])[:$reg_key] χ²: $start_ℓ -> $end_ℓ ($(round(end_ℓ/start_ℓ; digits=3)))")
        println("overall χ² change: $before_ℓ -> $end_ℓ ($(round(end_ℓ/before_ℓ; digits=3)))")
        if end_ℓ > (1.1 * before_ℓ)
            for field in reg_fields
                getfield(mws.om, field)[reg_key] = 0.
            end
            println("$(reg_fields[1])[:$reg_key] significantly increased the χ², so setting it to 0")
            return before_ℓ
        end
        return end_ℓ
    end
    return before_ℓ
end


_key_list = [:GP_μ, :L2_μ, :L1_μ, :L1_μ₊_factor, :GP_M, :L2_M, :L1_M, :shared_M]
_key_list_fit = [:GP_μ, :L2_μ, :L1_μ, :GP_M, :L2_M, :L1_M]
_key_list_bases = [:GP_M, :L2_M, :L1_M, :shared_M]
function check_for_valid_regularization(reg::Dict{Symbol, <:Real})
    for i in keys(reg)
        @assert i in _key_list "The requested regularization isn't valid"
    end
end


function fit_regularization!(mws::ModelWorkspace, testing_inds::AbstractVecOrMat; key_list::Vector{Symbol}=_key_list_fit, share_regs::Bool=false, kwargs...)
    om = mws.om
    n_obs = size(mws.d.flux, 2)
    training_inds = [i for i in 1:n_obs if !(i in testing_inds)]
    check_for_valid_regularization(om.reg_tel)
    check_for_valid_regularization(om.reg_star)
    if share_regs; @assert keys(om.reg_tel) == keys(om.reg_star) end
    hold_tel = copy(default_reg_star_full)
    hold_star = copy(default_reg_tel_full)
    copy_dict!(hold_tel, om.reg_tel)
    copy_dict!(hold_star, om.reg_star)
    zero_regularization(om)
    println("starting regularization searches")
    before_ℓ = _eval_regularization(copy(mws.om), mws, training_inds, testing_inds)
    println("initial training χ²: $before_ℓ")
    for key in key_list
        if key == :L1_μ₊_factor
            test_factor, reg_min, reg_max = 1.2, 1e-1, 1e1
        else
            test_factor, reg_min, reg_max = 10, 1e-3, 1e12
        end
        if share_regs
            before_ℓ = fit_regularization_helper!(_reg_fields, key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_tel[key], kwargs...)
        else
            if (!(key in _key_list_bases)) || is_time_variable(om.star)
                before_ℓ = fit_regularization_helper!([:reg_star], key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_star[key], kwargs...)
            end
            if (!(key in _key_list_bases)) || is_time_variable(om.tel)
                before_ℓ = fit_regularization_helper!([:reg_tel], key, before_ℓ, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; start=hold_tel[key], kwargs...)
            end
        end
    end
end
