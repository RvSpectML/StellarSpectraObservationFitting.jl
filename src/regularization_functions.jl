_reg_fields = [:reg_tel, :reg_star]

function eval_regularization(reg_fields::Vector{Symbol}, reg_key::Symbol, reg_val::Real, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat)
    om, d = copy(mws.om), mws.d
    for field in reg_fields
        getfield(om, field)[reg_key] = reg_val
    end
    train = typeof(mws)(om, d, training_inds)
    test = typeof(mws)(om, d, testing_inds; only_s=true)
    train_OrderModel!(train) # trains basis vectors and (scores at training time)
    train_OrderModel!(test)  # trains scores at testing times
    return _loss(test)
end


function fit_regularization_helper!(reg_fields::Vector{Symbol}, reg_key::Symbol, mws::ModelWorkspace, training_inds::AbstractVecOrMat, testing_inds::AbstractVecOrMat, test_factor::Real, reg_min::Real, reg_max::Real; kwargs...)
    om = mws.om
    @assert 0 < reg_min < reg_max < Inf
    ℓs = Array{Float64}(undef, 2)
    reg_hold = [1, test_factor] .* getfield(om, reg_fields[1])[reg_key]
    println("initial regularization eval")
    ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds)
    println("$(test_factor)x regularization eval")
    ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds)
    println()
    # need to try decreasing regularization
    if ℓs[2] > ℓs[1]
        while (ℓs[2] > ℓs[1]) && (reg_min < reg_hold[1] < reg_max)
            println("trying a lower regularization")
            ℓs[2] = ℓs[1]
            reg_hold ./= test_factor
            ℓs[1] = eval_regularization(reg_fields, reg_key, reg_hold[1], mws, training_inds, testing_inds)
        end
        for field in reg_fields
            getfield(om, field)[reg_key] = reg_hold[2]
        end
    # need to try increasing regularization
    else
        while (ℓs[1] > ℓs[2]) && (reg_min < reg_hold[2] < reg_max)
            println("trying a higher regularization")
            ℓs[1] = ℓs[2]
            reg_hold .*= test_factor
            ℓs[2] = eval_regularization(reg_fields, reg_key, reg_hold[2], mws, training_inds, testing_inds)
        end
        for field in reg_fields
            getfield(om, field)[reg_key] = reg_hold[1]
        end
    end
end


_key_list = [:L2_μ, :L1_μ, :L1_μ₊_factor, :L2_M, :L1_M, :shared_M]
function check_for_valid_regularization(reg::Dict{Symbol, <:Real})
    for i in keys(reg)
        @assert i in _key_list "The requested regularization isn't valid"
    end
end


function fit_regularization!(mws::ModelWorkspace, training_inds::AbstractVecOrMat; key_list::Vector{Symbol}=_key_list, share_regs::Bool=true, kwargs...)
    om = mws.om
    n_obs = size(d.flux, 2)
    testing_inds = [i for i in 1:n_obs if !(i in training_inds)]
    println("starting regularization searches")
    check_for_valid_regularization(om.reg_tel)
    check_for_valid_regularization(om.reg_star)
    if share_regs; @assert keys(om.reg_tel) == keys(om.reg_star) end
    for key in key_list
        if key == :L1_μ₊_factor
            test_factor, reg_min, reg_max = 1.2, 1e-1, 1e1
        else
            test_factor, reg_min, reg_max = 10, 1e-3, 1e10
        end
        if share_regs
            if haskey(om.reg_tel, key)
                println("before: regs[:$key]  = $(om.reg_tel[key])")
                fit_regularization_helper!(_reg_fields, key, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; kwargs...)
                println("after:  regs[:$key]  = $(om.reg_tel[key])")
            end
        else
            if haskey(om.reg_tel, key)
                println("before: reg_tel[:$key]  = $(om.reg_tel[key])")
                fit_regularization_helper!([:reg_tel], key, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; kwargs...)
                println("after:  reg_tel[:$key]  = $(om.reg_tel[key])")
            end
            if haskey(om.reg_star, key)
                println("before: reg_star[:$key] = $(om.reg_star[key])")
                fit_regularization_helper!([:reg_star], key, mws, training_inds, testing_inds, test_factor, reg_min, reg_max; kwargs...)
                println("after:  reg_star[:$key] = $(om.reg_star[key])")
            end
        end
    end
end
