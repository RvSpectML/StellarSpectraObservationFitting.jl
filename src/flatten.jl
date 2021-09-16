# Expanding ParameterHandling.jl funcs to deal with custom structs
using ParameterHandling
import ParameterHandling.flatten

function flatten(::Type{T}, x::SSOF.GenericData) where {T<:Real}
    x_vec, unflatten = flatten(T, [getfield(x, i) for i in fieldnames(typeof(x))])
    function unflatten_to_struct(v::Vector{T})
        v_vec_vec = unflatten(v)
        return SSOF.GenericData(v_vec_vec...)
    end
    return x_vec, unflatten_to_struct
end
function flatten(::Type{T}, x::SSOF.LinearModel) where {T<:Real}
    x_vec, unflatten = flatten(T, [getfield(x, i) for i in fieldnames(typeof(x))])
    function unflatten_to_struct(v::Vector{T})
        v_vec_vec = unflatten(v)
        return SSOF.LinearModel(v_vec_vec...)
    end
    return x_vec, unflatten_to_struct
end
