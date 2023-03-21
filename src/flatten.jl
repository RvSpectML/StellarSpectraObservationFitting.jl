# Expanding ParameterHandling.jl funcs to deal with custom structs and cases
using ParameterHandling
import ParameterHandling.flatten


"""
    flatten([eltype=Real], x::LinearModel)

Returns a "flattened" representation of `x::LinearModel` as a vector of vectors and a function
`unflatten` that takes a vector of reals of the same length and returns a LinearModel object
"""
function flatten(::Type{T}, x::LinearModel) where {T<:Real}
    x_vec, unflatten = flatten(T, [getfield(x, i) for i in fieldnames(typeof(x))])
    function unflatten_to_struct(v::Vector{T})
        v_vec_vec = unflatten(v)
        return LinearModel(v_vec_vec...)
    end
    return x_vec, unflatten_to_struct
end

"""
    flatten([eltype=Real], x::SubArray)
    
Returns a "flattened" representation of `x::SubArray` as a vector and a function
`unflatten` that takes a vector of reals of the same length and returns an Array object
"""
function flatten(::Type{T}, x::SubArray) where {T<:Real}
    x_vec, from_vec = flatten(T, vec(x))
    Array_from_vec(x_vec) = reshape(from_vec(x_vec), size(x))
    return x_vec, Array_from_vec
end
flatten(::Type{T}, x::Base.ReshapedArray) where {T<:Real} = custom_vector_flatten(T, x)
flatten(::Type{T}, x::Vector{<:SubArray}) where {T<:Real} = custom_vector_flatten(T, x)
function custom_vector_flatten(::Type{T}, x::AbstractVector) where {T<:Real}
    x_vecs_and_backs = map(val -> flatten(T, val), x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = ParameterHandling._cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[(sz[n] - length(x_vecs[n]) + 1):sz[n]]) for n in eachindex(x)]
        return x_Vec
    end
    return reduce(vcat, x_vecs), Vector_from_vec
end
