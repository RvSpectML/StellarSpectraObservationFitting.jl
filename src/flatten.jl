# Expanding ParameterHandling.jl funcs to deal with custom structs
using ParameterHandling
import ParameterHandling.flatten

# function flatten(::Type{T}, x::GenericData) where {T<:Real}
#     x_vec, unflatten = flatten(T, [getfield(x, i) for i in fieldnames(typeof(x))])
#     function unflatten_to_struct(v::Vector{T})
#         v_vec_vec = unflatten(v)
#         return GenericData(v_vec_vec...)
#     end
#     return x_vec, unflatten_to_struct
# end


# function flatten(::Type{T}, x::AbstractArray) where {T<:Real}
#     x_vec, from_vec = flatten(T, vec(x))
#     Array_from_vec(x_vec) = oftype(x, reshape(from_vec(x_vec), size(x)))
#     function Array_from_vec!(x_vec, holder::AbstractArray)
#         holder[:] = Array_from_vec(x_vec)
#     end
#     return x_vec, Array_from_vec, Array_from_vec!
# end
# function flatten(::Type{T}, x::AbstractVector) where {T<:Real}
#     x_vecs_and_backs = map(val -> flatten(T, val), x)
#     x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
#     function Vector_from_vec(x_vec)
#         sz = _cumsum(map(length, x_vecs))
#         x_Vec = [
#             backs[n](x_vec[(sz[n] - length(x_vecs[n]) + 1):sz[n]]) for n in eachindex(x)
#         ]
#         return oftype(x, x_Vec)
#     end
#     function Vector_from_vec!(x_vec, holder)
#         holder .= Vector_from_vec(x_vec)
#     end
#     return reduce(vcat, x_vecs), Vector_from_vec
# end
function flatten(::Type{T}, x::LinearModel) where {T<:Real}
    x_vec, unflatten = flatten(T, [getfield(x, i) for i in fieldnames(typeof(x))])
    function unflatten_to_struct(v::Vector{T})
        v_vec_vec = unflatten(v)
        return LinearModel(v_vec_vec...)
    end
    # function unflatten_to_struct!(v::Vector{T}, holder::LinearModel)
    #     v_vec_vec = unflatten(v)
    #     for i in fieldnames(typeof(holder))
    #         getfield(holder, i) .= v_vec_vec[i]
    #     end
    # end
    return x_vec, unflatten_to_struct#, unflatten_to_struct!
end
