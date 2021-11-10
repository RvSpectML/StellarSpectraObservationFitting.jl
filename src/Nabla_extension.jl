# Adding a function to properly deal with all view inputs
using Nabla

function Nabla.zerod_container(x::Vector{<:SubArray})
    y = [copy(θ) for θ in x]
    for n in eachindex(y)
        @inbounds y[n] = Nabla.zerod_container(y[n])
    end
    return y
end
