"""
    simplex_projection(z)

Compute the Euclidean projection of the vector `z` onto the probability simplex.

This function is differentiable thanks to a custom chain rule.

Reference: <https://arxiv.org/abs/1602.02068>.
"""
function simplex_projection(z::AbstractVector{<:Real}; kwargs...)
    p, _ = simplex_projection_and_support(z)
    return p
end

"""
    simplex_projection_and_support(z)

Compute the Euclidean projection `p` of `z` on the probability simplex as well as the indicators `s` of its support, which are useful for differentiation.

Reference: <https://arxiv.org/abs/1602.02068>.
"""
function simplex_projection_and_support(z::AbstractVector{<:Real})
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    k = maximum(j for j in 1:d if (1 + j * z_sorted[j]) > z_sorted_cumsum[j])
    τ = (z_sorted_cumsum[k] - 1) / k
    p = z .- τ
    p .= max.(p, zero(eltype(p)))
    s = [Int(p[i] > eps()) for i in 1:d]
    return p, s
end

function ChainRulesCore.rrule(::typeof(simplex_projection), z::AbstractVector{<:Real})
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function simplex_projection_pullback(dp)
        vjp = s .* (dp .- dot(dp, s) / S)
        return (NoTangent(), vjp)
    end
    return p, simplex_projection_pullback
end
