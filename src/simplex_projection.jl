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

relu(x) = max(x, zero(typeof(x)))

"""
    simplex_projection_and_support(z)

Compute the Euclidean projection `p` of `z` on the probability simplex as well as the indicators `s` of its support, which are useful for differentiation.

Reference: <https://arxiv.org/abs/1602.02068>.
"""
function simplex_projection_and_support(z::AbstractVector{T}) where {T<:Real}
    d = length(z)
    z_sorted = sort(z; rev=true)
    z_sorted_cumsum = cumsum(z_sorted)
    ind_filter = 1 .+ (1:d) .* z_sorted .> z_sorted_cumsum
    k = findlast(ind_filter)
    τ = (z_sorted_cumsum[k] - 1) / k
    p = relu.(z .- τ)
    s = p .> eps(T)
    return p, s
end

function ChainRulesCore.rrule(::typeof(simplex_projection), z::AbstractVector{<:Real})
    proj = ProjectTo(z)
    p, s = simplex_projection_and_support(z)
    S = sum(s)
    function simplex_projection_pullback(dp_thunked)
        dp = unthunk(dp_thunked)
        vjp = s .* (dp .- dot(dp, s) / S)
        return (NoTangent(), proj(vjp))
    end
    return p, simplex_projection_pullback
end
