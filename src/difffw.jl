"""
    ForwardFW{F,G,M}

Underlying solver for [`DiffFW`](@ref), which relies on `FrankWolfe.away_frank_wolfe`.
"""
struct ForwardFW{F,G,M}
    f::F
    f_grad1::G
    lmo::M
end

"""
    ConditionsFW{F,G,M}

Differentiable optimality conditions for [`DiffFW`](@ref), which rely on a custom [`sparse_argmax`](@ref) implementation.
"""
struct ConditionsFW{G}
    f_grad1::G
end

"""
    DiffFW{F,G,M,I}

Callable parametrized wrapper for the Frank-Wolfe algorithm `θ -> argmin_{x ∈ C} f(x, θ)`, which can be differentiated implicitly wrt `θ`.

Reference: <https://arxiv.org/abs/2105.15183> (especially section 2 and the end of appendix A).

# Fields
- `f::F`: function `f(x, θ)` to minimize wrt `x`
- `f_grad1::G`: gradient `∇ₓf(x, θ)` of `f` wrt `x`
- `lmo::M`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx`, implicitly defines the convex set `C`
- `implicit::I`: implicit function constructed from the previous fields
"""
struct DiffFW{F,G,M<:LinearMinimizationOracle,I<:ImplicitFunction}
    f::F
    f_grad1::G
    lmo::M
    implicit::I
end

function DiffFW(f, f_grad1, lmo)
    forward = ForwardFW(f, f_grad1, lmo)
    conditions = ConditionsFW(f_grad1)
    implicit = ImplicitFunction(forward, conditions)
    return DiffFW(f, f_grad1, lmo, implicit)
end

"""
    dfw(θ; frank_wolfe_kwargs)

Apply the Frank-Wolfe algorithm to `θ` with settings defined by `frank_wolfe_kwargs`.
"""
function (dfw::DiffFW)(θ::AbstractArray{<:Real}; kwargs...)
    p, V = dfw.implicit(θ; kwargs...)
    return sum(pᵢ * Vᵢ for (pᵢ, Vᵢ) in zip(p, V))
end

function (forward::ForwardFW)(θ::AbstractArray{<:Real}; frank_wolfe_kwargs=(;), kwargs...)
    f, f_grad1, lmo = forward.f, forward.f_grad1, forward.lmo
    obj(x) = f(x, θ)
    grad!(g, x) = g .= f_grad1(x, θ)
    x0 = compute_extreme_point(lmo, θ)
    x, v, primal, dual_gap, traj_data, active_set = away_frank_wolfe(
        obj, grad!, lmo, x0; frank_wolfe_kwargs...
    )
    p, V = active_set.weights, active_set.atoms
    return p, V
end

function (conditions::ConditionsFW)(
    θ::AbstractArray{<:Real},
    p::AbstractVector{<:Real},
    V::AbstractVector{<:AbstractArray{<:Real}};
    kwargs...,
)
    f_grad1 = conditions.f_grad1
    x = sum(pᵢ * Vᵢ for (pᵢ, Vᵢ) in zip(p, V))
    ∇ₓf = f_grad1(x, θ)
    ∇ₚg = [dot(Vᵢ, ∇ₓf) for Vᵢ in V]
    T = simplex_projection(p .- ∇ₚg)
    return T .- p
end
