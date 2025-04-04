"""
    ForwardFW{F,G,M,A}

Underlying solver for [`DiffFW`](@ref), which relies on a variant of Frank-Wolfe.
"""
struct ForwardFW{F,G,M,A}
    f::F
    f_grad1::G
    lmo::M
    alg::A
end

"""
    ConditionsFW{F,G,M}

Differentiable optimality conditions for [`DiffFW`](@ref), which rely on a custom [`simplex_projection`](@ref) implementation.
"""
struct ConditionsFW{G}
    f_grad1::G
end

"""
    DiffFW{F,G,M,A,I}

Callable parametrized wrapper for the Frank-Wolfe algorithm `θ -> argmin_{x ∈ C} f(x, θ)`, which can be differentiated implicitly wrt `θ`.

Reference: <https://arxiv.org/abs/2105.15183> (section 2 + end of appendix A).

# Fields

- `f::F`: function `f(x, θ)` to minimize wrt `x`
- `f_grad1::G`: gradient `∇ₓf(x, θ)` of `f` wrt `x`
- `lmo::M`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx` from [FrankWolfe.jl], implicitly defines the convex set `C`
- `alg::A`: optimization algorithm from [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl)
- `implicit::I`: implicit function from [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl)
"""
struct DiffFW{F,G,M<:LinearMinimizationOracle,A,I<:ImplicitFunction}
    f::F
    f_grad1::G
    lmo::M
    alg::A
    implicit::I
end

"""
    DiffFW(f, f_grad1, lmo[; alg=away_frank_wolfe, implicit_kwargs=(;)])

Constructor which chooses a default algorithm and creates the implicit function automatically.
"""
function DiffFW(f, f_grad1, lmo; alg=away_frank_wolfe, implicit_kwargs=NamedTuple())
    forward = ForwardFW(f, f_grad1, lmo, alg)
    conditions = ConditionsFW(f_grad1)
    implicit = ImplicitFunction(forward, conditions; implicit_kwargs...)
    return DiffFW(f, f_grad1, lmo, alg, implicit)
end

"""
    dfw(θ::AbstractVector; frank_wolfe_kwargs)

Apply the Frank-Wolfe algorithm to `θ` with settings defined by `frank_wolfe_kwargs`.
"""
function (dfw::DiffFW)(θ::AbstractVector{<:Real}; kwargs...)
    p, V = dfw.implicit(θ; kwargs...)
    return sum(pᵢ * Vᵢ for (pᵢ, Vᵢ) in zip(p, V))
end

function (forward::ForwardFW)(
    θ::AbstractVector{<:Real}; frank_wolfe_kwargs=NamedTuple(), kwargs...
)
    f, f_grad1, lmo, alg = forward.f, forward.f_grad1, forward.lmo, forward.alg
    obj(x) = f(x, θ)
    grad!(g, x) = g .= f_grad1(x, θ)
    x0 = compute_extreme_point(lmo, θ)
    x, v, primal, dual_gap, traj_data, active_set = alg(
        obj, grad!, lmo, x0; frank_wolfe_kwargs...
    )
    p, V = active_set.weights, active_set.atoms
    return p, V
end

function (conditions::ConditionsFW)(
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    V::AbstractVector{<:AbstractVector{<:Real}};
    kwargs...,
)
    f_grad1 = conditions.f_grad1
    x = sum(pᵢ * Vᵢ for (pᵢ, Vᵢ) in zip(p, V))
    ∇ₓf = f_grad1(x, θ)
    ∇ₚg = [dot(Vᵢ, ∇ₓf) for Vᵢ in V]
    T = simplex_projection(p .- ∇ₚg)
    return T .- p
end
