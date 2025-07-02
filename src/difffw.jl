"""
    ForwardFW

Underlying solver for [`DiffFW`](@ref), which relies on a variant of Frank-Wolfe with active set memorization.
"""
struct ForwardFW{F,G,M,A}
    f::F
    f_grad1::G
    lmo::M
    alg::A

    function ForwardFW(f, f_grad1, lmo, alg)
        @assert alg in (
            away_frank_wolfe,
            blended_conditional_gradient,
            blended_pairwise_conditional_gradient,
            pairwise_frank_wolfe,
        )
        return new{typeof(f),typeof(f_grad1),typeof(lmo),typeof(alg)}(f, f_grad1, lmo, alg)
    end
end

function (forward::ForwardFW)(θ::AbstractArray, x0::AbstractArray, frank_wolfe_kwargs)
    f, f_grad1, lmo, alg = forward.f, forward.f_grad1, forward.lmo, forward.alg
    obj(x) = f(x, θ)
    grad!(g, x) = copyto!(g, f_grad1(x, θ))
    x0_copy = copy(x0)
    x_final, v_final, primal_value, dual_gap, traj_data, active_set = alg(
        obj, grad!, lmo, x0_copy; frank_wolfe_kwargs...
    )
    stats = (; x_final, v_final, primal_value, dual_gap, traj_data, active_set)
    p = active_set.weights
    return p, stats
end

"""
    ConditionsFW

Differentiable optimality conditions for [`DiffFW`](@ref), which rely on a custom [`simplex_projection`](@ref) implementation.
"""
struct ConditionsFW{G}
    f_grad1::G
end

function (conditions::ConditionsFW)(
    θ::AbstractArray,
    p::AbstractVector,
    stats::NamedTuple,
    _x0::AbstractArray,
    _frank_wolfe_kwargs,
)
    V = stats.active_set.atoms
    f_grad1 = conditions.f_grad1
    V_mat = stack(V)
    x = V_mat * p
    ∇ₓf = f_grad1(x, θ)
    ∇ₚg = transpose(V_mat) * ∇ₓf
    T = simplex_projection(p .- ∇ₚg)
    return T .- p
end

"""
    DiffFW

Callable parametrized wrapper for the Frank-Wolfe algorithm to solve `θ -> argmin_{x ∈ C} f(x, θ)` from a given starting point `x0`.
The solution routine can be differentiated implicitly with respect `θ`, but not with respect to `x0`.

# Constructor

    DiffFW(f, f_grad1, lmo, alg=away_frank_wolfe; implicit_kwargs...)

- `f`: function `f(x, θ)` to minimize with respect to `x`
- `f_grad1`: gradient `∇ₓf(x, θ)` of `f` with respect to `x`
- `lmo`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx` from [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl), implicitly defines the convex set `C`
- `alg`: optimization algorithm from [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl), must return an `active_set`
- `implicit_kwargs`: keyword arguments passed to the `ImplicitFunction` object from [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl)

# References

> [Efficient and Modular Implicit Differentiation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html), Blondel et al. (2022)
"""
struct DiffFW{F,G,M<:LinearMinimizationOracle,A,I<:ImplicitFunction}
    f::F
    f_grad1::G
    lmo::M
    alg::A
    implicit::I
end

function DiffFW(
    f::F, f_grad1::G, lmo::L, alg::A=away_frank_wolfe; implicit_kwargs...
) where {F,G,L,A}
    forward = ForwardFW(f, f_grad1, lmo, alg)
    conditions = ConditionsFW(f_grad1)
    implicit = ImplicitFunction(forward, conditions; implicit_kwargs...)
    return DiffFW(f, f_grad1, lmo, alg, implicit)
end

"""
    detailed_output(dfw::DiffFW, θ::AbstractArray, x0::AbstractArray; kwargs...)

Apply the differentiable Frank-Wolfe algorithm defined by `dfw` to parameter `θ` with starting point `x0`.
Keyword arguments are passed on to the Frank-Wolfe algorithm inside `dfw`.

Return a couple (x, stats) where `x` is the solution and `stats` is a named tuple containing additional information (its contents are not covered by public API, and mostly useful for debugging).
"""
function detailed_output(dfw::DiffFW, θ::AbstractArray, x0::AbstractArray; kwargs...)
    p, stats = dfw.implicit(θ, x0, kwargs)
    V = stats.active_set.atoms
    V_mat = stack(V)
    x = V_mat * p
    return x, stats
end

"""
    (dfw::DiffFW)(θ::AbstractArray, x0::AbstractArray; kwargs...)

Apply the differentiable Frank-Wolfe algorithm defined by `dfw` to parameter `θ` with starting point `x0`.
Keyword arguments are passed on to the Frank-Wolfe algorithm inside `dfw`.

Return the optimal solution `x`.
"""
function (dfw::DiffFW)(θ::AbstractArray, x0::AbstractArray; kwargs...)
    x, _ = detailed_output(dfw, θ, x0; kwargs...)
    return x
end
