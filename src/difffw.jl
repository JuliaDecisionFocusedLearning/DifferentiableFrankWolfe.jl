"""
    ForwardFW

Underlying solver for [`DiffFW`](@ref), which relies on a variant of Frank-Wolfe.
"""
struct ForwardFW{F,G,M,A}
    f::F
    f_grad1::G
    lmo::M
    alg::A
end

"""
    ConditionsFW

Differentiable optimality conditions for [`DiffFW`](@ref), which rely on a custom [`simplex_projection`](@ref) implementation.
"""
struct ConditionsFW{G}
    f_grad1::G
end

"""
    DiffFW

Callable parametrized wrapper for the Frank-Wolfe algorithm to solve `θ -> argmin_{x ∈ C} f(x, θ)`, which can be differentiated implicitly wrt `θ`.

Reference: <https://arxiv.org/abs/2105.15183> (section 2 + end of appendix A).

# Fields

- `f`: function `f(x, θ)` to minimize wrt `x`
- `f_grad1`: gradient `∇ₓf(x, θ)` of `f` wrt `x`
- `lmo`: linear minimization oracle `θ -> argmin_{x ∈ C} θᵀx` from [FrankWolfe.jl], implicitly defines the convex set `C`
- `alg`: optimization algorithm from [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl)
- `implicit`: implicit function from [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl)
"""
struct DiffFW{F,G,M<:LinearMinimizationOracle,A,I<:ImplicitFunction}
    f::F
    f_grad1::G
    lmo::M
    alg::A
    implicit::I
end

"""
    DiffFW(f, f_grad1, lmo, alg=away_frank_wolfe; implicit_kwargs=(;))

Constructor for [`DiffFW`](@ref) which chooses a default algorithm and creates the implicit function automatically.
"""
function DiffFW(
    f::F, f_grad1::G, lmo::L, alg::A=away_frank_wolfe; implicit_kwargs=NamedTuple()
) where {F,G,L,A}
    forward = ForwardFW(f, f_grad1, lmo, alg)
    conditions = ConditionsFW(f_grad1)
    implicit = ImplicitFunction(forward, conditions; implicit_kwargs...)
    return DiffFW(f, f_grad1, lmo, alg, implicit)
end

"""
    (dfw::DiffFW)(θ::AbstractArray, frank_wolfe_kwargs::NamedTuple)

Apply the Frank-Wolfe algorithm to `θ` with settings defined by the named tuple `frank_wolfe_kwargs` (given as a positional argument).

Return a couple (x, stats) where `x` is the solution and `stats` is a named tuple containing additional information (its contents are not covered by public API, and mostly useful for debugging).
"""
function (dfw::DiffFW)(θ::AbstractArray, frank_wolfe_kwargs=NamedTuple())
    p, stats = dfw.implicit(θ, frank_wolfe_kwargs)
    V = stats.active_set.atoms
    x = mapreduce(*,+,p,V)
    return x, stats
end

function (forward::ForwardFW)(θ::AbstractArray, frank_wolfe_kwargs::NamedTuple)
    f, f_grad1, lmo, alg = forward.f, forward.f_grad1, forward.lmo, forward.alg
    obj(x) = f(x, θ)
    grad!(g, x) = copyto!(g, f_grad1(x, θ))
    x0 = compute_extreme_point(lmo, θ)
    x_final, v_final, primal_value, dual_gap, traj_data, active_set = alg(
        obj, grad!, lmo, x0; frank_wolfe_kwargs...
    )
    stats = (; x_final, v_final, primal_value, dual_gap, traj_data, active_set)
    p = active_set.weights
    return p, stats
end

function (conditions::ConditionsFW)(
    θ::AbstractArray, p::AbstractVector, stats::NamedTuple, frank_wolfe_kwargs::NamedTuple
)
    V = stats.active_set.atoms
    x = mapreduce(*,+,p,V)
    f_grad1 = conditions.f_grad1
    ∇ₓf = f_grad1(x, θ)
    ∇ₚg = dot.(V, Ref(∇ₓf))
    T = simplex_projection(p .- ∇ₚg)
    return T .- p
end
