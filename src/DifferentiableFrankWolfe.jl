"""
    DifferentiableFrankWolfe

Differentiable wrapper for [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) convex optimization routines.
"""
module DifferentiableFrankWolfe

using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, unthunk
using FrankWolfe: FrankWolfe, LinearMinimizationOracle
using FrankWolfe:
    away_frank_wolfe,
    blended_conditional_gradient,
    blended_pairwise_conditional_gradient,
    compute_extreme_point,
    pairwise_frank_wolfe
using ImplicitDifferentiation: ImplicitFunction
using LinearAlgebra: dot

export DiffFW

include("simplex_projection.jl")
include("difffw.jl")

end
