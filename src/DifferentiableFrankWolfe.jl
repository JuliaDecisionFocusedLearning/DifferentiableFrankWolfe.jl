"""
    DifferentiableFrankWolfe

Differentiable wrapper for [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) convex optimization routines.
"""
module DifferentiableFrankWolfe

using ChainRulesCore: ChainRulesCore, NoTangent
using FrankWolfe: FrankWolfe, LinearMinimizationOracle
using FrankWolfe: away_frank_wolfe, compute_extreme_point
using ImplicitDifferentiation: ImplicitFunction, IterativeLinearSolver
using LinearAlgebra: dot

export DiffFW

include("simplex_projection.jl")
include("difffw.jl")

end
