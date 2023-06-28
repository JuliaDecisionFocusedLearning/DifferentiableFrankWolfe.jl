"""
    DifferentiableFrankWolfe

Differentiable wrapper for [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) convex optimization routines.
"""
module DifferentiableFrankWolfe

using ChainRulesCore: ChainRulesCore, NoTangent
using FrankWolfe:
    FrankWolfe, LinearMinimizationOracle, away_frank_wolfe, compute_extreme_point
using ImplicitDifferentiation: ImplicitFunction
using LinearAlgebra: dot

export DiffFW
export LinearMinimizationOracleWithKwargs, LinearMaximizationOracleWithKwargs

include("simplex_projection.jl")
include("lmo_kwargs.jl")
include("difffw.jl")

end
