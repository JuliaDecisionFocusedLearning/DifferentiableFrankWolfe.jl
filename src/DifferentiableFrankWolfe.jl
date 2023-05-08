module DifferentiableFrankWolfe

using ChainRulesCore: ChainRulesCore, NoTangent
using FrankWolfe: LinearMinimizationOracle, away_frank_wolfe, compute_extreme_point
using ImplicitDifferentiation: ImplicitFunction
using LinearAlgebra: dot

export DiffFW

include("simplex_projection.jl")
include("difffw.jl")

end
