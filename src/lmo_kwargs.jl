"""
    LinearMinimizationOracleWithKwargs{F,K}

Wraps a linear minimizer as a `FrankWolfe.LinearMinimizationOracle` with predefined keyword arguments.

# Fields
- `minimizer::F`: black box linear minimizer
- `minimizer_kwargs::K`: keyword arguments passed to the minimizer whenever it is called
"""
struct LinearMinimizationOracleWithKwargs{F,K} <: LinearMinimizationOracle
    minimizer::F
    minimizer_kwargs::K
end

"""
    LinearMaximizationOracleWithKwargs{F,K}

Wraps a linear maximizer as a `FrankWolfe.LinearMinimizationOracle` with a sign switch and predefined keyword arguments.

# Fields
- `maximizer::F`: black box linear maximizer
- `maximizer_kwargs::K`: keyword arguments passed to the maximizer whenever it is called
"""
struct LinearMaximizationOracleWithKwargs{F,K} <: LinearMinimizationOracle
    maximizer::F
    maximizer_kwargs::K
end

function LinearMinimizationOracleWithKwargs(maximizer)
    return LinearMinimizationOracleWithKwargs(maximizer, NamedTuple())
end

function LinearMaximizationOracleWithKwargs(maximizer)
    return LinearMaximizationOracleWithKwargs(maximizer, NamedTuple())
end

"""
    FrankWolfe.compute_extreme_point(lmokw::LinearMinimizationOracleWithKwargs, direction)
"""
function FrankWolfe.compute_extreme_point(
    lmokw::LinearMinimizationOracleWithKwargs, direction; kwargs...
)
    (; minimizer, minimizer_kwargs) = lmokw
    v = minimizer(direction; minimizer_kwargs...)
    return v
end

"""
    FrankWolfe.compute_extreme_point(lmokw::LinearMaximizationOracleWithKwargs, direction)
"""
function FrankWolfe.compute_extreme_point(
    lmokw::LinearMaximizationOracleWithKwargs, direction; kwargs...
)
    (; maximizer, maximizer_kwargs) = lmokw
    opposite_direction = -direction
    v = maximizer(opposite_direction; maximizer_kwargs...)
    return v
end
