using TestItems

@testitem "Quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(
        DifferentiableFrankWolfe; ambiguities = false, deps_compat = (check_extras = false,)
    )
end

@testitem "Correctness (JET.jl)" begin
    using JET
    JET.test_package(DifferentiableFrankWolfe; target_defined_modules = true)
end

@testitem "Doctests (Documenter.jl)" begin
    using Documenter
    doctest(DifferentiableFrankWolfe)
end
