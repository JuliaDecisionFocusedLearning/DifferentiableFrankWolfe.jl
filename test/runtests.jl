using Aqua
using DifferentiableFrankWolfe
using Documenter
using JET
using JuliaFormatter
using Test
using Zygote

@testset verbose = true "DifferentiableFrankWolfe.jl" begin
    @testset "Quality (Aqua.jl)" begin
        Aqua.test_all(DifferentiableFrankWolfe; ambiguities=false)
    end

    @testset "Formatting (JuliaFormatter.jl)" begin
        @test format(DifferentiableFrankWolfe; verbose=false, overwrite=false)
    end

    @testset "Correctness (JET.jl)" begin
        if VERSION >= v"1.9"
            JET.test_package(DifferentiableFrankWolfe; target_defined_modules=true)
        end
    end

    @testset "Doctests (Documenter.jl)" begin
        doctest(DifferentiableFrankWolfe)
    end

    @testset "Tutorial" begin
        include(joinpath(@__DIR__, "..", "examples", "tutorial.jl"))
    end
end
