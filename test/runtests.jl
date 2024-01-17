using Aqua
using DifferentiableFrankWolfe
using Documenter
using ImplicitDifferentiation
using JET
using JuliaFormatter
using Test
using Zygote

@testset verbose = true "DifferentiableFrankWolfe.jl" begin
    @testset "Quality (Aqua.jl)" begin
        Aqua.test_all(
            DifferentiableFrankWolfe; ambiguities=false, deps_compat=(check_extras=false,)
        )
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

    @testset "Constructor" begin
        dfw1 = DiffFW(f, f_grad1, lmo)
        @test !dfw1.implicit.linear_solver.accept_inconsistent

        implicit_kwargs = (;
            linear_solver=IterativeLinearSolver(; accept_inconsistent=true)
        )
        dfw2 = DiffFW(f, f_grad1, lmo; implicit_kwargs)
        @test dfw2.implicit.linear_solver.accept_inconsistent
    end
end
