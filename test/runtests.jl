using Aqua
using DifferentiableFrankWolfe
using Documenter
using JET
using JuliaFormatter
using Test

@testset verbose = true "DifferentiableFrankWolfe.jl" begin
    @testset verbose = false "Code quality (Aqua.jl)" begin
        Aqua.test_all(DifferentiableFrankWolfe; ambiguities=false)
    end
    @testset verbose = true "Formatting (JuliaFormatter.jl)" begin
        @test format(DifferentiableFrankWolfe; verbose=true, overwrite=false)
    end
    @testset verbose = true "Static checking (JET.jl)" begin
        if VERSION >= v"1.8"
            JET.test_package(DifferentiableFrankWolfe; toplevel_logger=nothing)
        end
    end
    doctest(DifferentiableFrankWolfe)
    @testset verbose = true "DiffFW" begin
        include("difffw.jl")
    end
end
