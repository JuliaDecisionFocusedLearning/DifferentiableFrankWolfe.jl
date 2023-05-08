@testitem "Quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(DifferentiableFrankWolfe; ambiguities=false)
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(DifferentiableFrankWolfe; verbose=false, overwrite=false)
end

@testitem "Correctness (JET.jl)" begin
    using JET
    using Zygote
    if VERSION >= v"1.8"
        JET.test_package(DifferentiableFrankWolfe; toplevel_logger=nothing, mode=:typo)
    end
end
