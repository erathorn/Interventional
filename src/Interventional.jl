module Interventional
    using LinearAlgebra
    using Combinatorics
    using LogExpFunctions
    using StatsFuns
    using Statistics
    using ProgressMeter
    
    include("types.jl")
    include("utils.jl")
    include("priors.jl")
    include("predictors.jl")
    include("datareader.jl")
    include("interventionalinference.jl")
    include("inference.jl")

    export InterventionalInference, format_DBNstyle


end # module
