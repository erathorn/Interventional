module Interventional
    using LinearAlgebra
    using Combinatorics
    using LogExpFunctions
    using StatsFuns
    using Statistics
    
    include("types.jl")
    include("utils.jl")
    include("priors.jl")
    include("predictors.jl")
    include("datareader.jl")
    include("interventionalinference.jl")

    export InterventionalInference, format_DBNstyle


end # module
