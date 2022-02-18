using Revise
using Pkg
Pkg.activate(".")
using Interventional
using DelimitedFiles
using BenchmarkTools, Cthulhu, LinearAlgebra

data, header = readdlm("lode.csv", ',', header=true)

dbn_data = format_DBNstyle(data,gradients=false, intercept=true, initialIntercept=true)



Z = zeros(528,3)
Z[265:330, 2] .= 1
Z[331:396, 3] .= 1
Z[397:462, 1] .= 1
Z[397:462, 2] .= 1
Z[463:528, 2] .= 1
Z[463:528, 3] .= 1


#@run 
InterventionalInference(dbn_data, Z, 3, mechanismChangeOut=true,fixedEffectOut=true)






