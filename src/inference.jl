
function fittedValues(
    dbn_data::DBN_Data{R},
    parentvec::Vector{Parent_struct{R}},
    n_grphs::Int,
    maxindegree::Int,
    n::Int,
    Z::Matrix{<:Real},
    g::Int,
    IP::InterventionPattern{Bool},
    covariance::Bool,
    lpost::Matrix{R}
)::Matrix{R} where {R<:Real}
    P = length(parentvec)
    yhat = zeros(n, P)
    parentsets = zeros(P, n_grphs)
    prog_bar = Progress(n_grphs, dt=0.001, desc="Processing $n_grphs models: ", showspeed=true)
    
    @inbounds for (m, p_inds) in enumerate(powerset(1:P, 0, maxindegree))
        parentsets[p_inds, m] .= 1
        
        uninhibitedResponses = collect(1:P)
        inhibitedResponses = Int[]
        if !IP.allowSelfEdges
            uninhibitedResponses = intersect(uninhibitedResponses, setdiff(1:P, p_inds))
        end
        if IP.perfectIn || IP.fixedEffectIn || IP.mechanismChangeIn
            inhibitedResponses =
                intersect(uninhibitedResponses, findall(maximum(Z, dims = 1) .== 1))
            uninhibitedResponses = setdiff(uninhibitedResponses, inhibitedResponses)
        end

        X0 = hcat([parentvec[par].X0 for par in p_inds]...)
        X1 = hcat([parentvec[par].X1 for par in p_inds]...)
        """
        get uninhibited responses here
        """
        b = size(X1, 2)
        H = zeros(n, n)
        if length(p_inds) != 0
            H = crossfun1(X1, g / (g + 1.0))
        end

        @inbounds for p in uninhibitedResponses
            yhat[:, p] .+= exp(lpost[p, m]) .* H * parentvec[p].y
        end
        
        @inbounds for p in inhibitedResponses
             # Start assembling predictor indices

             X1 = predictors_mechanismchangein(Val(IP.mechanismChangeIn), parentvec[p],X1, dbn_data, Z, p)                    #
             X1 = perfectout(Val(IP.perfectOut), X1, dbn_data, Z, p_inds, dbn_data.Sigma, dbn_data.R)

             if IP.fixedEffectIn || IP.fixedEffectOut
                 to_use = Int[]
                 if IP.fixedEffectOut
                     to_use = union(to_use, [x for x in p_inds if maximum(Z[pbs, x]) == 1])
                 end
                 if IP.fixedEffectIn
                     to_use = union(to_use, p)
                 end
                 Z_new = zeros(size(Z[:, to_use]))
                 Z_new[:] .= Z[parentvec[p].obs , to_use]
                 X1 = hcat(X1, Z_new)
             end
             # end assembling predictors
             
             
             X1 = Xfun(Val(covariance), X1, X0, dbn_data, parentvec[p].obs)
             
             H = I(n)
             b = size(X1, 2)
             if b != 0
                 H = crossfun1(X1, g / (g + 1.0))
             end

            yhat[parentvec[p].obs, p] .+= exp(lpost[p, m]) .* H * parentvec[p].y[parentvec[p].obs]
        end
    next!(prog_bar)
    end
    yhat
end