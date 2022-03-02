
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
    lpost::Matrix{R},
)::Matrix{R} where {R<:Real}
    P = length(parentvec)
    yhat = zeros(n, P)
    parentsets = zeros(P, n_grphs)
    prog_bar = Progress(
        n_grphs,
        dt = 0.001,
        desc = "Processing $n_grphs models: ",
        showspeed = true,
    )

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


        @inbounds for p in uninhibitedResponses   
            H = zeros(n, n)
            if length(p_inds) != 0
                H = crossfun1(X1, g / (g + 1.0))
            end
            yhat[:, p] .+= exp(lpost[p, m]) .* H * parentvec[p].y
        end

        @inbounds for p in inhibitedResponses
            # Start assembling predictor indices
            H, _ = H_func(dbn_data, parentvec, Z, X1, X0, IP, g, p, p_inds, n, covariance)
            yhat[parentvec[p].obs, p] .+=
                exp(lpost[p, m]) .* H * parentvec[p].y[parentvec[p].obs]
        end
        next!(prog_bar)
    end
    yhat
end
