
function inference(dbn_data, P, n, Z, Sigma, R, IP)
    parents = zeros(P)
    yhat = zeros(n, P)
    parentsets = zeros(P, n_grphs)
    for (m, p_inds) in enumerate(powerset(1:P, 0, maxindegree))
        parentsets[p_inds, m] .= 1
        if IP.mechanismChangeOut
            X = predictor_mechanism_out(dbn_data, n, p_inds, Z, Sigma, R)
        end
        if IP.fixedEffectOut
            fixed_effect_out(X, IP0, Z, p_inds)
        end
        uninhibitedResponses = collect(1:P)
        inhibitedResponses = Int[]
        if !allowSelfEdges
            uninhibitedResponses = intersect(uninhibitedResponses, setdiff(1:P, p_inds))
        end
        if IP.perfectIn || IP.fixedEffectIn || IP.mechanismChangeIn
            inhibitedResponses =
                intersect(uninhibitedResponses, findall(maximum(Z, dims = 1) .== 1))
            uninhibitedResponses = setdiff(uninhibitedResponses, inhibitedResponses)
        end


        """
        get uninhibited responses here
        """
        b = size(X, 2)
        H = zeros(n, n)
        if b != 0
            H = crossfun1(X, g / (g + 1.0))
        end

        @inbounds for p in uninhibitedResponses
            yhat[:, p] .= yhat[:, p] .+ exp(lpost[p, m]) .* H * dbn_data.y_trans[:, p]
        end


    end

end