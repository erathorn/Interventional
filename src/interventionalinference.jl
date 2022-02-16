
function InterventionalInference(
    y::Matrix{T1},
    X0::Matrix{T1},
    X1::Matrix{T1},
    Z::Array{<:Real},
    maxindegree::Int;
    Sigma::Union{Array{<:Real},Missing} = missing,
    perfect_out::Bool = false,
    priorStrength::Int = 3,
    allowSelfEdges::Bool = false,
    perfectIn::Bool = false,
    fixedEffectIn::Bool = false,
    fixedEffectOut::Bool = false,
    mechanismChangeIn::Bool = false,
    mechanismChangeOut::Bool = false,
    priorGraph::Union{Missing,Matrix{<:Real}} = missing,
    g1::Union{Int,Missing} = missing,
    priorType::AbstractString = "uninformed",
)::Tuple where {T1<:Real}
    
    IP = InterventionPattern(;
        allowSelfEdges = allowSelfEdges,
        perfectOut = perfect_out,
        perfectIn = perfectIn,
        fixedEffectOut = fixedEffectOut,
        fixedEffectIn = fixedEffectIn,
        mechanismChangeOut = mechanismChangeOut,
        mechanismChangeIn = mechanismChangeIn,
    )

    X1_fun = deepcopy(X1) # Do deepcopy to not alter input    
    P = size(y, 2)
    n = size(y, 1)
    g = ismissing(g1) ? n : g1
    a = size(X0, 2)

    if !ismissing(Sigma)
        @assert size(Sigma) == (n, n) "Sigma must have dimension (n x n)"
    end

    @assert priorType in ["uninformed", "Hamming", "Mukherjee"] "$priorType is not allowed as priorType. Must be 'uninformed', 'Hamming' or 'Mukherjee'."


    if priorType in ["Hamming", "Mukherjee"]
        if !ismissing(priorGraph)
            @assert size(priorGraph) == (P, P) "priorGraph must be of dimension (P x P) "
        end
    end

    # 1: remove X0 from y
    Y_fun, IP0, R = disentangle(y, X0, 1:n, Sigma)


    # 2: ONLY DO FOR PERFECT OUT
    if IP.perfectOut
        X1_fun[findall(Z .== 1)] .= NaN
    end

    # 3: Orhtogonalize Predictors
    @views for p = 1:P
        wh = findall(broadcast(!, isnan.(X1_fun[:, p])))
        if length(wh) == n
            X1_fun[:, p] .= IP0 * X1_fun[:, p]
        else
            X1_fun[wh, p] .= crossfun1(X0[wh, :]) * X1_fun[wh, p]
        end
        X1_fun[findall(isnan.(X1_fun[:, p])), p] .= 0
    end




    # 5: prior
    n_grphs = count_graphs(P, maxindegree)

    prior = zeros(P, n_grphs) # Base Case
    if priorType == "Hamming"
        HammingPrior!(prior, priorGraph, P, maxindegree)
    elseif priorType == "Mukherjee"
        MukherjeePrior!(prior, priorGraph, P, maxindegree)
    end

    # 6: Initilisation
    ll, parentsets = main_loop(
        P,
        n_grphs,
        maxindegree,
        X1_fun,
        n,
        Z,
        X0,
        Sigma,
        R,
        IP0,
        g,
        a,
        Y_fun,
        IP
    )

    # 8: renornmalization & MAP    
    MAP, MAPmodel, MAPprob, lpost =
        MAP_function(P, n_grphs, parentsets, priorStrength, prior, ll, maxindegree)

    # 9: Model averaging
    pep = posterior(lpost, parentsets, P)


    #10: Inference

    #fitted = inference()



    pep, MAP, MAPmodel, MAPprob
end

function inference(P, n, Z, X0, X1, Sigma, R, IP)
    parents = zeros(P)
    yhat = zeros(n, P)
    parentsets = zeros(P, n_grphs)
    for (m, p_inds) in enumerate(powerset(1:P, 0, maxindegree))
        parentsets[p_inds, m] .= 1
        if IP.mechanismChangeOut
            X = predictor_mechanism_out(n, p_inds, Z, X0, X1_fun, Sigma, R)
        end
        if IP.fixedEffectOut
            fixed_effect_out!(X, IP0, Z, p_inds)
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
            yhat[:, p] .= yhat[:, p] .+ exp(lpost[p, m]) .* H * y[:, p]
        end


    end

end

function main_loop(
    P::Int,
    n_grphs::Int,
    maxindegree::Int,
    X1_fun::Matrix{T},
    n::Int,
    Z::Matrix,
    X0::Matrix{T},
    Sigma,
    R,
    IP0::Matrix{T},
    g::Int,
    a::Int,
    Y_fun::Matrix{T},
    IP::InterventionPattern{Bool}
) where T<:Real
    ll = zeros(P, n_grphs)
    parentsets = zeros(P, n_grphs)
    # 7: Main Loop
    @views @inbounds for (m, p_inds) in enumerate(powerset(1:P, 0, maxindegree))

        parentsets[p_inds, m] .= 1

        X = X1_fun[:, p_inds] # default

        if IP.mechanismChangeOut
            X = predictor_mechanism_out(n, p_inds, Z, X0, X1_fun, Sigma, R)
        end

        if IP.fixedEffectOut
            fixed_effect_out!(X, IP0, Z, p_inds)
        end

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


        """
        get uninhibited responses here
        """
        b = size(X, 2)
        H = I(n)
        if b != 0
            H = crossfun1(X, g / (g + 1.0))
        end

        @inbounds for p in uninhibitedResponses
            ll[p, m] =
                -b / 2 * log(1 + g) - (n - a) / 2 * log(dot(Y_fun[:, p], H, Y_fun[:, p]))
        end

        for p in inhibitedResponses

            # Start assembling predictor indices
            obs = collect(1:n)
            if perfectIn
                obs = findall(Z[:, p] .== 0)
            end
            # End assembling predictor indices

            # Start assembling predictors
            X = X1_fun[obs, p_inds] # default
            if IP.mechanismChangeIn
                X = predictors_mechanismchangein(X1_fun, X0, Z, p_inds, p, Sigma, R)
            elseif IP.perfect_out
                perfectout!(X, X1_fun, X0, Z, p_inds, Sigma, R)
            end

            if IP.fixedEffectIn || IP.fixedEffectOut
                to_use = Int[]
                if IP.fixedEffectOut
                    to_use = union(to_use, [x for x in p_inds if maximum(Z[pbs, x]) == 1])
                end
                if IP.fixedEffectIn
                    to_use = union(to_use, p)
                end
                X = hcat(X, Z[obs, to_use])
            end
            # end assembling predictors
            X0p = collect(X0[obs, :])
            if ismissing(Sigma)
                X = crossfun1(X0p) * X
            else
                X = sigma_mult(R, Sigma, X0p, obs) * X
            end
            H = I(n)
            b = size(X, 2)
            if b != 0
                H = crossfun1(X, g / (g + 1.0))
            end
            ll[p, m] =
                -b / 2 * log(1 + g) -
                (length(obs) - a) / 2 * log(dot(Y_fun[obs, p], H, Y_fun[obs, p]))
        end # inhibited
    end # graphs
    ll, parentsets
end



function posterior(
    lpost::Matrix{R},
    parentsets::Matrix{R},
    P::Int,
)::Matrix{R} where {R<:Real}
    pep = Matrix{R}(undef, P, P)
    @views @inbounds for i = 1:P, j = 1:P
        pep[i, j] = sum(exp.(lpost[j, findall(parentsets[i, :] .== 1)]))
    end
    pep
end

function MAP_function(
    P::Int,
    n_grphs::Int,
    parentsets::Matrix{R},
    priorStrength::Int,
    prior::Matrix{R},
    ll::Matrix{R},
    maxindegree::Int,
)::Tuple{Matrix{R},Vector{R},Vector{R},Matrix{R}} where {R<:Real}
    normalisedPrior = Matrix{R}(undef, P, n_grphs)
    lpost = Matrix{R}(undef, P, n_grphs)
    parentCount = vec(sum(parentsets, dims = 1))
    marginallikelihood = zeros(P, length(priorStrength))
    if length(priorStrength) > 1 && maximum(prior) > 0
        llinf = ll
        llinf[ll.==0] .= -Inf
        for i = 1:length(priorStrength)
            set_normalised_prior!(normalisedPrior, parentCount, maxindegree)
            normalisedPrior -= priorStrength[i] .* prior
            normalisedPrior .-= logsumexp(normalisedPrior, dims = 2)
            marginallikelihood[:, i] = minimum(ll + normalisedPrior, dims = 2)
            marginallikelihood[:, i] +=
                logsumexp(llinf + normalisedPrior - marginallikelihood[:, i], dims = 2)
        end
        logmlprod = sum(marginallikelihood, dims = 1)
        epriorstrength = priorStrength[argmax(logmlprod)]
        const1 = marginallikelihood[:, argmax(logmlprod)]

        set_normalised_prior!(normalisedPrior, parentCount, maxindegree)

        normalisedPrior -= epriorstrength .* prior
        normalisedPrior .-= logsumexp(normalisedPrior, dims = 2)
        lpost = llinf + normalisedPrior - const1

    else
        set_normalised_prior!(normalisedPrior, parentCount, maxindegree)
        normalisedPrior -= priorStrength * prior
        normalisedPrior .-= logsumexp(normalisedPrior, dims = 2)
        const1 = vec(minimum(ll + normalisedPrior, dims = 2))
        ll[ll.==0] .= -Inf
        const1 .+= logsumexp(ll + normalisedPrior .- const1, dims = 2)
        lpost = ll + normalisedPrior .- const1
        marginallikelihood = const1
    end
    MAP = Matrix{R}(undef, P, P)

    MAPprob = Vector{R}(undef, P)
    MAPmodel = Vector{R}(undef, P)

    @views @inbounds for p = 1:P
        if length(argmax(lpost[p, :])) > 0
            pt = argmax(lpost[p, :])
            MAPmodel[p] = pt
            MAP[:, p] .= parentsets[:, pt]
            MAPprob[p] = exp(lpost[p, pt])
        else
            MAPmodel[p] = NaN
            MAP[:, p] .= NaN
            MAPprob[p] = NaN
        end
    end
    MAP, MAPmodel, MAPprob, lpost
end