"""
function InterventionalInference(
    dbn_data::DBN_Data{T},
    Z::Array{<:Real},
    maxindegree::Int;
    covariance::Bool = false,
    perfectOut::Bool = false,
    priorStrength::Vector{<:Real} = [3],
    allowSelfEdges::Bool = false,
    perfectIn::Bool = false,
    fixedEffectIn::Bool = false,
    fixedEffectOut::Bool = false,
    mechanismChangeIn::Bool = false,
    mechanismChangeOut::Bool = false,
    priorGraph::Union{Missing,Matrix{<:Real}} = missing,
    g1::Union{Int,Missing} = missing,
    priorType::AbstractString = "uninformed",
)::Tuple where {T<:Real}

Run the Interventional Inference function.
"""
function InterventionalInference(
    dbn_data::DBN_Data{T},
    Z::Array{<:Real},
    maxindegree::Int;
    covariance::Bool = false,
    perfectOut::Bool = false,
    priorStrength::Vector{<:Real} = [3],
    allowSelfEdges::Bool = false,
    perfectIn::Bool = false,
    fixedEffectIn::Bool = false,
    fixedEffectOut::Bool = false,
    mechanismChangeIn::Bool = false,
    mechanismChangeOut::Bool = false,
    priorGraph::Union{Missing,Matrix{<:Real}} = missing,
    g1::Union{Int,Missing} = missing,
    priorType::AbstractString = "uninformed",
)::Tuple where {T<:Real}

    IP = InterventionPattern(;
        allowSelfEdges = allowSelfEdges,
        perfectOut = perfectOut,
        perfectIn = perfectIn,
        fixedEffectOut = fixedEffectOut,
        fixedEffectIn = fixedEffectIn,
        mechanismChangeOut = mechanismChangeOut,
        mechanismChangeIn = mechanismChangeIn,
    )

    #X1_fun = deepcopy(X1) # Do deepcopy to not alter input    
    P = size(dbn_data.y, 2)
    n = size(dbn_data.y, 1)
    g = ismissing(g1) ? n : g1
    a = size(dbn_data.X0, 2)

    @assert priorType in ["uninformed", "Hamming", "Mukherjee"] "$priorType is not allowed as priorType. Must be 'uninformed', 'Hamming' or 'Mukherjee'."


    if priorType in ["Hamming", "Mukherjee"]
        if !ismissing(priorGraph)
            @assert size(priorGraph) == (P, P) "priorGraph must be of dimension (P x P) "
        end
    end


    parentvec = Parent_struct{T}[]
    for p = 1:P
        obs =
            IP.perfectIn && maximum(Z[:, p]) == 1 ? vec(findall(Z[:, p] .== 0)) :
            collect(1:n)
        p_struct = disentangle(Val(covariance), dbn_data, obs, p)
        if IP.perfectOut
            wh = findall(vec(Z[:, p]) .== 1)
            n_wh = settdiff(1:n, wh)
            p_struct.X1[wh] .= NaN
            p_struct.X1[n_wh] .= crossfun1(p_struct.X0[n_wh, :]) * p_struct.X1[n_wh]
            #else
            #    p_struct.X1[:] .= p_struct.IP0 * p_struct.X1            
        end
        p_struct.X1[findall(isnan.(p_struct.X1))] .= 0

        p_struct = predictor_mechanism_out(
            Val(IP.mechanismChangeOut),
            dbn_data,
            p_struct,
            p,
            Z,
            covariance,
        )

        # relevant function is selected on dispatch
        p_struct = fixed_effect_out(Val(IP.fixedEffectOut), p_struct, Z, p)
        push!(parentvec, p_struct)
    end


    # 5: prior
    n_grphs = count_graphs(P, maxindegree)

    prior = zeros(P, n_grphs) # Base Case
    if priorType == "Hamming"
        HammingPrior!(prior, priorGraph, P, maxindegree)
    elseif priorType == "Mukherjee"
        MukherjeePrior!(prior, priorGraph, P, maxindegree)
    end


    println("Model Inference...")
    # 6: Initilisation
    ll, parentsets =
        main_loop(dbn_data, parentvec, n_grphs, maxindegree, n, Z, g, a, IP, covariance)

    # 8: renornmalization & MAP    
    MAP, MAPmodel, MAPprob, lpost =
        MAP_function(P, n_grphs, parentsets, priorStrength, prior, ll, maxindegree)

    # 9: Model averaging
    log_pep = log_posterior(lpost, parentsets, P)

    println("Fitted Values...")
    fitted = fittedValues(
        dbn_data,
        parentvec,
        n_grphs,
        maxindegree,
        n,
        Z,
        g,
        IP,
        covariance,
        lpost,
    )



    log_pep, MAP, MAPmodel, MAPprob, fitted
end


function main_loop(
    dbn_data::DBN_Data{T},
    parentvec::Vector{Parent_struct{T}},
    n_grphs::Int,
    maxindegree::Int,
    n::Int,
    Z::Matrix{<:Real},
    g::Int,
    a::Int,
    IP::InterventionPattern{Bool},
    covariance::Bool,
) where {T<:Real}
    P = length(parentvec)
    ll = zeros(P, n_grphs)
    parentsets = zeros(P, n_grphs)
    # 7: Main Loop
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
                intersect(uninhibitedResponses, findall(vec(maximum(Z, dims = 1) .== 1)))
            uninhibitedResponses = setdiff(uninhibitedResponses, inhibitedResponses)
        end


        if length(p_inds) == 0
            b = 0
            @inbounds for p in uninhibitedResponses
                ll[p, m] =
                    -b / 2 * log(1 + g) -
                    (n - a) / 2 * log(dot(parentvec[p].y, parentvec[p].y))
            end
            @inbounds for p in inhibitedResponses
                ll[p, m] =
                    -b / 2 * log(1 + g) -
                    (length(parentvec[p].obs) - a) / 2 *
                    log(dot(parentvec[p].y, parentvec[p].y))
            end
        else

            X0 = hcat([parentvec[par].X0 for par in p_inds]...)
            X1 = hcat([parentvec[par].X1 for par in p_inds]...)
            H = crossfun1(X1, g / (g + 1))
            b = size(X1, 2)
            @inbounds for p in uninhibitedResponses
                """
                get uninhibited responses here
                """
                
                ll[p, m] +=
                    -b / 2 * log(1 + g) -
                    (n - a) / 2 * log(dot(parentvec[p].y, H, parentvec[p].y))
            end

            @inbounds for p in inhibitedResponses
                H, b =
                    H_func(dbn_data, parentvec, Z, X1, X0, IP, g, p, p_inds, n, covariance)
                ll[p, m] =
                    -b / 2 * log(1 + g) -
                    (length(parentvec[p].obs) - a) / 2 *
                    log(dot(parentvec[p].y, H, parentvec[p].y))

            end
        end
        next!(prog_bar)
    end # graphs
    ll, parentsets
end

function log_posterior(
    lpost::Matrix{R},
    parentsets::Matrix{R},
    P::Int,
)::Matrix{R} where {R<:Real}
    lpep = Matrix{R}(undef, P, P)
    @views @inbounds for i = 1:P, j = 1:P
        lpep[i, j] = logsumexp(lpost[j, findall(parentsets[i, :] .== 1)])
    end
    lpep
end

function MAP_function(
    P::Int,
    n_grphs::Int,
    parentsets::Matrix{R},
    priorStrength::Vector{<:Real},
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
        normalisedPrior -= priorStrength .* prior
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

function H_func(
    dbn_data::DBN_Data{R},
    parentvec::Vector{Parent_struct{R}},
    Z::Matrix{<:Real},
    X1_f::Matrix{R},
    X0::Matrix{R},
    IP::InterventionPattern{Bool},
    g::Int,
    p::Int,
    p_inds::Vector{Int},
    n::Int,
    covariance::Bool,
)::Tuple{Matrix{R},Int} where {R<:Real}
    X1 = predictors_mechanismchangein(
        Val(IP.mechanismChangeIn),
        parentvec[p],
        deepcopy(X1_f),
        dbn_data,
        Z,
        p,
    )                    #
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
        Z_new[:] .= Z[parentvec[p].obs, to_use]
        X1 = hcat(X1, Z_new)
    end
    # end assembling predictors

    X1 = Xfun(Val(covariance), X1, X0, dbn_data, parentvec[p].obs)

    H = I(n)
    b = size(X1, 2)
    if b != 0
        H = crossfun1(X1, g / (g + 1.0))
    end
    H, b
end