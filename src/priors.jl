
    function HammingPrior!(
        prior::Matrix{<:Real},
        priorGraph::Matrix{<:Real},
        P::Int,
        maxin::Int,
    )::Nothing
        @inbounds for (i, par) in enumerate(powerset(1:P, 0, maxin)), p = 1:P
            prior[p, i] = sum(x -> (x in par) != priorGraph[x, p], 1:P)
        end
    end




    function MukherjeePrior!(
        prior::Matrix{<:Real},
        priorGraph::Matrix{<:Real},
        P::Int,
        maxin::Int,
    )::Nothing
        @inbounds for (i, par) in enumerate(powerset(1:P, 0, maxin)), p = 1:P
            prior[p, i] = sum(x -> ((x in par) && (priorGraph[x, p] == 0)), 1:P)
        end

    end