
"""
function format_DBNstyle(
    d::Array{R,2};
    gradients::Bool = true,
    intercept::Bool = true,
    initialIntercept::Bool = true,
)::DBN_Data{R} where {R<:Real}

transform a data array into a DBN_Data struct which can be used for the interventional inference scheme
"""
function format_DBNstyle(
    d::Array{R,2};
    gradients::Bool = true,
    intercept::Bool = true,
    initialIntercept::Bool = true,
)::DBN_Data{R} where {R<:Real}

    cell_lines = unique(d[:, 1])
    inhibitors = unique(d[:, 2])
    stimuli = unique(d[:, 3])
    sampleTimes = sort(unique(d[:, 4]))
    times = sampleTimes
    sample_timepoints = length(sampleTimes)
    timepoints = sampleTimes[findall(x -> x in times, sampleTimes)]
    timeIntervals =
        sampleTimes[2:length(sampleTimes)] - sampleTimes[1:(length(sampleTimes)-1)]
    nodes = 1:(size(d, 2)-4)
    dm = d[:, 5:end]


    data = DBN_Data(
        Matrix{R}(undef, size(d, 1), length(nodes)),
        Matrix{R}(undef, size(d, 1), intercept + initialIntercept),
        Matrix{R}(undef, size(d, 1), length(nodes)),
        zeros(R, size(d, 1), size(d, 1))
    )

    n = 0
    ncelllines = zeros(length(cell_lines))
    ninhibitors = zeros(length(inhibitors))
    nstimuli = zeros(length(stimuli))
    ntimepoints = zeros(length(timepoints))

    interpolated = Vector{Float64}[]
    sampleInfo = Vector{Float64}[]
    cond = Float64[]

    current_condition = 0

    for cellline in cell_lines
        for i in inhibitors
            for j in stimuli
                current_condition += 1
                for k in timepoints
                    responses = findall(
                        d[:, 1] .== cellline .&&
                        d[:, 2] .== i .&&
                        d[:, 3] .== j .&&
                        d[:, 4] .== k,
                    )
                    if length(responses) > 0

                        predictor = zeros(length(nodes))
                        if k > 1
                            wh = findall(
                                d[:, 1] .== cellline .&&
                                d[:, 2] .== i .&&
                                d[:, 3] .== j .&&
                                d[:, 4] .== sampleTimes[Int(k - 1)],
                            )
                            if length(wh) == 1
                                predictor = dm[wh, nodes]
                            elseif length(wh) > 1
                                predictor = mean(dm[wh, nodes], dims = 1)
                            elseif !gradients && k > 2
                                before = findall(
                                    d[:, 1] .== cellline .&&
                                    d[:, 2] .== i .&&
                                    d[:, 3] .== j .&&
                                    d[:, 4] .== sampleTimes[Int(k - 2)],
                                )

                                # This should be able to be reformatted
                                if length(before) > 1 && length(responses) > 1
                                    predictor .=
                                        (
                                            timeIntervals[k+1] *
                                            mean(dm[before, nodes], dims = 1) +
                                            timeIntervals[k] *
                                            mean(dm[response, nodes], dims = 1)
                                        ) / (timeIntervals[k+1] + timeIntervals[k])
                                elseif length(before) == 1 && length(responses) > 1
                                    predictor .=
                                        (
                                            timeIntervals[k+1] * dm[before, nodes] +
                                            timeIntervals[k] *
                                            mean(dm[response, nodes], dims = 1)
                                        ) / (timeIntervals[k+1] + timeIntervals[k])
                                elseif length(before) > 1 && length(responses) == 1
                                    predictor .=
                                        (
                                            timeIntervals[k+1] *
                                            mean(dm[before, nodes], dims = 1) +
                                            timeIntervals[k] * dm[response, nodes]
                                        ) / (timeIntervals[k+1] + timeIntervals[k])
                                elseif length(before) == 1 && length(responses) == 1
                                    predictor .=
                                        (
                                            timeIntervals[k+1] * dm[before, nodes] +
                                            timeIntervals[k] * dm[response, nodes]
                                        ) / (timeIntervals[k+1] + timeIntervals[k])
                                else
                                    predictor .= NaN
                                end
                            else
                                predictor .= NaN
                            end
                        end

                        if !all(isnan.(predictor)) && gradients
                            n += 1
                            if length(responses) == 1
                                data.y[n, :] =
                                    (dm[response, nodes] - predictor) / timeIntervals[k-1]
                                #push!(y, (dm[response,nodes]-predictor)/timeIntervals[k-1])
                            else
                                data.y[n, :] =
                                    (mean(dm[response, nodes], dims = 1) - predictor) /
                                    timeIntervals[k-1]
                                #push!(y, (mean(dm[response,nodes],dims=1)-predictor)/timeIntervals[k-1])
                            end
                            data.Sigma[n,n] = (1/length(response)+1/length(wh))/(timeIntervals[k-1])^2
                            if (n>1 && prod(sampleInfo[n-1,:] .== [cellLine,i,j,sampleTimes[k-1]]) == 1)
                                data.Sigma[n-1,n] = -1/length(wh)/timeIntervals[k-1]/timeIntervals[k-2]
                                data.Sigma[n,n-1] = -1/length(wh)/timeIntervals[k-1]/timeIntervals[k-2]
                            end
                            data.X1[n, :] = predictor
                            push!(sampleInfo, [cellline, i, j, sampleTimes[k]])
                            push!(cond, current_condition)

                            # This looks to complicated
                            ncelllines[findall(cell_lines == cellline)] .=
                                ncelllines[findall(cell_lines == cellline)] + 1
                            ninhibitors[findall(inhibitors == i)] <
                            -ninhibitors[findall(inhibitors == i)] + 1
                            nstimuli[findall(stimuli == j)] <
                            -nstimuli[findall(stimuli == j)] + 1
                            ntimepoints[findall(timepoints == k)] <
                            -ntimepoints[findall(timepoints == k)] + 1

                            if intercept
                                push!(data.X0, ones(n))
                            end
                        elseif !all(isnan.(predictor))
                            data.Sigma[diagind(size(data.Sigma)...)] .= 1
                            for r in responses
                                n += 1
                                data.y[n, :] = dm[r, nodes]

                                data.X1[n, :] = predictor

                                push!(sampleInfo, [cellline, i, j, sampleTimes[Int(k)]])
                                push!(cond, current_condition)

                                if k > 1 && length(wh) == 0
                                    push!(
                                        interpolated,
                                        [cellLine, i, j, sampleTimes[Int(k)]],
                                    )
                                    ninterpolated += 1
                                end

                                # This looks to complicated
                                ncelllines[findall(cell_lines == cellline)] .=
                                    ncelllines[findall(cell_lines == cellline)] .+ 1
                                ninhibitors[findall(inhibitors == i)] .=
                                    ninhibitors[findall(inhibitors == i)] .+ 1
                                nstimuli[findall(stimuli == j)] .=
                                    nstimuli[findall(stimuli == j)] .+ 1
                                ntimepoints[findall(timepoints == k)] .=
                                    ntimepoints[findall(timepoints == k)] .+ 1
                                if intercept && initialIntercept && k == 1
                                    data.X0[n, :] = [1, 1]
                                elseif intercept & initialIntercept && k > 1
                                    data.X0[n, :] = [1, 0]
                                elseif intercept | (initialIntercept && k == 1)
                                    data.X0[n] = 1
                                elseif initialIntercept && k > 1
                                    data.X0[n] = 0
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    data
end

