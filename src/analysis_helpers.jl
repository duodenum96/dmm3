using Statistics
using LinearAlgebra
using StatsBase
using MAT

function cormat_crosscor(x; lags=-79:79)
    # x: matrix of time x rois
    n = size(x, 2)
    cormat = zeros(n, n)
    Threads.@threads for i in 1:n
        for j in 1:n
            if i != j
                cormat[i, j] = maximum(crosscor(x[:, i], x[:, j], lags))
            end
        end
    end
    return cormat
end

function calculate_stuff(x; trunc=100000, windowsize=5000, flow=0.001, fhigh=0.01, xcor=false, overlap=true, spear=false, fs=30, siegel=false)
    nroi = size(x, 2)
    x_trunc = x[trunc:end, :]
    
    plets = []
    for i in 1:nroi
        if overlap
            push!(plets, ple_slidingwindow(x_trunc[:, i], flow=flow, fhigh=fhigh, windowsize=windowsize, fs=fs, siegel=siegel))
            println(i)
        else
            push!(plets, ple_slidingwindow_nooverlap(x_trunc[:, i], flow=flow, fhigh=fhigh, windowsize=windowsize, fs=fs, siegel=siegel))
            println(i)
        end
    end

    plets = stack(plets)

    if xcor
        fc_data = cor(x_trunc)
        fc_ple = cormat_crosscor(plets)
    else
        if spear
            fc_ple = cor(plets)
            fc_data = cor(x)    
        else
            fc_ple = cor(plets)
            fc_data = cor(x)
        end
        fc_ple[I(nroi)] .= 0
        fc_data[I(nroi)] .= 0
    end

    return plets, fc_ple, fc_data, plets
end

function readmat(path)
    file = matopen(path)
    data = read(file)
    close(file)
    return data
end

function scale(x, min_y, max_y)
    return ( (x .- minimum(x)) ./ (maximum(x) - minimum(x)) ) .* (max_y - min_y) .+ min_y
end

function siegelslopes(x, y)
    n = length(x)
    slopes = zeros(n)
    c = 1
    for i in 1:n
        x_i = x[i]
        y_i = y[i]
        slopes_i = zeros(n-1)
        c2 = 1
        for j in 1:n
            if j != i
                x_j = x[j]
                y_j = y[j]
                slopes_i[c2] = (y_j - y_i) / (x_j - x_i)
                c2 += 1
            end
        end
        slopes[c] = median(slopes_i)
        c += 1
    end
    return median(slopes)
end

##### Test SiegelSlopes
# x = rand(1000)
# y = 0.5 * x .+ 0.1*rand(1000)

# siegelslopes(x, y)