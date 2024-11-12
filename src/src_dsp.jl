using DSP
using Polynomials
using RollingFunctions
using Plots

function calc_ple(x; flow=0.001, fhigh=0.01, fs=30, welch=false, siegel=false)
    """
    x: vector
    flow, fhigh: scalar
    calculate ple between flow and fhigh
    """
    if welch
        pxx = welch_pgram(x; fs=fs)
    else
        pxx = periodogram(x; fs=fs)
    end
    freq = pxx.freq
    power = pxx.power
    freq_select = freq[(freq .>= flow) .& (freq .<= fhigh)]
    power_select = power[(freq .>= flow) .& (freq .<= fhigh)]
    if siegel
        ple = -siegelslopes(log10.(freq_select), log10.(power_select))
    else
        polyfit = Polynomials.fit(Polynomial, log10.(freq_select), log10.(power_select), 1)
        ple = -polyfit.coeffs[2]
    end
    return ple
end

function ple_slidingwindow(x; flow=0.001, fhigh=0.01, windowsize=10000, fs=30, siegel=false)
    rollple(x) = calc_ple(x; flow=flow, fhigh=fhigh, fs=fs, siegel=siegel)
    slide_ple = rolling(rollple, x, windowsize)
    return slide_ple
end

function ple_slidingwindow_parallel(
    x; flow=0.001, fhigh=0.01, windowsize=10000, fs=30, siegel=false
)
    n = length(x)
    n_window = n - windowsize
    slide_ple = zeros(n_window)

    Threads.@threads for i in 1:(n - windowsize + 1)
        window = [i, i + windowsize - 1]
        slide_ple[i] = calc_ple(
            x[window[1]:window[2]]; flow=flow, fhigh=fhigh, fs=fs, siegel=siegel
        )
    end

    return slide_ple
end

function ple_slidingwindow_nooverlap(
    x; flow=0.001, fhigh=0.01, windowsize=10000, fs=30, siegel=false
)
    n = length(x)
    n_window = n ÷ windowsize
    slide_ple = zeros(n_window)
    Threads.@threads for i in 1:n_window
        window = [(i - 1) * windowsize + 1, i * windowsize]
        slide_ple[i] = calc_ple(
            x[window[1]:window[2]]; flow=flow, fhigh=fhigh, fs=fs, siegel=siegel
        )
    end
    return slide_ple
end

"""
Calculate the start and stop indices for the sliding window technique with % overlap. 
overlap is between 0 and 1.
"""
function sl_overlap_indices(n, windowsize, overlap)
    nwindow = (n - windowsize) ÷ ((1 - overlap) * windowsize) + 1 # (alternatively, div(n, windowsize))
    return [
        [
            floor(Int, 1 + (1 - overlap) * windowsize * (i - 1)),
            floor(Int, (1 - overlap) * windowsize * (i - 1) + windowsize),
        ] for i in 1:nwindow
    ]
end

function ple_slidingwindow_overlap(
    x; flow=0.001, fhigh=0.01, windowsize=1000, overlap=0.5, fs=30, siegel=false
)
    n = length(x)
    idx = sl_overlap_indices(n, windowsize, overlap)
    n_idx = length(idx)
    slide_ple = zeros(n_idx)

    for i in 1:n_idx
        window = idx[i]
        slide_ple[i] = calc_ple(
            x[window[1]:window[2]]; flow=flow, fhigh=fhigh, fs=fs, siegel=siegel
        )
    end
    return slide_ple
end
