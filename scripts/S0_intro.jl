cd(raw"C:\Users\duodenum\Desktop\brain_stuff\dmm3")
using Pkg
Pkg.activate(raw"C:\Users\duodenum\Desktop\brain_stuff\dmm3")
include("..\\src\\dmm3.jl")
using .dmm3
using GLMakie
using Statistics
using DSP
using LinearAlgebra
using JLD2
using Polynomials
import Plots: palette
using LaTeXStrings
using Format

Ks = collect(0.01:0.01:0.11)
n_K = length(Ks)
N = 100
tsteps = Int(1e5)

fs = 1000
windowsize = 3000

nsim = 10

all_x = zeros(tsteps, nsim, n_K)
for i in eachindex(Ks)
    for j in 1:nsim
        s, x = simulate_dmm_1d(Ks[i], N, tsteps)
        all_x[:, j, i] = x
    end
    println(i)
end

jldsave(raw"data\f1\K_search.jld2"; all_x=all_x, Ks=Ks)

########### Figure 1: Power Spectra, Correlation between K and PLEs
flow = 1
fhigh = 10
i = 1;
j = 1;
x = all_x[:, j, i]
x_selected = copy(x)[1001:end]
pxx = welch_pgram(x_selected; fs=fs)
nfft = length(pxx.freq)

all_power_spectra = zeros(nfft, nsim, n_K)
ples = zeros(nsim, n_K)
for i in eachindex(Ks)
    for j in 1:nsim
        x = all_x[:, j, i]
        x_selected = copy(x)[1001:end]
        pxx = welch_pgram(x_selected; fs=fs)

        freqs = pxx.freq
        power = pxx.power
        freq_select = freqs[(freqs .>= flow) .& (freqs .<= fhigh)]
        power_select = power[(freqs .>= flow) .& (freqs .<= fhigh)]
        polyfit = Polynomials.fit(Polynomial, log.(freq_select), log.(power_select), 1)
        ples[j, i] = -polyfit.coeffs[2]
        all_power_spectra[:, j, i] = power
    end
end

jldsave(
    raw"data\f1\S0_PLEvals.jld2"; ples=ples, freq=freqs, all_power_spectra=all_power_spectra
)

################################# Figure #############################

psddata = load(raw"data\f1\S0_PLEvals.jld2")
ples = psddata["ples"]
freqs = psddata["freq"]
all_power_spectra = psddata["all_power_spectra"]

mean_power = dropdims(mean(all_power_spectra; dims=2); dims=2)
sd_power = dropdims(std(all_power_spectra; dims=2); dims=2)
upper = mean_power .+ sd_power
lower = mean_power .- sd_power

# Three Examples: K = [0.1, 1.0, 1.4]
# f = Figure(size=(700, 1600))
f = Figure(; size=(1200, 400))
### PSD
ax1 = Axis(
    f[1, 1];
    xscale=log10,
    yscale=log10,
    limits=(1, 10, nothing, nothing),
    xlabel="Frequency",
    ylabel="Power",
)
for i in 1:n_K
    band!(
        ax1,
        pxx.freq,
        lower[:, i],
        upper[:, i];
        alpha=0.2,
        color=palette(:twelvebitrainbow, 1:n_K)[i],
    )
    lines!(
        ax1,
        pxx.freq,
        mean_power[:, i];
        alpha=1.0,
        label="K = $(Ks[i])",
        color=palette(:twelvebitrainbow, 1:n_K)[i],
    )
end
# vlines!(ax1, [flow, fhigh], color=:black)
axislegend(ax1)

ax2 = Axis(
    f[1, 2]; xticks=0:0.1:Ks[end], xlabel="K", ylabel="PLE", xticklabelrotation=pi / 4
)
for i in eachindex(Ks)
    Makie.scatter!(
        ax2, ones(nsim) * Ks[i], ples[:, i]; color=palette(:twelvebitrainbow, 1:n_K)[i]
    )
end

f
save("figs\\f1\\One_Network.png", f; px_per_unit=10.0)

################################# Figure BUT no Log Scale #############################

psddata = load(raw"data\f1\S0_PLEvals.jld2")
ples = psddata["ples"]
freqs = psddata["freq"]
all_power_spectra = psddata["all_power_spectra"]

mean_power = dropdims(mean(all_power_spectra; dims=2); dims=2)
sd_power = dropdims(std(all_power_spectra; dims=2); dims=2)
upper = mean_power .+ sd_power
lower = mean_power .- sd_power

# Three Examples: K = [0.1, 1.0, 1.4]
# f = Figure(size=(700, 1600))
f = Figure(; size=(1200, 400))
### PSD
ax1 = Axis(
    f[1, 1]; 
    limits=(0.001, 1.3, -5, -2), 
    xlabel="log(Frequency)",
    ylabel="log(Power)",
    xtickformat = values -> [cfmt("%.1f", 10.0 .^ value) for value in values]
)
for i in 1:n_K
    band!(
        ax1,
        log10.(freqs),
        log10.(lower[:, i]),
        log10.(upper[:, i]);
        alpha=0.2,
        color=palette(:twelvebitrainbow, 1:n_K)[i],
    )
    lines!(
        ax1,
        log10.(freqs),
        log10.(mean_power[:, i]);
        alpha=1.0,
        label="K = $(Ks[i])",
        color=palette(:twelvebitrainbow, 1:n_K)[i],
    )
end

ax2 = Axis(
    f[1, 2]; xticks=Ks, xlabel=L"$\theta$", ylabel="PLE", xticklabelrotation=pi / 4
)
for i in eachindex(Ks)
    Makie.scatter!(
        ax2, ones(nsim) * Ks[i], ples[:, i]; color=palette(:twelvebitrainbow, 1:n_K)[i]
    )
end

f
save("figs\\f1\\One_Network.png", f; px_per_unit=10.0)
