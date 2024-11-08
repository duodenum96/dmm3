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
################################# Load data
glasser_sc_path = raw"C:\Users\Duodenum\Desktop\brain_stuff\modeling_utilities\averageConnectivity_Fpt.mat"
myelinpath = raw"C:\Users\duodenum\Desktop\brain_stuff\modeling_utilities\parcelMyelinationIndices.mat"
sc = readmat(glasser_sc_path)

scmat = sc["Fpt"]
ids = sc["parcelIDs"]

################## Let's just try left ##################
# rois: V: v1, v2, v3, v4; A: A1, LBC, MBC, PBC, RIC
lr = "L_"
a_roi_names = lr .* ["A1", "LBelt", "MBelt", "PBelt", "RI"]

idx = getindex.([findall(ids .== i)[1] for i in a_roi_names], 1)
C = 10 .^ scmat[idx, idx]
C[isnan.(C)] .= 0

narea = length(idx)

###################### Set parameters
N = 100
tsteps = Int(1e5)
K = [0.825, 0.675, 0.882, 0.93, 0.85]
Ks = (K / maximum(K)) * 0.095
mu = 0.2
C_scaled = (C ./ maximum(C)) .* mu

all_x = []
nsim = 10
all_ples = zeros(narea + 1, nsim)
all_ms = zeros(narea + 1, nsim)
all_sds = zeros(narea + 1, nsim)

flow = 1
fhigh = 10
fs = 1000

######### Load data and process ############

uncoupled = load("data\\f2\\glasser.jld2")
K_unc = uncoupled["K"]
C_unc = uncoupled["C_scaled"]
m_ple_unc = uncoupled["m_ple"][1:5] # 1:5 due to little typo
sd_ple_unc = uncoupled["sd_ple"][1:5]
m_sd_x_unc = uncoupled["m_sd_x"][1:5]
sd_sd_x_unc = uncoupled["sd_sd_x"][1:5]
all_ples = uncoupled["all_ples"][1:5, :]

uncoupled_psd = load("data\\f2\\glasser_psd.jld2")
mean_power_unc = uncoupled_psd["mean_power"]
sd_power_unc = uncoupled_psd["sd_power"]
lower_unc = uncoupled_psd["lower"]
upper_unc = uncoupled_psd["upper"]

x_selected = uncoupled["all_x"][1][1001:end, 1]
pxx = welch_pgram(x_selected; fs=fs) # Get example freq

################# Plot uncoupled ####################

a_roi_names_cropped = [a_roi_names[i][3:end] for i in eachindex(a_roi_names)]
data_rest_ples_auditory = [1.42 1.40 1.20 1.63 1.29]

f = Figure(; size=(800, 400))
### PSD
ax1 = Axis(
    f[1, 1];
    limits=(0.001, 1.3, -5, -2),
    xlabel="log(Frequency)",
    ylabel="log(Power)",
    xtickformat=values -> [cfmt("%.1f", 10.0 .^ value) for value in values],
)
for i in 1:narea
    band!(
        ax1,
        log10.(pxx.freq),
        log10.(lower_unc[:, i]),
        log10.(upper_unc[:, i]);
        alpha=0.2,
        color=palette(:twelvebitrainbow, 1:narea)[i],
    )
    lines!(
        ax1,
        log10.(pxx.freq),
        log10.(mean_power_unc[:, i]);
        alpha=1.0,
        label="K = $(Ks[i])",
        color=palette(:twelvebitrainbow, 1:narea)[i],
    )
end

# PLE
ax2 = Axis(
    f[1, 2]; xticks=(1:narea, a_roi_names_cropped), ylabel="PLE", xticklabelrotation=pi / 4
)
for i in eachindex(Ks)
    Makie.scatter!(
        ax2, ones(nsim) * i, all_ples[i, :]; color=palette(:twelvebitrainbow, 1:narea)[i]
    )
end

save("figs\\f2\\Uncoupled_PLE.png", f; px_per_unit=10.0)


####################### Plot Coupled #######################
#### Load data

coupled = load("data\\f2\\glasser_coupled.jld2")
K_netw = coupled["K_netw"]
C_netw = coupled["C_netw"]
m_ple = vec(coupled["m_ple"])
sd_ple = vec(coupled["sd_ple"])
m_sd_x = vec(coupled["m_sd_x"])
sd_sd_x = vec(coupled["sd_sd_x"])

coupled_psd = load("data\\f2\\glasser_coupled_psd.jld2")
mean_power = coupled_psd["mean_power"]
sd_power = coupled_psd["sd_power"]
lower = coupled_psd["lower"]
upper = coupled_psd["upper"]

a_roi_names_netw = cat(
    "Input", [a_roi_names[i][3:end] for i in eachindex(a_roi_names)]; dims=1
)

f = Figure(; size=(800, 400))
### PSD
ax1 = Axis(f[1, 1]; xscale=log10, yscale=log10, limits=(0.02, 15, nothing, nothing))
# for i in 1:5
#     lines!(ax1, pxx.freq, mean_power_unc[:, i], alpha=0.5, linestyle=:dash)
# end
lines!(ax1, pxx.freq, mean_power[:, 1]; alpha=1.0, label=a_roi_names_netw[1], color=:black)
band!(ax1, pxx.freq, lower[:, 1], upper[:, 1]; alpha=0.2, color=:black)

for i in 2:6
    lines!(ax1, pxx.freq, mean_power[:, i]; alpha=1.0, label=a_roi_names_netw[i])
    band!(ax1, pxx.freq, lower[:, i], upper[:, i]; alpha=0.2)
end
vlines!(ax1, [0.05, 0.5]; color=:black)
axislegend(ax1)

# PLE
ax2 = Axis(
    f[1, 2];
    xticks=(1:(narea + 1), a_roi_names_netw),
    ylabel="PLE",
    xticklabelrotation=pi / 4,
)

Makie.scatter!(
    ax2, 2:(narea + 1), vec(m_ple_unc); color=:black, label="Uncoupled", marker=:cross
)
Makie.errorbars!(ax2, 2:(narea + 1), vec(m_ple_unc), vec(sd_ple_unc); color=:black)

Makie.scatter!(ax2, 1:(narea + 1), vec(m_ple); color=:purple, label="Coupled")
Makie.errorbars!(ax2, 1:(narea + 1), vec(m_ple), vec(sd_ple); color=:purple)
axislegend(ax2)

save("figs\\f2\\Coupled_PLE.png", f; px_per_unit=10.0)
