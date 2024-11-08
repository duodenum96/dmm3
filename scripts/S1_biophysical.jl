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
C = 10 .^ scmat[idx, idx] # scmat is in log scale
C[isnan.(C)] .= 0

narea = length(idx)

###################### Set parameters
N = 100
tsteps = Int(1e5)
K = [0.825, 0.675, 0.882, 0.93, 0.85]
K = (K / maximum(K)) * 0.095
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
###### Simulate
Threads.@threads for j in 1:nsim
    s, x = simulate_dmm(K, C_scaled, N, tsteps; p_s=0.88, p_ext=0.01)
    i_ples = zeros(narea + 1)
    push!(all_x, x)
    for i in 1:narea
        x_selected = x[1001:end, :]
        all_ples[i, j] = calc_ple(x_selected[:, i]; flow=flow, fhigh=fhigh, fs=fs)
        all_ms[i, j] = mean(x_selected[:, i])
        all_sds[i, j] = std(x_selected[:, i])
    end
    println(j)
end
# Check the shit
f = Figure()
ax = Axis(f[1, 1])
[lines!(ax, all_x[1][10001:end, i]) for i in 1:5]
################################## Calculate means and stds ##################################

m_ple = mean(all_ples; dims=2)
sd_ple = std(all_ples; dims=2)
m_m_x = mean(all_ms; dims=2)
sd_m_x = std(all_ms; dims=2)
m_sd_x = mean(all_sds; dims=2)
sd_sd_x = std(all_sds; dims=2)

jldsave(
    "data\\f2\\glasser.jld2";
    K=K,
    C_scaled=C_scaled,
    m_ple=m_ple,
    sd_ple=sd_ple,
    m_m_x=m_m_x,
    sd_m_x=sd_m_x,
    m_sd_x=m_sd_x,
    sd_sd_x=sd_sd_x,
    all_ples=all_ples,
    all_ms=all_ms,
    all_sds=all_sds,
    all_x=all_x,
)

#### PSD Figure
powers = []
for i in 1:nsim
    i_power = []
    x_selected = all_x[i][1001:end, :]
    for j in 1:5
        pxx = welch_pgram(x_selected[:, j]; fs=fs)
        push!(i_power, pxx.power)
    end
    i_power = stack(i_power)
    push!(powers, i_power)
end
powers = stack(powers)
mean_power = dropdims(mean(powers; dims=3); dims=3)
sd_power = dropdims(std(powers; dims=3); dims=3)
upper = mean_power .+ sd_power
lower = mean_power .- sd_power

x_selected = all_x[1][1001:end, 1]
pxx = welch_pgram(x_selected; fs=fs) # Get example freq

# Save PSDs
jldsave(
    "data\\f2\\glasser_psd.jld2";
    pxx=pxx,
    powers=powers,
    mean_power=mean_power,
    sd_power=sd_power,
    upper=upper,
    lower=lower,
)
