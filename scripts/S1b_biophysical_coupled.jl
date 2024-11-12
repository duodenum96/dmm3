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
using RollingFunctions
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
K = [0.8, 0.675, 0.882, 0.82, 0.76]

K = (K / maximum(K)) * 0.06
mu = 0.5
C_scaled = (C ./ maximum(C)) .* mu
C_scaled_netw = (C ./ maximum(C)) .* mu .* (6 / 5)
C_netw = cat(zeros(1, narea + 1), cat(zeros(narea, 1), C_scaled_netw; dims=2); dims=1)

dynamic_idx = 1 # third network
K_outernetwork = 0.11
K_netw = cat(K_outernetwork, K; dims=1)

C_outernetwork = 0.5
C_netw[2, 1] = C_outernetwork

all_x = []
all_x_uncoupled = []
nsim = 10
all_ples = zeros(narea + 1, nsim)
all_ms = zeros(narea + 1, nsim)
all_sds = zeros(narea + 1, nsim)

all_ples_uncoupled = zeros(narea, nsim)
all_ms_uncoupled = zeros(narea, nsim)
all_sds_uncoupled = zeros(narea, nsim)

flow = 1
fhigh = 10
fs = 1000

Threads.@threads for j in 1:nsim
    s_uncoupled, x_uncoupled = simulate_dmm(K, C_scaled, N, tsteps)
    s, x = simulate_dmm(K_netw, C_netw, N, tsteps)
    i_ples = zeros(narea + 1)
    push!(all_x, x)
    push!(all_x_uncoupled, x_uncoupled)
    for i in 1:(narea + 1)
        x_selected = x[1001:end, :]
        all_ples[i, j] = calc_ple(
            x_selected[:, i]; flow=flow, fhigh=fhigh, fs=fs, welch=true
        )
        all_ms[i, j] = mean(x_selected[:, i])
        all_sds[i, j] = std(x_selected[:, i])
    end
    for i in 1:narea
        x_selected = x_uncoupled[1001:end, :]
        all_ples_uncoupled[i, j] = calc_ple(
            x_selected[:, i]; flow=flow, fhigh=fhigh, fs=fs, welch=true
        )
        all_ms_uncoupled[i, j] = mean(x_selected[:, i])
        all_sds_uncoupled[i, j] = std(x_selected[:, i])
    end
    println(j)
end
# Check the shit
f = Figure()
ax = Axis(f[1, 1])
[lines!(ax, all_x[1][1001:end, i]) for i in 1:6]
f

f = Figure()
ax = Axis(f[1, 1])
[lines!(ax, all_x_uncoupled[1][1001:end, i]) for i in 1:5]
f

m_ple = mean(all_ples; dims=2)
m_ple_uncoupled = mean(all_ples_uncoupled; dims=2)

m_ple[2:end] .- m_ple_uncoupled

sd_ple = std(all_ples; dims=2)
m_m_x = mean(all_ms; dims=2)
sd_m_x = std(all_ms; dims=2)
m_sd_x = mean(all_sds; dims=2)
sd_sd_x = std(all_sds; dims=2)

data_rest_ples_auditory = [1.42 1.40 1.20 1.63 1.29]
cor(vec(data_rest_ples_auditory), m_ple_uncoupled)

jldsave(
    "data\\f2\\glasser_coupled.jld2";
    K_netw=K_netw,
    C_netw=C_netw,
    m_ple=m_ple,
    m_ple_uncoupled=m_ple_uncoupled,
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
powers_uncoupled = []
for i in 1:nsim
    i_power = []
    x_selected = all_x[i][1001:end, :]
    for j in 1:6
        pxx = welch_pgram(x_selected[:, j]; fs=fs)
        push!(i_power, pxx.power)
    end
    i_power = stack(i_power)
    push!(powers, i_power)

    i_power_uncoupled = []
    x_selected = all_x_uncoupled[i][1001:end, :]
    for j in 1:5
        pxx = welch_pgram(x_selected[:, j]; fs=fs)
        push!(i_power_uncoupled, pxx.power)
    end
    i_power_uncoupled = stack(i_power_uncoupled)
    push!(powers_uncoupled, i_power_uncoupled)
end
powers = stack(powers)
mean_power = dropdims(mean(powers; dims=3); dims=3)
sd_power = dropdims(std(powers; dims=3); dims=3)
upper = mean_power .+ sd_power
lower = mean_power .- sd_power

powers_uncoupled = stack(powers_uncoupled)
mean_power_uncoupled = dropdims(mean(powers_uncoupled; dims=3); dims=3)
sd_power_uncoupled = dropdims(std(powers_uncoupled; dims=3); dims=3)
upper_uncoupled = mean_power_uncoupled .+ sd_power_uncoupled
lower_uncoupled = mean_power_uncoupled .- sd_power_uncoupled

x_selected = all_x[1][1001:end, 1]
pxx = welch_pgram(x_selected; fs=fs) # Get example freq

# Save PSDs
jldsave(
    "data\\f2\\glasser_coupled_psd.jld2";
    pxx=pxx,
    powers=powers,
    mean_power=mean_power,
    sd_power=sd_power,
    upper=upper,
    lower=lower,
    powers_uncoupled=powers_uncoupled,
    mean_power_uncoupled=mean_power_uncoupled,
    sd_power_uncoupled=sd_power_uncoupled,
    upper_uncoupled=upper_uncoupled,
    lower_uncoupled=lower_uncoupled,
)
