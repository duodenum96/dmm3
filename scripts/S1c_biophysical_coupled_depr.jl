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
K = [0.825, 0.675, 0.882, 0.93, 0.85]
K = (K / maximum(K)) * 0.08
mu = 0.3
C_scaled = (C ./ maximum(C)) .* mu .* (6 / 5) # 6 / 5: little hack to keep driving network out of normalization
C_netw = cat(zeros(1, narea + 1), cat(zeros(narea, 1), C_scaled; dims=2); dims=1)

all_x = []
nsim = 10
all_ples = zeros(narea + 1, nsim)
all_ms = zeros(narea + 1, nsim)
all_sds = zeros(narea + 1, nsim)

flow = 1
fhigh = 10
fs = 1000

dynamic_idx = 1 # third network
K_outernetwork = 0.4
K_netw = cat(K_outernetwork, K; dims=1)
K_netw_t = repeat((ones(narea + 1) .* K_netw)', tsteps, 1)
# Now we gotta finesse K_netw_t[:, 1]: The time series for the K of the outer network
# Prescription: Alternate between 0.01, 0.05, 0.1, 0.05 between each 10 second
# theta_values_outer = [0.01, 0.05, 0.1, 0.05]
theta_values_outer = [0.075, 0.1, 0.11, 0.1]
sample_10sec = fs * 10
nwindow = tsteps ÷ sample_10sec
sample_10sec_idx = zeros(Int, nwindow, 2)
for i in 1:nwindow
    sample_10sec_idx[i, 1] = (i - 1) * sample_10sec + 1
    sample_10sec_idx[i, 2] = (i - 1) * sample_10sec + sample_10sec
end
c = 1
for i in 1:nwindow
    K_netw_t[sample_10sec_idx[i, 1]:sample_10sec_idx[i, 2], 1] .= theta_values_outer[c]
    c += 1
    if c > 4
        c = 1
    end
end

C_outernetwork = 0.3
C_netw[2, 1] = C_outernetwork

all_x = []
nsim = 10
all_ples = zeros(narea + 1, nsim)
all_ms = zeros(narea + 1, nsim)
all_sds = zeros(narea + 1, nsim)

Threads.@threads for j in 1:nsim
    s, x = simulate_dmm_dynamic_theta(K_netw_t, C_netw, N, tsteps)
    i_ples = zeros(narea + 1)
    push!(all_x, x)
    for i in 1:(narea + 1)
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
[lines!(ax, all_x[1][1001:end, i]) for i in 1:6]
f

m_ple = mean(all_ples; dims=2)
sd_ple = std(all_ples; dims=2)
m_m_x = mean(all_ms; dims=2)
sd_m_x = std(all_ms; dims=2)
m_sd_x = mean(all_sds; dims=2)
sd_sd_x = std(all_sds; dims=2)
# Sliding window PLE calculation
windowsize = 10000
n = length(all_x[1][1001:end, 1])
overlap = 0.99
nwindow = Int( (n - windowsize) ÷ ((1 - overlap) * windowsize) + 1 )
ples_dynamic = zeros(narea + 1, nwindow, nsim)
for i in 1:nsim
    x_selected = all_x[i][1001:end, :]
    for j in 1:6
        ples_dynamic[j, :, i] = ple_slidingwindow_overlap(
            x_selected[:, j];
            flow=flow,
            fhigh=fhigh,
            windowsize=windowsize,
            overlap=overlap,
            fs=fs,
        )
        println("j = $(j)")
    end
    println("i = $(i)")
end

sm_ples_dynamic = zeros(narea + 1, 881, nsim)
for i in 1:nsim
    for j in 1:6
        sm_ples_dynamic[j, :, i] = rolling(mean, ples_dynamic[j, :, i], 10)
    end
end

println([cor(ples_dynamic[1, :, i], ples_dynamic[2, :, i]) for i in 1:nsim])
println([cor(sm_ples_dynamic[1, :, i], sm_ples_dynamic[4, :, i]) for i in 1:nsim])
f = Figure()
ax = Axis(f[1, 1])
[lines!(ax, sm_ples_dynamic[i, :, 1]) for i in [1,3]]
f

jldsave(
    "data\\f2\\glasser_coupled.jld2";
    K_netw=K_netw,
    C_netw=C_netw,
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
    x_selected = all_x[i][10001:end, :]
    for j in 1:6
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

x_selected = all_x[1][10001:end, 1]
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
)
