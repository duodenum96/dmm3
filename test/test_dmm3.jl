cd(raw"C:\Users\duodenum\Desktop\brain_stuff\dmm3")
using Pkg
Pkg.activate(raw"C:\Users\duodenum\Desktop\brain_stuff\dmm3")
include("..\\src\\dmm3.jl")
using .dmm3
using GLMakie
using Statistics

tsteps = 100000
N = 100
c_connection = 0.2
n = 2
C = zeros(n, n)
for i = 1:n
    for j = 1:n
        if i != j
            C[i, j] = c_connection
        end
    end
end
i_theta = 0.25
theta = i_theta*ones(n)

s, x = simulate_dmm(theta, C, N, tsteps; p_s=0.88, p_ext=0.01)

f = Figure()
ax = Axis(f[1, 1])
for i = 1:n
    lines!(ax, x[:, i])
end
ylims!(ax, (-0.1, 1.2))
xlims!(ax, (9000, 14000))
f

cor(x[:, 1], x[:, 2])
