module dmm3

export simulate_dmm
export simulate_dmm_1d
export simulate_dmm_dynamic_theta
export prepare_dynamic_K

include("src_dsp.jl")
export calc_ple
export ple_slidingwindow
export ple_slidingwindow_overlap

include("analysis_helpers.jl")
export calculate_stuff
export cormat_crosscor
export readmat
export scale
export siegelslopes

# include("bold.jl")
# bw(z, dt, alpha=0.32, kappa=0.65, gamma=0.41, tau=0.98, rho=0.34, V0=0.02) = balloonWindkessel(z', dt, alpha, kappa, gamma, tau, rho, V0) # transpose z to be consistent
# export bw

function _initialize_dmm(tsteps, N)
    """
    tsteps: 1 x 1
    N: number of units in network, 1 x 1

    return
    s: n_sides x n_sides x t
    x: 1 x t
    neighbors: n_sides x n_sides
    """
    n_sides = Int(sqrt(N))
    x = zeros(tsteps)
    s = zeros(n_sides, n_sides, tsteps)
    rands = rand(n_sides, n_sides)
    s1view = view(s, :, :, 1)
    s1view[rands .< 0.5] .= 0.0
    s1view[rands .>= 0.5] .= 1.0

    up = sum(s1view .== 1.0)
    # down = sum(s1view .== 0.0)

    x[1] = up / N
    neighbors = zeros(n_sides, n_sides)
    neighbors = _update_neighbors!(neighbors, s1view)

    return s, x, neighbors
end

function initialize_dmm_non(tsteps, N, n)
    """
    Initialize DMMs for network of networks
    tsteps: scalar
    N: scalar, # of units in a network
    theta: n x 1 vector: for each network

    return:
    s: N x N x t x n
    x: t x n
    neighbors: N x N x n
    """
    n_sides = Int(sqrt(N))
    s = zeros(n_sides, n_sides, tsteps, n)
    x = zeros(tsteps, n)
    neighbors = zeros(n_sides, n_sides, n)
    for i in 1:n
        i_s, i_x, i_neighbors = _initialize_dmm(tsteps, N)
        s[:, :, 1, i] .= i_s[:, :, 1]
        x[:, i] .= i_x
        neighbors[:, :, i] .= i_neighbors
    end

    return s, x, neighbors
end


function _update_neighbors!(neighbors, s)
    # Neighbors w/ periodic boundary conditions
    # Ones in the middle
    n_sides = size(neighbors, 1)

    for i in 2:(n_sides-1)
        for j in 2:(n_sides-1)
            neighbors[i, j] = s[i - 1, j] + s[i + 1, j] + s[i, j - 1] + s[i, j + 1]
        end
    end

    
    for i in 2:(n_sides - 1)
        # Column 1
        neighbors[i, 1] = s[i - 1, 1] + s[i + 1, 1] + s[i, n_sides] + s[i, 2]
        # Column n
        neighbors[i, n_sides] = s[i - 1, n_sides] + s[i + 1, n_sides] + s[i, n_sides - 1] + s[i, 1]
        # Top row
        neighbors[1, i] = s[n_sides, i] + s[2, i] + s[1, i - 1] + s[1, i + 1]
        # Bottom row
        neighbors[n_sides, i] = s[1, i] + s[n_sides-1, i] + s[n_sides, i - 1] + s[n_sides, i + 1]
    end

    # Corners
    # Lower right
    neighbors[n_sides, n_sides] = s[n_sides - 1, n_sides] + s[1, n_sides] + s[n_sides, n_sides - 1] + s[n_sides, 1]
    # Lower left
    neighbors[n_sides, 1] = s[n_sides - 1, 1] + s[1, 1] + s[n_sides, n_sides] + s[n_sides, 2]
    # Upper right
    neighbors[1, n_sides] = s[n_sides, n_sides] + s[2, n_sides] + s[1, n_sides - 1] + s[1, 1]
    # Upper left
    neighbors[1, 1] = s[n_sides, 1] + s[2, 1] + s[1, n_sides] + s[1, 2]

    return neighbors
end

function f(x_local, x_global, theta; n=4)
    return 1 .- exp.( -((theta ./ n) .* x_local .+ (x_global./n)))
end

function update_s(s, neighbors, theta, x, C; p_s=0.88, p_ext=0.01)
    """
    Update s
    s: N x N x n
    neighbors: N x N x n
    theta: 1 x n
    x: 1 x n (view of x[t, :])
    C: Coupling matrix: n x n

    return: s_new: N x N

    formula for updating:
        P(↑ → ↓) = α₂ - β₂ f(∑ⱼsⱼ + ∑ⱼCᵢⱼ xⱼ)
        P(↓ → ↑) = α₁ + β₁ f(∑ⱼsⱼ + ∑ⱼCᵢⱼ xⱼ)
        f(x) = 1 - exp(-(θ/n) x)

    in discrete time:
        P(↓ → ↑) = p_ext + f(∑ⱼsⱼ + ∑ⱼCᵢⱼ xⱼ)
        P(↑ → ↓) = 1 - p_ext - p_s - f(∑ⱼsⱼ + ∑ⱼCᵢⱼ xⱼ)

    values for p_ext, p_s from Shi et al.: 
        p_s = 0.88
        p_ext = 0.0001
    """
    N = size(s, 1)
    n_networks = size(C, 1)
    s_new = copy(s)
    
    for i in 1:n_networks
        i_neighbors = view(neighbors, :, :, i)
        i_s = view(s, :, :, i)
        i_s_new = view(s_new, :, :, i)

        couplingterm = C[i, :]' * x
        localterm = i_neighbors # 4 neighbors for each neuron
        p1to0 = 1 .- p_ext .- p_s .- f(localterm, couplingterm ./ n_networks, theta[i])
        p0to1 = p_ext .+ f(localterm, couplingterm ./ n_networks, theta[i])
        # Update 
        r = rand(N, N)
        i_s_new[(i_s .== 1) .&  (r .<= p1to0)] .= 0.0
        r = rand(N, N)
        i_s_new[(i_s .== 0) .&  (r .<= p0to1)] .= 1.0
    end

    return s_new
end

function update_states(theta, s, x, neighbors, C; p_s=0.88, p_ext=0.01)
    """
    Take states s[t], x[t], neighbors and return s[t+1], x[t+1] and updated neighbors
    N: # of elements in one side of each network
    n: # of networks
    theta: 1 x n
    s: N x N x n (    enter a view of s here, view(s[:, :, t, :])    )
    x: n (    again, enter a view: view(x[t, :])   )
    neighbors: N x N x n
    C: Coupling matrix: n x n

    return: 
    s_new: N x N x n
    x_new: n
    neighbors_new: N x N x n
    """
    n = size(s, 3)
    N = size(s, 1)

    neighbors_new = zero(neighbors)
    for i in 1:n
        neighbors_new[:, :, i] .= _update_neighbors!(neighbors[:, :, i], s[:, :, i])
    end
    s_new = update_s(s, neighbors_new, theta, x, C; p_s=p_s, p_ext=p_ext)
    nsquared = N^2

    x = zeros(n)
    for i in 1:n
        up = sum(s_new[:, :, i] .== 1.0)
        # down = sum(s_new[:, :, i] .== 0.0)
        x[i] = up / nsquared
    end

    return s_new, x, neighbors_new
end

function simulate_dmm(theta, C, N, tsteps; p_s=0.88, p_ext=0.01)
    n = size(C, 1)
    s, x, neighbors = initialize_dmm_non(tsteps, N, n)

    for t in 2:tsteps
        s_new, x_new, neighbors = update_states(theta, s[:, :, t-1, :], x[t-1, :], neighbors, C; p_s=p_s, p_ext=p_ext)
        s[:, :, t, :] = s_new
        x[t, :] = x_new
    end
    return s, x
end

function simulate_dmm_1d(theta, N, tsteps; p_s=0.88, p_ext=0.01)
    s, x = simulate_dmm([theta], [0.0], N, tsteps; p_s=p_s, p_ext=p_ext)
    return s, x
end

function simulate_dmm_dynamic_theta(theta, C, N, tsteps; p_s=0.88, p_ext=0.01)
    """
    theta is a matrix: t x n
    """
    n = size(C, 1)
    s, x, neighbors = initialize_dmm_non(tsteps, N, n)

    for t in 2:tsteps
        s_new, x_new, neighbors = update_states(theta[t-1, :], s[:, :, t-1, :], x[t-1, :], neighbors, C; p_s=p_s, p_ext=p_ext)
        s[:, :, t, :] = s_new
        x[t, :] = x_new
    end
    return s, x
end

end # module dmm
