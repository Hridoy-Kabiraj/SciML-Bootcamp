using NeuralPDE
using ModelingToolkit
using Lux  # Use Lux for the neural network
using Optimisers  # Use Optimisers for the optimizer
using Plots
using DomainSets
using Optimization, OptimizationOptimJL, LineSearches

@parameters t x y z
@variables ϕ(..)  # Quantity of interest (e.g., neutron flux)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

# Define the diffusion coefficient (can be a function of x, y, z, t)
D(x, y, z, t) = 1.0 + 0.1 * sin(π * x) * sin(π * y) * sin(π * z) * exp(-t)

# Define the 3D diffusion equation with variable D and source term
eq = Dt(ϕ(t, x, y, z)) ~ D(x, y, z, t) * (Dxx(ϕ(t, x, y, z)) + Dyy(ϕ(t, x, y, z)) + Dzz(ϕ(t, x, y, z)))

# Initial and boundary conditions
bcs = [
    ϕ(0, x, y, z) ~ sin(π * x) * sin(π * y) * sin(π * z),  # Initial condition
    ϕ(t, 0, y, z) ~ 0.0,  # Boundary condition at x = 0
    ϕ(t, 1, y, z) ~ 0.0,  # Boundary condition at x = 1
    ϕ(t, x, 0, z) ~ 0.0,  # Boundary condition at y = 0
    ϕ(t, x, 1, z) ~ 0.0,  # Boundary condition at y = 1
    ϕ(t, x, y, 0) ~ 0.0,  # Boundary condition at z = 0
    ϕ(t, x, y, 1) ~ 0.0   # Boundary condition at z = 1
]

# Define the domain
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0),
    z ∈ Interval(0.0, 1.0)
]

# Define the PDE system
@named pde_system = PDESystem(eq, bcs, domains, [t, x, y, z], [ϕ(t, x, y, z)])

# Discretize the PDE using a neural network (Lux)
chain = Lux.Chain(
    Lux.Dense(4, 32, tanh),  # Input: (t, x, y, z), Output: 32 neurons
    Lux.Dense(32, 32, tanh), # Hidden layer
    Lux.Dense(32, 1)         # Output: ϕ(t, x, y, z)
)

# Discretization
discretization = PhysicsInformedNN(
    chain, QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6)
)

# Discretize the PDE system
prob = discretize(pde_system, discretization)

# Callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# Optimizer
opt = LBFGS(linesearch = BackTracking())
res = solve(prob, opt, callback = callback, maxiters = 50)
phi = discretization.phi

# Prepare grid points for evaluation
xs = range(0.0, 1.0, length=20)
ys = range(0.0, 1.0, length=20)
zs = range(0.0, 1.0, length=20)
ts = range(0.0, 1.0, length=20)



# Evaluate the solution at grid points
ϕ_predict = [first(phi([t, x, y, z], res.u)) for t in ts, x in xs, y in ys, z in zs]

# Plot the results (slice at a fixed z and t)
z_fixed = 0.5
t_fixed = 0.5

# Find the indices corresponding to z_fixed and t_fixed
z_index = argmin(abs.(zs .- z_fixed))  # Find the closest index to z_fixed
t_index = argmin(abs.(ts .- t_fixed))  # Find the closest index to t_fixed

# Extract the slice
ϕ_slice = ϕ_predict[t_index, :, :, z_index]

# Plot the heatmap
heatmap(xs, ys, ϕ_slice, xlabel="x", ylabel="y", title="ϕ(t=$t_fixed, x, y, z=$z_fixed)", color=:viridis)





using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays

# Define parameters and variables
@parameters t x y z
@variables ϕ(..)

# Define differential operators
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)

# Define the domain
tspan = (0.0, 1.0)
xspan = (0.0, 1.0)
yspan = (0.0, 1.0)
zspan = (0.0, 1.0)

# Initial condition
u0 = sin(π * x) * sin(π * y) * sin(π * z)  # Example initial condition

# Generate synthetic data
N = 20
t = range(tspan[1], tspan[2], length=N)
x = range(xspan[1], xspan[2], length=N)
y = range(yspan[1], yspan[2], length=N)
z = range(zspan[1], zspan[2], length=N)

# Define the neural networks for each term
rng = Random.default_rng()

# Neural network for the diffusion term
NN_diff = Lux.Chain(Lux.Dense(4, 16, relu), Lux.Dense(16, 1))
p_diff, st_diff = Lux.setup(rng, NN_diff)

# Neural network for the source term
NN_source = Lux.Chain(Lux.Dense(4, 16, relu), Lux.Dense(16, 1))
p_source, st_source = Lux.setup(rng, NN_source)

# Combine parameters into a single ComponentArray
p0_vec = (diff = p_diff, source = p_source)
p0_vec = ComponentArray(p0_vec)

# Define the UDE
function diffusion_ude!(du, u, p, t)
    ϕ = u[1]
    ∇²ϕ = Dx(Dx(ϕ)) + Dy(Dy(ϕ)) + Dz(Dz(ϕ))  # Laplacian

    # Neural network approximations
    diffusion_term = NN_diff([t, x, y, z], p.diff, st_diff)[1][1]
    source_term = NN_source([t, x, y, z], p.source, st_source)[1][1]

    du[1] = diffusion_term * ∇²ϕ + source_term
end

# Define the ODE problem
prob = ODEProblem(diffusion_ude!, [u0], tspan, p0_vec)

# Solve the ODE to generate synthetic data
sol = solve(prob, Tsit5(), saveat=t)
data = Array(sol)

# Define the prediction function
function predict_adjoint(θ)
    Array(solve(prob, Tsit5(), p=θ, saveat=t))
end

# Define the loss function
function loss_adjoint(θ)
    pred = predict_adjoint(θ)
    sum(abs2, data .- pred)
end

# Callback function to monitor training
iter = 0
function callback(θ, l)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Loss: ", l)
    end
    return false
end

# Optimize the neural network parameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)
res = Optimization.solve(optprob, ADAM(0.001), callback=callback, maxiters=1000)

# Visualize the results
pred = predict_adjoint(res.u)
plot(t, data[1, :], label="True Solution", linewidth=2)
plot!(t, pred[1, :], label="Predicted Solution", linestyle=:dash, linewidth=2)
xlabel!("Time (t)")
ylabel!("ϕ(t, x, y, z)")
title!("3D Diffusion Equation with UDE")