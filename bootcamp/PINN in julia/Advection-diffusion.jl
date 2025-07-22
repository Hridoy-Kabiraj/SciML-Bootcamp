using NeuralPDE, Lux, Optimization, OptimizationOptimJL, LineSearches, Plots
using ModelingToolkit: Interval
using Lux 

# Define parameters and variables
@parameters x y t
@variables C(..)

# Define differential operators
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
∇²(C) = Dx(Dx(C)) + Dy(Dy(C))  # Laplacian

# Define constants and functions
v = [1.0, 0.5]  # Velocity vector (example)
D = 0.1         # Diffusion coefficient (example)
R(C) = -0.01 * C # Reaction term (example)
P(x, y, t) = exp(-((x-0.5)^2 + (y-0.5)^2)) * sin(t)  # Source term (example)

# Define the PDE
eq = Dt(C(x, y, t)) + v[1] * Dx(C(x, y, t)) + v[2] * Dy(C(x, y, t)) ~ D * ∇²(C(x, y, t)) + R(C(x, y, t)) + P(x, y, t)

# Boundary and initial conditions
bcs = [
    C(x, y, 0) ~ exp(-((x-0.3)^2 + (y-0.3)^2)),  # Initial condition (Gaussian)
    C(0, y, t) ~ 0.0,  # Dirichlet boundary condition
    C(1, y, t) ~ 0.0,
    C(x, 0, t) ~ 0.0,
    C(x, 1, t) ~ 0.0
]

# Space and time domains
domains = [
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0),
    t ∈ Interval(0.0, 10.0)
]

# Neural network architecture
dim = 3  # Number of dimensions (x, y, t)
chain = Lux.Chain(Lux.Dense(dim, 16, σ), Lux.Dense(16, 16, σ), Lux.Dense(16, 1))

# Discretization using PINNs
discretization = PhysicsInformedNN(
    chain, QuadratureTraining(; batch=200, abstol=1e-6, reltol=1e-6)
)

@named pde_system = PDESystem([eq], bcs, domains, [x, y, t], [C(x, y, t)])
prob = discretize(pde_system, discretization)

# Callback function to monitor training
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# Optimizer
opt = LBFGS(linesearch=BackTracking())
res = solve(prob, opt, callback=callback, maxiters=50)
phi = discretization.phi

# Generate grid for evaluation
dx = 0.05
xs = range(0.0, 10.0, step=dx)
ys = range(0.0, 10.0, step=dx)
ts = range(0.0, 10.0, step=0.001)

# Evaluate the neural network solution
u_predict = [first(phi([x, y, t], res.u)) for x in xs for y in ys for t in ts]
u_predict = reshape(u_predict, (length(xs), length(ys), length(ts)))

# Plot the solution at a specific time step
t_idx = 1  # Time index to plot
p1 = plot(xs, ys, u_predict[:, :, t_idx], linetype=:contourf, title="Predicted C(x, y, t) at t=$(ts[t_idx])")
plot(p1) 


# Animate the solution
anim = @animate for k in 1:length(ts)
    heatmap(xs, ys, u_predict[:, :, k], title="C(x, y, t) at t=$(ts[k])", xlabel="x", ylabel="y", color=:viridis)
end
gif(anim, "AdvectionDiffusionReaction.gif", fps=8)