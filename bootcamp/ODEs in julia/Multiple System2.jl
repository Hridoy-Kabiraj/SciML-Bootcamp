using DifferentialEquations

# u = [x, y, z]
# u[1] = x
# u[2] = y
# u[3] = z
function parameterized_lorenz!(du, u, p, t)
    x, y, z = u
    σ, ρ, β = p
    du[1] = σ*(y-x)
    du[2] = x*(ρ-z)-y
    du[3] = x*y-β*z
end

u0 = [1.0, 0.0, 0.0]
p = [10.0, 28.0, 8/3]
tspan = (0.0, 100.0)

prob = ODEProblem(parameterized_lorenz!, u0, tspan, p)
sol = solve(prob)

using Plots
plot(sol, idxs = (1, 2, 3))
