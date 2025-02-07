using DifferentialEquations

# u = [θ, ω]
# u[1] = θ
# u[2] = ω

function pendulum!(du, u, p, t)
    θ, ω = u
    g, l, m, M = p
    du[1] = ω
    du[2] = -(3/2)*(g/l)*sin(θ) + (3/(m*l*l))*M(t)
end

u0 = [0.01, 0.0]
p = [9.81, 1.0, 1.0,  t -> 0.1*sin(t)]
tspan = (0.0, 10.0)

prob = ODEProblem(pendulum!, u0, tspan, p)
sol = solve(prob)

using Plots
plot(sol, linewidth = 2, xaxis = "t",
    label = ["θ[rad]" "ω[rad/s]"], layout = (2,1))