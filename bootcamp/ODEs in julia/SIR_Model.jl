using DifferentialEquations

function SIR!(du, u, p, t)
    S, I, R = u 
    N, β, γ = p 
    du[1] = -(β*S*I)/N 
    du[2] = (β*S*I)/N - γ*I 
    du[3] = γ*I 
end 

u0 = [999.0, 1.0, 0.0]
p = [1000.0, 0.3, 0.1]
tspan = [0.0, 160.0]

prob = ODEProblem(SIR!, u0, tspan, p)

sol = solve(prob)

# Plot all three variables against time 
using Plots
plot(sol.t, hcat(sol.u...)', 
    label=["Susceptible (S)" "Infected (I)" "Recovered (R)"], 
    xlabel="Time (days)", 
    ylabel="Population",
    title="SIR Model Simulation",
    lw=2)  # lw = line width