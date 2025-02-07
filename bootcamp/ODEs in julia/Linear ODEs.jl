using DifferentialEquations
using Plots

# in the privious problems we were getting a vector for u
# now we will find a matrix result for u


A = [1.0 0 0 -5
     4 -2 4 -3
     -4 0 0 1
     5 -2 2 3] # 4*4 matrix

f(u, p, t) = A * u

u0 = rand(4, 2) # 4*2 matrix
tspan = (0.0, 1.0)

prob = ODEProblem(f, u0, tspan)
sol = solve(prob)
plot(sol, xaxis = "t")