using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

 rng = Random.default_rng()
u0 = [999,1,0]
p = [1000,0.3,0.1]
tspan = [0,160]
datasize = 160
tsteps = range(tspan[1], tspan[2]; length = datasize)

function SIR(du, u, p, t)
    S,I,R = u
    N,β,γ = p
    du[1] = -(β*S*I)/N 
    du[2] = ((β*S*I))/N - γ*I 
    du[3] = γ*I 
end

prob_true = ODEProblem(SIR, u0, tspan, p)
true_data = Array(solve(prob_true, Tsit5(); saveat = tsteps))


dudt2 = Chain(Dense(3, 60, softplus), Dense(60, 3))
p, st = Lux.setup(rng, dudt2)
prob_neural = NeuralODE(dudt2, tspan, Rosenbrock23(); saveat = tsteps)


function predict_neuralode(p)
    Array(prob_neural(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, true_data .- pred)
    return loss
end

function callback(state, l; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps, true_data[1, :], label = "Data")
        scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback((; u = pinit), loss_neuralode(pinit); doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.07);callback = callback, maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2, Optim.BFGS(; initial_stepnorm = 0.01); callback, allow_f_increases = false)

callback((; u = result_neuralode2.u), loss_neuralode(result_neuralode2.u); doplot = true)