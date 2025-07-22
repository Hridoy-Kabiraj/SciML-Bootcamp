using OrdinaryDiffEq, MethodOfLines, ModelingToolkit, DomainSets
using Lux, DiffEqFlux, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays

# ==============================
# 1️⃣ PDE SOLUTION USING MOL
# ==============================

@parameters x y t
@variables ψ(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2
Dy = Differential(y)
Dyy = Differential(y)^2

xmin, xmax = 1.0, 4.0
ymin, ymax = -2.0, 2.0
tmin, tmax = 0.0, 12.0

v_ψ_value = -1.0

ψ0_func(x, y, t) = exp(-(((x-2)^2 + y^2) / 0.5^2))

eqs = [Dt(ψ(x, y, t)) + v_ψ_value * (Dx(ψ(x, y, t)) + Dy(ψ(x, y, t))) ~  
       -(Dx(ψ(x, y, t)) / x) + Dxx(ψ(x, y, t)) + Dyy(ψ(x, y, t))]

bcs = [ψ(x, y, 0) ~ ψ0_func(x, y, 0),
       Dx(ψ(xmin, y, t)) ~ 0,
       Dx(ψ(xmax, y, t)) ~ 0,
       Dy(ψ(x, ymin, t)) ~ 0,
       Dy(ψ(x, ymax, t)) ~ 0]

domains = [x ∈ Interval(xmin, xmax),
           y ∈ Interval(ymin, ymax),
           t ∈ Interval(tmin, tmax)]

@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [ψ(x, y, t)])

N = 32
order = 2
discretization = MOLFiniteDifference([x => N, y => N], t, approx_order=order)

println("Discretization:")
@time prob = discretize(pdesys, discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)

dis_x = sol[x]
dis_y = sol[y]
dis_t = sol[t]
dis_ψ = sol[ψ(x, y, t)]

# ==============================
# 2️⃣ DEFINE THE UDE
# ==============================

Random.seed!(1234)

# Define Neural Networks
NN1 = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))
NN2 = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))
NN3 = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))
NN4 = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))
NN5 = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))

p1, st1 = Lux.setup(Random.default_rng(), NN1)
p2, st2 = Lux.setup(Random.default_rng(), NN2)
p3, st3 = Lux.setup(Random.default_rng(), NN3)
p4, st4 = Lux.setup(Random.default_rng(), NN4)
p5, st5 = Lux.setup(Random.default_rng(), NN5)

p0_vec = (layer_1 = p1, layer_2 = p2, layer_3 = p3, layer_4 = p4, layer_5 = p5)
p0_vec = ComponentArray(p0_vec)
st_vec = (layer_1 = st1, layer_2 = st2, layer_3 = st3, layer_4 = st4, layer_5 = st5)

# Correct NN evaluations using Lux.apply
NNt(p, x, y, t, st) = abs(Lux.apply(NN1, [x, y, t], p.layer_1, st.layer_1)[1][1])
NNx(p, x, y, t, st) = abs(Lux.apply(NN2, [x, y, t], p.layer_2, st.layer_2)[1][1])
NNy(p, x, y, t, st) = abs(Lux.apply(NN3, [x, y, t], p.layer_3, st.layer_3)[1][1])
NNxx(p, x, y, t, st) = abs(Lux.apply(NN4, [x, y, t], p.layer_4, st.layer_4)[1][1])
NNyy(p, x, y, t, st) = abs(Lux.apply(NN5, [x, y, t], p.layer_5, st.layer_5)[1][1])

# Define UDE
function UDE!(du, u, p, t)
    x, y = u  # Extract spatial coordinates

    # Neural network approximations of the PDE terms
    dψ_dt = NNt(p, x, y, t, st_vec)
    dψ_dx = NNx(p, x, y, t, st_vec)
    dψ_dy = NNy(p, x, y, t, st_vec)
    dψ_dxx = NNxx(p, x, y, t, st_vec)
    dψ_dyy = NNyy(p, x, y, t, st_vec)

    # PDE equation reformulated as an ODE system
    du[1] = dψ_dt + v_ψ_value * (dψ_dx + dψ_dy) + (dψ_dx / x) - dψ_dxx - dψ_dyy
end

# Define UDE Problem
u0 = [2.0, 0.0]  # Initial condition (adjust based on the problem)
tspan = (0.0, 12.0)

prob_ude = ODEProblem(UDE!, u0, tspan, p0_vec)

# ==============================
# 3️⃣ TRAIN THE UDE
# ==============================

function loss(p)
    sol = solve(prob_ude, Tsit5(), p=p, saveat=0.1)
    return sum(abs2, sol.u .- dis_ψ)  # Compare with MOL solution
end

# Define optimizer
optf = Optimization.OptimizationFunction((p, _)->loss(p), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p0_vec)

println("Training UDE...")
@time res = Optimization.solve(optprob, Optim.BFGS())

# ==============================
# 4️⃣ VISUALIZATION
# ==============================

# Solve the trained UDE
sol_ude = solve(prob_ude, Tsit5(), p=res.u, saveat=0.1)

# Plot MOL vs UDE
p1 = heatmap(dis_ψ[:, :, 1], title="MOL Solution")
p2 = heatmap(sol_ude[:, :, 1], title="Trained UDE Solution")

display(plot(p1, p2, layout=(1,2)))