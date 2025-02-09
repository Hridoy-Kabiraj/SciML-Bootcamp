using OrdinaryDiffEq, MethodOfLines, ModelingToolkit, DomainSets


# x = R , y = Z
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

#Assuming constant Plasma flow velocity
v_ψ_value = -1.0

ψ0_func(x, y, t) = exp(-(((x-2)^2 + y^2) / 0.5^2))

eqs = [Dt(ψ(x, y, t)) + v_ψ_value * (Dx(ψ(x, y, t)) + Dy(ψ(x, y, t))) ~  -(Dx(ψ(x, y, t)) / x) + Dxx(ψ(x, y, t)) + Dyy(ψ(x, y, t))]

# Inital and Boundary Conditions
bcs = [ψ(x, y, 0) ~ ψ0_func(x, y, 0),
       Dx(ψ(xmin, y, t)) ~ 0,
       Dx(ψ(xmax, y, t)) ~ 0,
       Dy(ψ(x, ymin, t)) ~ 0,
       Dy(ψ(x, ymax, t)) ~ 0]


# Domains
domains = [x ∈ Interval(xmin, xmax),
           y ∈ Interval(ymin, ymax),
           t ∈ Interval(tmin, tmax)]


@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [ψ(x, y, t)])


N = 64
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


using Plots

# Assuming dis_ψ is your 3D array with shape (32, 32, 121)
# You can plot the first frame (time step) as an example

# Extract the first time step
frame = dis_ψ[:, :, 1]

# Create a surface plot
plot_surface = surface(frame)
display(plot_surface)
