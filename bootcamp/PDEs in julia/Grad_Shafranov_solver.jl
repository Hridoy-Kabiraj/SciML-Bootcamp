using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

@parameters r z t
@variables ψ(..)

Dt = Differential(t)
Dr = Differential(r)
Drr = Differential(r)^2
Dz = Differential(z)
Dzz = Differential(z)^2

∇(ψ) = Dr(ψ) + Dz(ψ)
∇²(ψ) = Drr(ψ) + Dzz(ψ)


#= η(R,Z)= C / (Te)^(3/2)
function η(R, Z)
    Te = 1000.0 - 500.0 * exp(-((R - 1.0)^2 + Z^2) / 0.1)  # Example temperature profile
    C = 1e-4  # Resistivity constant
    return C / Te^(3/2)  # Spitzer resistivity
end
=#


#= J(R,z) = - (1/μ0*R)*(∇^2(ψ))
function J(R, Z, ψ(R, Z, t))
    μ0 = 4 * π * 1e-7  # Permeability of free space
    return -(1 / (μ0 * R)) * (∇²(ψ(R, Z, t)))
end
=#


#= v_ψ = -η*J
function v_ψ(R, Z, ψ(R, z, t))
    return -η(R, Z) * J(R, Z, ψ(R, Z, t))
end
=#
Rmin = 1.0
Rmax = 4.0
Zmin = -1.0
Zmax = 1.0
tmin = 0.0
tmax = 12
v_ψ_value = -1.0

eqs = Dt(ψ(r, z, t)) + v_ψ_value*∇(ψ(r, z, t)) ~ r * Dr((1/r)*Dr(ψ(r, z, t))) + Dzz(ψ(r, z, t))

# R0 = 2m, Z0 = 0m, σ = 0.5m
ψ0_func(r, z) = exp(-((((r-2)^2)+z^2)/(0.5)^2))



bcs = [ψ(r,z,0) ~ ψ0_func(r,z),
       DR(ψ(Rmin, z, t)) ~ 0,
       DR(ψ(Rmax, z, t)) ~ 0,
       DZ(ψ(r, Zmin, t)) ~ 0,
       DZ(ψ(r, Zmax, t)) ~ 0]


domains = [r ∈ Interval(Rmin, Rmax),
           z ∈ Interval(Zmin, Zmax),
           t ∈ Interval(tmin, tmax)]

@named pdesys = PDESystem(eqs, bcs, domains,[r, z, t], [ψ(r, z, t)])

N = 32
order = 2
discretization = MOLFiniteDifference([r=>N, z=>N], t, approx_order=order)

println("Discretization:")
@time prob = discretize(pdesys, discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)



