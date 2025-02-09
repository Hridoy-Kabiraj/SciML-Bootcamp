using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

@parameters t x 
@variables ψ(..)

Dt = Differential(t)
Dxx = Differential(x)^2

V(x) = 0.0
xmin=0
xmax=1

eqs = [im*Dt(ψ(t, x)) ~ Dxx(ψ(t, x)) + V(x)*ψ(t, x)]

ψ0 = x -> ((1+im)/sqrt(2))*sin(2*π*x)

bcs = [ψ(0, x) => ψ0(x),
       ψ(t, xmin) ~ 0,
       ψ(t, xmax) ~ 0]

domains = [t ∈ Interval(0, 1),
           x ∈ Interval(xmin, xmax)]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [ψ(t, x)])

disc = MOLFiniteDifference([x=>100], t)

prob = discretize(pdesys, disc)

sol = solve(prob, TRBDF2(), saveat=0.01)

disc_x = sol[x]
disc_t = sol[t]
disc_ψ = sol[ψ(t, x)]

using Plots
anim = @animate for i in 1:length(disc_t)
    u = disc_ψ[i,:]
    plot(disc_x, [real.(u), imag.(u)], ylim=(-1.5,1.5), title="t = $(disc_t[i])", xlabel="x", ylabel="ψ(t,x)", label=["re(ψ)" "im(ψ)"], legend=:topleft)
end
gif(anim, "schroedinger.gif", fps = 10)