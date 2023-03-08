using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics


mutable struct neuron
    Act::Float64            
    Ge::Float64
    Gl::Float64
    GbarE::Float64
    GbarL::Float64
    ErevE::Float64
    ErevL::Float64
    theta::Float64
    Vm::Float64
    ΔVm::Float64
    GKNA::Float64   #for KNA adapt
    Rise::Float64   #for KNA adapt
    Max::Float64    #for KNA adapt
    Tau::Float64    #for KNA adapt 

    function neuron(GbarE, GbarL, ErevE, ErevL, Vm, Tau, Rise, Max) 
        Act = 0
        Ge = 0
        Gl = GbarL
        ΔVm = 0
        theta = 0.5
        GKNA = 0
        
        return new(Act, Ge, Gl, GbarE, GbarL, ErevE, ErevL, theta, Vm, ΔVm, GKNA , Tau, Rise, Max)
    end
end

GbarE = 0.3; GbarL = 0.3; ErevE = 1; ErevL = 0.3; Vm = 0.3
Tau = 1000; Rise = 0.001; Max = 1 #KNa Adapt
spiking = true
KNaAdapt = true

n = neuron(GbarE, GbarL,ErevE, ErevL, Vm, Tau, Rise, Max)

VmPlot = Vector{Float64}(undef, 200)
ΔVmPlot = Vector{Float64}(undef, 200)
GKNAPlot = Vector{Float64}(undef, 200) 

for cycle in 1:200
    if cycle > 20 && cycle < 180
        n.Ge = 0.3
    else
        n.Ge = 0
    end
    
    n.ΔVm = n.Ge*(n.ErevE - n.Vm) + n.Gl*(n.ErevL - n.Vm) # update current
    n.Vm += n.ΔVm / 10 # change membrane potential based on current
    #n.Act += ΔVm * (activation_function() - n.Act)

    if n.Vm > n.theta && spiking # spiking
        n.Vm = 0.3
        if KNaAdapt
            n.GKNA += (n.Rise * (n.Max - n.GKNA)) / 10
        end
    elseif KNaAdapt
        n.GKNA -= (1/n.Tau * n.GKNA) / 10
    end

    VmPlot[cycle] = n.Vm
    ΔVmPlot[cycle] = n.ΔVm
    GKNAPlot[cycle] = n.GKNA

end

fig = plot(VmPlot)
fig = plot!(ΔVmPlot)
fig = plot!(GKNAPlot)
fig = ylims!(0,1)
