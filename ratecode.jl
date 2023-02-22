using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics


mutable struct neuron
    Act::Float64            
    Ge::Float64
    Gl::Float64
    GbarE::Float64
    GbarL::Float64
    ErevE::Float64
    ErevL::Float64
    Vm::Float64
    ΔVm::Float64
    #=GKNA::Float64   #for KNA adapt
    Rise::Float64   #for KNA adapt
    Max::Float64    #for KNA adapt
    Tau::Float64    #for KNA adapt =#

    function neuron(GbarE, GbarL, ErevE, ErevL, Vm) 
        Act = 0
        Ge = 0
        Gl = GbarL
        ΔVm = 0
        
        return new(Act, Ge, Gl, GbarE, GbarL, ErevE, ErevL, Vm, ΔVm)
    end
end

GbarE = 0.3; GbarL = 0.3; ErevE = 1; ErevL = 0.3; Vm = 0.3

n = neuron(GbarE, GbarL,ErevE,ErevL,Vm)

VmPlot = Vector{Float64}(undef, 200)
ΔVmPlot = Vector{Float64}(undef, 200) 

for cycle in 1:200
    if cycle > 20 && cycle < 180
        n.Ge = 0.3
    else
        n.Ge = 0
    end

    n.ΔVm = n.Ge*(n.ErevE - n.Vm) + n.Gl*(n.ErevL - n.Vm)
    n.Vm += n.ΔVm
    #n.Act += ΔVm * (activation_function() - n.Act)

    VmPlot[cycle] = n.Vm
    ΔVmPlot[cycle] = n.ΔVm

end

fig = plot(VmPlot)
fig = plot!(ΔVmPlot)

getproperty.(timeseries, :Vm)