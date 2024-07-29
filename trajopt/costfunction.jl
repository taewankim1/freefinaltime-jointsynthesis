using LinearAlgebra
include("dynamics.jl")


function get_cost(dynamics::Unicycle,x::Vector,u::Vector,idx::Int,N::Int)
    return dot(u,u)
end

function get_cost(dynamics::ThreeDOFManipulatorDynamics,x::Vector,u::Vector,idx::Int,N::Int)
    return dot(u,u)
end

function get_cost(dynamics::QuadrotorDynamics,x::Vector,u::Vector,idx::Int,N::Int)
    return dot(u,u)
end