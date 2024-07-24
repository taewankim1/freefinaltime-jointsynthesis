
using LinearAlgebra
using JuMP

abstract type FunnelConstraint end

struct StateConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function StateConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

struct InputConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function InputConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

function impose!(constraint::StateConstraint,model::Model,Q::Matrix,K::Matrix,Qbar::Matrix,Kbar::Matrix,xnom::Vector,unom::Vector,idx::Int)
    a = constraint.a
    b = constraint.b

    @constraint(model, a'*Q*a <= (b-a'*xnom)^2)
end

function impose!(constraint::InputConstraint,model::Model,Q::Matrix,K::Matrix,Qbar::Matrix,Kbar::Matrix,xnom::Vector,unom::Vector,idx::Int)
    a = constraint.a
    b = constraint.b

    q = vec(Q)
    qbar = vec(Qbar)
    k = vec(K)
    kbar = vec(Kbar)

    # fk = 2.0 * kbar' * kron(Qbar, a*a')
    kron_ = kron(Qbar,a*a')
    fk = kbar' * kron_ + (kron_ * kbar)'
    aKbar = a'*Kbar
    fq = kron(aKbar,aKbar)
    # f = a' * Kbar * Qbar * Kbar' * a
    f = aKbar * Qbar * aKbar'

    @constraint(model, f + fq * (q - qbar) + fk * (k - kbar) <= (b-a'*unom)^2)
end

struct ObstacleAvoidance <: FunnelConstraint
    H::Matrix
    c::Vector
    function ObstacleAvoidance(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose!(constraint::ObstacleAvoidance,model::Model,Q::Matrix,K::Matrix,Qbar::Matrix,Kbar::Matrix,xnom::Vector,unom::Vector,idx::Int)
    H = constraint.H
    c = constraint.c
    M = [1 0 0;0 1 0]
    a = - M'*H'*H*(M*xnom-c) / norm(H*(M*xnom-c))
    s = 1 - norm(H*(M*xnom-c))
    b = -s + a'*xnom

    @constraint(model, a'*Q*a <= (b-a'*xnom)^2)
end

struct WayPoint <: FunnelConstraint
    Qpos_max::Matrix{Float64}
    function WayPoint(Qmax::Matrix)
        new(Qmax)
    end
end

function impose!(constraint::WayPoint,model::Model,Q::Matrix,K::Matrix,Qbar::Matrix,Kbar::Matrix,xnom::Vector,unom::Vector,idx::Int)
    # hard coding
    # 1,4,7,10,13,16
    if (idx == 4) || (idx == 7) || (idx == 10) || (idx == 13)
        @constraint(model, Q[1:3,1:3] <= constraint.Qpos_max, PSDCone())
    end
end
