using Interpolations
using LinearAlgebra
using Plots
using DifferentialEquations
function print_jl(x,flag_val = false)
    println("Type is $(typeof(x))")
    println("Shape is $(size(x))")
    if flag_val == true
        println("Value is $(x)")
    end
end

function plot_ellipse(plot,Q::Matrix,xbar::Vector,color;alpha=0.3,label=nothing)
    θ = range(0, 2pi + 0.05; step = 0.05)
    x_y = √Q[1:2,1:2] * hcat(cos.(θ), sin.(θ))' .+ xbar[1:2]
    plot!(plot, x_y[1, :], x_y[2, :], c = color,linewidth=2,label=label)
    plot!(plot, x_y[1, :], x_y[2, :], label = nothing,fill=true, fillcolor=color,alpha=alpha)
    plot!(legendfontsize=12)
end

include("dynamics.jl")
function matrix_to_vector(matrix::Array)
    return [vec(col) for col in eachcol(matrix)]
end

function propagate_multiple_FOH(model::Dynamics,x::Matrix,u::Matrix,T::Vector)
    N = size(x,2) - 1
    ix = size(x,1)
    iu = size(u,1)

    function model_wrapper!(f,x,p,t)
        um = p[1]
        up = p[2]
        dt = p[3]
        alpha = 1 - t
        beta = t
        u1 = alpha*um + beta*up
        f .= dt*forward(model,x,u1)
    end

    tspan = (0,1)
    tprop = []
    xprop = []
    xnew = zeros(size(x))
    xnew[:,1] .= x[:,1]
    for i in 1:N
        prob = ODEProblem(model_wrapper!,x[:,i],tspan,(u[:,i],u[:,i+1],T[i]))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);
        tode = sol.t
        xode = stack(sol.u)
        if i == 1
            tprop = T[i]*tode
            xprop = xode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+T[i]*tode)
            xprop = hcat(xprop,xode)
        end
        xnew[:,i+1] .= xode[:,end]
    end
    return xnew,tprop,xprop
end

using JuMP
function Lipschitz_estimation_around_traj(N,num_sample,xnom,unom,dynamics,Qnode,Rnode)
    gamma_sample = zeros(num_sample,N+1)
    for idx in 1:N+1
        for j in 1:num_sample
            sqrt_Q = sqrt(Qnode[:,:,idx])
            sqrt_R = sqrt(Rnode[idx])

            z = randn(ix)
            z .= z / norm(z)
            eta_sample = sqrt_Q * z

            z = randn(iu)
            z .= z / norm(z)
            xii_sample = sqrt_R * z

            K = Knode[:,:,idx]
            x_ = xnom[:,idx] + eta_sample
            u_ = unom[:,idx] + xii_sample

            A,B = diff(dynamics,xnom[:,idx],unom[:,idx])

            eta_dot = forward(dynamics,x_,u_) - forward(dynamics,xnom[:,idx],unom[:,idx])
            LHS = eta_dot - A * eta_sample - B * xii_sample
            delta_q = dynamics.Cv * eta_sample +  dynamics.Dvu * xii_sample

            model = Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", false) # Turn off verbosity for Mosek
            @variable(model, Delta[1:dynamics.iϕ,1:dynamics.iv])
            @constraint(model,LHS == dynamics.G * Delta * delta_q)
            @objective(model,Min,dot(vec(Delta),vec(Delta)))
            optimize!(model)

            gamma_sample[j,idx] = opnorm(value.(Delta),2)
        end
    end
    return maximum(gamma_sample,dims=1)[:]
end