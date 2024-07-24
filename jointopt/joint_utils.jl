include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
include("joint_dynamics.jl")
using LinearAlgebra

function propagate_multiple_FOH(model::JointDynamics,dynamics::Dynamics,
    x::Matrix,u::Matrix,T::Vector,
    Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    N = size(x,2) - 1
    ix = model.ix
    iu = model.iu
    iq = model.iq
    iy = model.iy

    idx_x = 1:ix
    idx_q = (ix+1):(ix+iq)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        dt = p[7]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp

        x_ = V[idx_x]
        q_ = V[idx_q]

        # traj terms
        f = forward(dynamics,x_,u_)
        # funl terms
        F = forward(model,dynamics,q_,y_,z_,x_,u_)

        dxdt = f
        dqdt = F
        dV = [dxdt;dqdt]
        out .= dV[:]
    end

    tprop = []
    xprop = []
    uprop = []
    qprop = []
    yprop = []
    xfwd = zeros(size(x))
    xfwd[:,1] .= x[:,1]
    Qfwd = zeros(size(Q))
    Qfwd[:,:,1] = Q[:,:,1]
    for i = 1:N
        V0 = [x[:,i];vec(Q[:,:,i])][:]

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[:,:,i])
        zp = vec(Z[:,:,i+1])
        dt = T[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt))
        sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8;verbose=false);

        tode = sol.t
        uode = zeros(iu,size(tode,1))
        yode = zeros(iy,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt
            uode[:,idx] .= alpha * um + beta * up
            yode[:,idx] .= alpha * ym + beta * yp
        end
        ode = stack(sol.u)
        xode = ode[idx_x,:]
        qode = ode[idx_q,:]
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
            qprop = qode
            yprop = yode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
            qprop = hcat(qprop,qode)
            yprop = hcat(yprop,yode)
        end
        xfwd[:,i+1] .= xode[:,end]
        Qfwd[:,:,i+1] = reshape(qode[:,end],(ix,ix))
    end
    Qprop = zeros(ix,ix,length(tprop))
    Yprop = zeros(iu,Int64(iy/iu),length(tprop))
    for i in 1:length(tprop)
        Qprop[:,:,i] .= reshape(qprop[:,i],(ix,ix))
        Yprop[:,:,i] .= reshape(yprop[:,i],(iu,Int64(iy/iu)))
    end
    return xfwd,Qfwd,tprop,xprop,uprop,Qprop,Yprop
end