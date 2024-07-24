
include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
include("funl_dynamics.jl")
using LinearAlgebra


function diff_numeric(model::FunnelDynamics,x::Vector,u::Vector,A::Matrix,B::Matrix)
    ix = length(x)
    iu = length(u)
    eps_x = Diagonal{Float64}(I, ix)
    eps_u = Diagonal{Float64}(I, iu)
    fx = zeros(ix,ix)
    fu = zeros(ix,iu)
    h = 2^(-18)
    for i in 1:ix
        fx[:,i] = (forward(model,x+h*eps_x[:,i],u,A,B) - forward(model,x-h*eps_x[:,i],u,A,B)) / (2*h)
    end
    for i in 1:iu
        fu[:,i] = (forward(model,x,u+h*eps_u[:,i],A,B) - forward(model,x,u-h*eps_u[:,i],A,B)) / (2*h)
    end
    return fx,fu
end

function get_radius_angle_Ellipse2D(Q_list)
    radius_list = []
    angle_list = []

    for i in 1:size(Q_list,3)
        Q_ = Q_list[:,:,i]
        # eigval = eigvals(inv(Q_))
        # radius = sqrt.(1 ./ eigval)
        # println("radius of x,y,theta: ", radius)
        A = [1 0 0; 0 1 0]
        Q_proj = A * Q_ * A'
        Q_inv = inv(Q_proj)
        eigval, eigvec = eigen(Q_inv)
        radius = sqrt.(1 ./ eigval)
        # println("radius of x and y: ", radius)
        rnew = eigvec * [radius[1]; 0]
        angle = atan(rnew[2], rnew[1])
        push!(radius_list, radius)
        push!(angle_list, angle)
    end
    return radius_list, angle_list
end

function propagate_multiple_FOH(model::FunnelDynamics,dynamics::Dynamics,
    x::Matrix,u::Matrix,T::Vector,
    X::Matrix,U::Matrix;
    flag_single::Bool=false)
    N = size(x,2) - 1
    ix = model.ix
    iu = model.iu
    iX = model.iX
    iU = model.iU

    idx_x = 1:ix
    idx_X = (ix+1):(ix+iX)
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        Um = p[3]
        Up = p[4]
        dt = p[5]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        U_ = alpha * Um + beta * Up

        x_ = V[idx_x]
        X_ = V[idx_X]

        # traj terms
        f = forward(dynamics,x_,u_)
        fx,fu = diff(dynamics,x_,u_)

        # funl terms
        F = forward(model,X_,U_,fx,fu)

        dV = [f;F]
        out .= dV[:]
    end

    tprop = []
    xprop = []
    uprop = []
    Xprop = []
    Uprop = []
    Xfwd = zeros(size(X))
    Xfwd[:,1] .= X[:,1]
    for i = 1:N
        if flag_single == true
            V0 = [x[:,i];Xfwd[:,i]][:]
        else
            V0 = [x[:,i];X[:,i]][:]
        end

        um = u[:,i]
        up = u[:,i+1]
        Um = U[:,i]
        Up = U[:,i+1]
        dt = T[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,Um,Up,dt))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);

        tode = sol.t[1:end-1]
        uode = zeros(iu,size(tode,1))
        Uode = zeros(iU,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt
            uode[:,idx] .= alpha * um + beta * up
            Uode[:,idx] .= alpha * Um + beta * Up
        end
        ode = stack(sol.u)
        xode = ode[idx_x,1:end-1]
        Xode = ode[idx_X,1:end-1]
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
            Xprop = Xode
            Uprop = Uode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
            Xprop = hcat(Xprop,Xode)
            Uprop = hcat(Uprop,Uode)
        end
        Xfwd[:,i+1] = ode[idx_X,end]
    end
    tprop = vcat(tprop,[sum(T[1:N])])
    xprop = hcat(xprop,x[:,end])
    uprop = hcat(uprop,u[:,end])
    Xprop = hcat(Xprop,X[:,end])
    Uprop = hcat(Uprop,U[:,end])
    return Xfwd,tprop,xprop,uprop,Xprop,Uprop
end

function propagate_from_funnel_entry(x0::Vector,model::FunnelDynamics,dynamics::Dynamics,
    xnom::Matrix,unom::Matrix,Tnom::Vector,
    K::Array{Float64,3})
    N = size(xnom,2) - 1
    ix = model.ix
    iu = model.iu

    idx_x = 1:ix
    idx_xnom = ix+1:2*ix
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        km = p[3]
        kp = p[4]
        dt = p[5]

        alpha = (dt - t) / dt
        beta = t / dt

        unom_ = alpha * um + beta * up
        k_ = alpha * km + beta * kp
        K_ = reshape(k_,(iu,ix))

        x_ = V[idx_x]
        xnom_ = V[idx_xnom]

        u_ = unom_ + K_ * (x_ - xnom_)

        # traj terms
        f = forward(dynamics,x_,u_)
        fnom = forward(dynamics,xnom_,unom_)

        dV = [f;fnom]
        out .= dV[:]
    end

    xfwd = zeros(size(xnom))
    xfwd[:,1] .= x0
    tprop = []
    xprop = []
    xnom_prop = []
    uprop = []
    for i = 1:N
        V0 = [xfwd[:,i];xnom[:,i]][:]
        um = unom[:,i]
        up = unom[:,i+1]
        km = vec(K[:,:,i])
        kp = vec(K[:,:,i+1])
        dt = Tnom[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,km,kp,dt))
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12;verbose=false);

        ode = stack(sol.u)
        if i != N
            tode = sol.t[1:end-1]
            xode = ode[idx_x,1:end-1]
            xnomode = ode[idx_xnom,1:end-1]
        else
            tode = sol.t
            xode = ode[idx_x,:]
            xnomode = ode[idx_xnom,:]
        end
        uode = zeros(iu,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt

            unom_ = alpha * um + beta * up
            k_ = alpha * km + beta * kp
            x_ = xode[:,idx]
            xnom_ = xnomode[:,idx]
            K_ = reshape(k_,(iu,ix))
            uode[:,idx] .= unom_ + K_ * (x_ - xnom_)
        end
        if i == 1
            tprop = tode
            xprop = xode
            xnom_prop = xnomode
            uprop = uode
        else 
            tprop = vcat(tprop,sum(Tnom[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            xnom_prop = hcat(xnom_prop,xnomode)
            uprop = hcat(uprop,uode)
        end
        xfwd[:,i+1] = ode[idx_x,end]
    end
    return xfwd,tprop,xprop,uprop,xnom_prop
end

function XU_to_QKZS(X::Matrix{Float64},U::Matrix{Float64},ix::Int,iu::Int)
    N = size(X,2)
    Q = reshape(X,(ix,ix,N))
    K = reshape(U[1:ix*iu,:],(iu,ix,N))
    Z = reshape(U[ix*iu+1:ix*iu+ix*ix,:],(ix,ix,N))
    S = U[ix*iu+ix*ix+1:end,:]
    return Q,K,Z,S
end

function QKZS_to_XU(Q::Array{Float64,3},K::Array{Float64,3},Z::Array{Float64,3},S::Matrix{Float64})
    N = size(Q,3)
    ix = size(Q,1)
    iu = size(K,1)
    q = reshape(Q,(ix*ix,N))
    k = reshape(K,(iu*ix,N))
    z = reshape(Z,(ix*ix,N))
    return q,vcat(k,z,S)
end