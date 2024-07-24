include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")

using LinearAlgebra
using BlockDiagonals
using SparseArrays

abstract type FunnelDynamics end

function create_block_diagonal(right::Matrix, n::Int)
    blocks = [right for _ in 1:n]
    return BlockDiagonal(blocks)
end

function com_mat(m, n)
    A = reshape(1:m*n, n, m)  # Note the swapped dimensions for Julia
    v = reshape(A', :)

    P = Matrix{Int}(I, m*n, m*n)  # Create identity matrix
    P = P[v, :]
    return P'
end

struct NonlinearDLMI <: FunnelDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    ik::Int
    is::Int

    iX::Int
    iU::Int

    C::Matrix
    D::Matrix

    Cn::Matrix # commutation matrix
    Cm::Matrix # commutation matrix
    function NonlinearDLMI(alpha,ix,iu,C,D)
        Cn = com_mat(ix,ix)
        Cm = com_mat(iu,ix)
        is = 1
        iX = ix*ix
        iU = iu*ix+ix*ix+is
        new(alpha,ix,iu,ix*ix,ix*iu,is,iX,iU,C,D,Cn,Cm)
    end
end

function forward(model::NonlinearDLMI, X::Vector, U::Vector, A::Matrix, B::Matrix)
    q = X[1:model.iq]
    k = U[1:model.ik]
    z = U[model.ik+1:model.ik+model.iq]
    lam = U[model.ik+model.iq+1]

    Q = reshape(q,(model.ix,model.ix))
    K = reshape(k,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))

   
    L = (A+B*K)*Q + 0.5*model.alpha*Q
    M = (model.C+model.D*K)*Q
    dQ = L + L' + lam*M'*M + Z
    return vec(dQ)
end

function diff(model::NonlinearDLMI,X::Vector,U::Vector,A::Matrix,B::Matrix)
    ix = model.ix
    iX = length(X)
    iU = length(U)
    is = model.is

    q = X[1:model.iq]
    k = U[1:model.ik]
    z = U[model.ik+1:model.ik+model.iq]
    lam = U[model.ik+model.iq+1]

    Q = reshape(q,(model.ix,model.ix))
    K = reshape(k,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))

    C_cl = model.C+model.D*K

    Imat = sparse(Matrix(1.0I,ix,ix))
    Cn = sparse(model.Cn)
    Cm = sparse(model.Cm)

    right = A + 0.5*model.alpha.*Imat
    kron_ = create_block_diagonal(right,ix)
    Aq = kron_ + Cn * kron_

    right = B*K
    kron_ = create_block_diagonal(right,ix)
    Aq += kron(Imat, right) + Cn * kron_

    right = lam*Q'*C_cl'*C_cl
    kron_ = create_block_diagonal(right,ix)
    Aq += kron(Imat, right) + Cn * kron_

    kron_ = kron(Q',B)
    FK = kron_ + Cn * kron_

    right = lam*Q'*model.C'*model.D
    kron_ = kron(Q',right)
    FK += kron_ + Cn * kron_

    right = lam*Q'*K'*model.D'*model.D
    kron_ = kron(Q',right)
    FK += kron_ + Cn * kron_

    FZ = kron(Imat,Imat)
    # Fz = Matrix(1.0I,ix*ix,ix*ix)

    Flam = zeros(iX,is)
    Flam[:,1] .= vec(Q'*C_cl'*C_cl*Q)
    # Flam[:,2] .= 
    # Sq = kron(Imat,Imat)
    return Aq,hcat(FK,FZ,Flam)
end

function get_interval(start,size)
    return start:(start+size-1)
end

function discretize_foh(model::FunnelDynamics,dynamics::Dynamics,
        x::Matrix,u::Matrix,T::Vector,
        X::Matrix,U::Matrix)
    @assert size(x,2) == size(X,2)
    @assert size(x,2) + 1 == size(u,2)
    @assert size(X,2) + 1 == size(U,2)

    N = size(x,2)
    ix = model.ix
    iX = size(X,1)
    iU = size(U,1)

    idx_x = get_interval(1,ix)
    idx_X = get_interval(idx_x[end]+1,iX)
    idx_A = get_interval(idx_X[end]+1,iX*iX)
    idx_Bm = get_interval(idx_A[end]+1,iX*iU)
    idx_Bp = get_interval(idx_Bm[end]+1,iX*iU)
    idx_s = get_interval(idx_Bp[end]+1,iX)
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        Um = p[3]
        Up = p[4]
        dt = p[5]

        alpha = 1 - t
        beta = t

        u_ = alpha * um + beta * up
        U_ = alpha * Um + beta * Up

        x_ = V[idx_x]
        X_ = V[idx_X]
        Phi = reshape(V[idx_A], (iX,iX))
        Bm_ = reshape(V[idx_Bm],(iX,iU))
        Bp_ = reshape(V[idx_Bp],(iX,iU))
        s_ = reshape(V[idx_s],(iX,1))

        # traj terms
        f = forward(dynamics,x_,u_)
        fx,fu = diff(dynamics,x_,u_)

        # funl terms
        F = forward(model,X_,U_,fx,fu)
        FX_,FU_ = diff(model,X_,U_,fx,fu)
        FX_ .= FX_ * dt
        FU_ .= FU_ * dt
        dA = FX_*Phi
        dBm = FX_*Bm_ + FU_*alpha
        dBp = FX_*Bp_ + FU_*beta
        ds = FX_ * s_ + F
        dV = [f*dt;F*dt;dA[:];dBm[:];dBp[:];ds[:]]
        out .= dV[:]
    end

    A = zeros(iX,iX,N)
    Bm = zeros(iX,iU,N)
    Bp = zeros(iX,iU,N)
    s = zeros(iX,N)
    z = zeros(iX,N)
    x_prop = zeros(ix,N)
    X_prop = zeros(iX,N)
    for i = 1:N
        A0 = Matrix{Float64}(I,iX,iX)
        Bm0 = zeros(iX,iU)
        Bp0 = zeros(iX,iU)
        s0 = zeros(iX,1)
        V0 = [x[:,i];X[:,i];A0[:];Bm0[:];Bp0[:];s0][:]

        um = u[:,i]
        up = u[:,i+1]
        Um = U[:,i]
        Up = U[:,i+1]
        dt = T[i]

        t, sol = RK4(dvdt,V0,(0,1),(um,up,Um,Up,dt),50)
        x_prop[:,i] .= sol[idx_x,end]
        X_prop[:,i] .= sol[idx_X,end]
        A[:,:,i] .= reshape(sol[idx_A,end],iX,iX)
        Bm[:,:,i] .= reshape(sol[idx_Bm,end],iX,iU)
        Bp[:,:,i] .= reshape(sol[idx_Bp,end],iX,iU)
        s[:,i] .= sol[idx_s,end]
        z[:,i] .= X_prop[:,i] - A[:,:,i]*X[:,i] - Bm[:,:,i]*Um - Bp[:,:,i]*Up - s[:,i] * dt
    end
    return A,Bm,Bp,s,z,x_prop,X_prop
end