include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")

using LinearAlgebra
abstract type JointDynamics end

function com_mat(m, n)
    A = reshape(1:m*n, n, m)  # Note the swapped dimensions for Julia
    v = reshape(A', :)

    P = Matrix{Int}(I, m*n, m*n)  # Create identity matrix
    P = P[v, :]
    return P'
end

struct LinearQY <: JointDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    iy::Int

    Cn::Matrix # commutation matrix
    Cm::Matrix # commutation matrix
    function LinearQY(alpha,ix,iu)
        Cn = com_mat(ix,ix)
        Cm = com_mat(iu,ix)
        new(alpha,ix,iu,ix*ix,ix*iu,Cn,Cm)
    end
end

function forward(model::LinearQY, dynamics::Dynamics,
    q::Vector, y::Vector, z::Vector,
    x::Vector, u::Vector)

    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))

    A,B = diff(dynamics,x,u)
   
    L = A*Q + B*Y
    dQ = L + L' + model.alpha*Q + Z
    return vec(dQ)
end

function diff(model::LinearQY,dynamics::Dynamics,
    q::Vector, y::Vector, z::Vector,
    x::Vector, u::Vector)

    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))

    A,B = diff(dynamics,x,u)
    fxx,fxu,fux,fuu = diff2(dynamics,x,u)

    Imat = I(model.ix)

    Aq = kron(Imat,A) + kron(A,Imat) * model.Cn + model.alpha * kron(Imat,Imat)
    Bq = kron(Imat,B) + kron(B,Imat) * model.Cm
    Sq = kron(Imat,Imat)

    Ax = zeros(model.iq,model.ix)
    for i in 1:model.ix
        Ax[:,i] = (kron(Imat,fxx[:,:,i]) + kron(fxx[:,:,i],Imat)) * q + (kron(Imat,fux[:,:,i]) + kron(fux[:,:,i],Imat)*model.Cm) * y
    end
    Bu = zeros(model.iq,model.iu)
    for i in 1:model.iu
        Bu[:,i] = (kron(Imat,fxu[:,:,i]) + kron(fxu[:,:,i],Imat)) * q + (kron(Imat,fuu[:,:,i]) + kron(fuu[:,:,i],Imat)*model.Cm) * y
    end

    return Aq,Bq,Sq,Ax,Bu
end

function discretize_foh(model::JointDynamics,dynamics::Dynamics,
        Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3},
        x::Matrix,u::Matrix,T::Vector)
    @assert size(x,2) == size(Q,3)
    @assert size(x,2) + 1 == size(u,2)
    @assert size(u,2) == size(Y,3)
    @assert size(Y,3) == size(Z,3)

    N = size(x,2)
    ix = model.ix
    iu = model.iu
    iq = model.iq
    iy = model.iy
    function get_interval(start,size)
        return start:(start+size-1)
    end

    idx_x = get_interval(1,ix)
    idx_q = get_interval(idx_x[end]+1,iq)
    idx_A = get_interval(idx_q[end]+1,ix*ix)
    idx_Bm = get_interval(idx_A[end]+1,ix*iu)
    idx_Bp = get_interval(idx_Bm[end]+1,ix*iu)
    idx_Aq = get_interval(idx_Bp[end]+1,iq*iq)
    idx_Ax = get_interval(idx_Aq[end]+1,iq*ix)
    idx_Bum = get_interval(idx_Ax[end]+1,iq*iu)
    idx_Bup = get_interval(idx_Bum[end]+1,iq*iu)
    idx_Bym = get_interval(idx_Bup[end]+1,iq*iy)
    idx_Byp = get_interval(idx_Bym[end]+1,iq*iy)
    idx_Bzm = get_interval(idx_Byp[end]+1,iq*iq)
    idx_Bzp = get_interval(idx_Bzm[end]+1,iq*iq)
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
        A_ = reshape(V[idx_A], (ix,ix))
        Bm_ = reshape(V[idx_Bm], (ix,iu))
        Bp_ = reshape(V[idx_Bp], (ix,iu))
        Aq_ = reshape(V[idx_Aq], (iq,iq))
        Ax_ = reshape(V[idx_Ax], (iq,ix))
        Bum_ = reshape(V[idx_Bum], (iq,iu))
        Bup_ = reshape(V[idx_Bup], (iq,iu))
        Bym_ = reshape(V[idx_Bym], (iq,iy))
        Byp_ = reshape(V[idx_Byp], (iq,iy))
        Bzm_ = reshape(V[idx_Bzm], (iq,iq))
        Bzp_ = reshape(V[idx_Bzp], (iq,iq))

        # traj terms
        f = forward(dynamics,x_,u_)
        fx,fu = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,dynamics,q_,y_,z_,x_,u_)
        Fq,Fy,Fz,Fx,Fu = diff(model,dynamics,q_,y_,z_,x_,u_)

        dx = f
        dq = F
        dA = fx*A_
        dBm = fx*Bm_ + fu*alpha
        dBp = fx*Bp_ + fu*beta
        dAq = Fq*Aq_
        dAx = Fq*Ax_ + Fx*A_
        dBum = Fq*Bum_ + Fx*Bm_ + Fu*alpha
        dBup = Fq*Bup_ + Fx*Bp_ + Fu*beta
        dBym = Fq*Bym_ + Fy*alpha
        dByp = Fq*Byp_ + Fy*beta
        dBzm = Fq*Bzm_ + Fz*alpha
        dBzp = Fq*Bzp_ + Fz*beta
        dV = [dx;dq;dA[:];dBm[:];dBp[:];
            dAq[:];dAx[:];dBum[:];dBup[:];dBym[:];dByp[:];dBzm[:];dBzp[:]]
        out .= dV[:]
    end

    A = zeros(ix,ix,N)
    Bm = zeros(ix,iu,N)
    Bp = zeros(ix,iu,N)
    Aq = zeros(iq,iq,N)
    Ax = zeros(iq,ix,N)
    Bum = zeros(iq,iu,N)
    Bup = zeros(iq,iu,N)
    Bym = zeros(iq,iy,N)
    Byp = zeros(iq,iy,N)
    Bzm = zeros(iq,iq,N)
    Bzp = zeros(iq,iq,N)

    x_prop = zeros(ix,N)
    q_prop = zeros(iq,N)
    zf = zeros(ix,N)
    zF = zeros(iq,N)

    for i = 1:N
        A0 = I(ix)
        Bm0 = zeros(ix,iu)
        Bp0 = zeros(ix,iu)
        Aq0 = I(iq)
        Ax0 = zeros(iq,ix)
        Bum0 = zeros(iq,iu)
        Bup0 = zeros(iq,iu)
        Bym0 = zeros(iq,iy)
        Byp0 = zeros(iq,iy)
        Bzm0 = zeros(iq,iq)
        Bzp0 = zeros(iq,iq)
        V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bm0[:];Bp0[:];
            Aq0[:];Ax0[:];Bum0[:];Bup0[:];Bym0[:];Byp0[:];Bzm0[:];Bzp0[:]]

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[:,:,i])
        zp = vec(Z[:,:,i+1])
        dt = T[i]

        t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt),50)
        x_prop[:,i] .= sol[idx_x,end]
        q_prop[:,i] .= sol[idx_q,end]
        A[:,:,i] .= reshape(sol[idx_A,end],ix,ix)
        Bm[:,:,i] .= reshape(sol[idx_Bm,end],ix,iu)
        Bp[:,:,i] .= reshape(sol[idx_Bp,end],ix,iu)
        Aq[:,:,i] .= reshape(sol[idx_Aq,end],iq,iq)
        Ax[:,:,i] .= reshape(sol[idx_Ax,end],iq,ix)
        Bum[:,:,i] .= reshape(sol[idx_Bum,end],iq,iu)
        Bup[:,:,i] .= reshape(sol[idx_Bup,end],iq,iu)
        Bym[:,:,i] .= reshape(sol[idx_Bym,end],iq,iy)
        Byp[:,:,i] .= reshape(sol[idx_Byp,end],iq,iy)
        Bzm[:,:,i] .= reshape(sol[idx_Bzm,end],iq,iq)
        Bzp[:,:,i] .= reshape(sol[idx_Bzp,end],iq,iq)
        zf[:,i] .= x_prop[:,i] - A[:,:,i]*x[:,i] - Bm[:,:,i]*um - Bp[:,:,i]*up
        zF[:,i] .= q_prop[:,i] - Aq[:,:,i]*vec(Q[:,:,i]) - Ax[:,:,i]*x[:,i] - Bum[:,:,i]*um - Bup[:,:,i]*up - Bym[:,:,i]*ym - Byp[:,:,i]*yp - Bzm[:,:,i]*zm - Bzp[:,:,i]*zp
    end
    return A,Bm,Bp,Aq,Ax,Bum,Bup,Bym,Byp,Bzm,Bzp,zf,zF,x_prop,q_prop
end
