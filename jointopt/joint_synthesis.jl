using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

include("joint_dynamics.jl")
include("joint_constraint.jl")
include("joint_utils.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")
include("../trajopt/costfunction.jl")
# include("../trajopt/utils.jl")

mutable struct JointSolution
    x::Matrix{Float64}
    u::Matrix{Float64}
    dt::Vector{Float64}

    Q::Array{Float64,3}
    Y::Array{Float64,3}
    Z::Array{Float64,3}

    A::Array{Float64,3}
    Bm::Array{Float64,3}
    Bp::Array{Float64,3}
    Aq::Array{Float64,3}
    Ax::Array{Float64,3}
    Bum::Array{Float64,3}
    Bup::Array{Float64,3}
    Bym::Array{Float64,3}
    Byp::Array{Float64,3}
    Bzm::Array{Float64,3}
    Bzp::Array{Float64,3}
    zf::Matrix{Float64}
    zF::Matrix{Float64}

    xi::Vector{Float64}
    xf::Vector{Float64}
    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    t::Any
    tprop::Any
    xprop::Any
    uprop::Any
    Qprop::Any
    Yprop::Any
    function JointSolution(N::Int64,ix::Int64,iu::Int64,iq::Int64,iy::Int64) 
        x = zeros(ix,N+1)
        u = zeros(iu,N+1)
        dt = zeros(N)
        
        Q = zeros(ix,ix,N+1)
        Y = zeros(iu,Int64(iy/iu),N+1)
        Z = zeros(ix,ix,N+1)

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
        zf = zeros(ix,N)
        zF = zeros(iq,N)

        xi = zeros(ix)
        xf = zeros(ix)
        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        new(x,u,dt,Q,Y,Z,A,Bm,Bp,Aq,Ax,Bum,Bup,Bym,Byp,Bzm,Bzp,zf,zF,xi,xf,Qi,Qf)
    end
end

struct JointSynthesis
    dynamics::Dynamics
    joint_dynamics::JointDynamics
    joint_constraint::Vector{JointConstraint}
    scaling::Any
    solution::JointSolution

    N::Int64  # number of subintervals (number of node - 1)
    w_traj::Float64
    w_funl::Float64  # weight for funnel cost
    w_vc_traj::Float64  # weight for virtual control
    w_vc_funl::Float64  # weight for virtual control
    w_tr_traj::Float64  # weight for trust-region
    w_tr_funl::Float64  # weight for trust-region
    w_rate::Float64
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    max_iter::Int64  # maximum iteration
    verbosity::Bool

    function JointSynthesis(N::Int,max_iter::Int,
        dynamics::Dynamics,joint_dynamics::JointDynamics,joint_constraint::Vector{T},scaling::Scaling,
        w_traj::Float64,w_funl::Float64,w_vc_traj::Float64,w_vc_funl::Float64,w_tr_traj::Float64,w_tr_funl::Float64,
        w_rate::Float64,
        tol_tr::Float64,tol_vc::Float64,tol_dyn::Float64,verbosity::Bool) where T <: JointConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        iq = joint_dynamics.iq
        iy = joint_dynamics.iy
        solution = JointSolution(N,ix,iu,iq,iy)
        new(dynamics,joint_dynamics,joint_constraint,scaling,solution,
            N,w_traj,w_funl,
            w_vc_traj,w_vc_funl,w_tr_traj,w_tr_funl,
            w_rate,
            tol_tr,tol_vc,tol_dyn,max_iter,verbosity)
    end

end

function sdpopt!(js::JointSynthesis,solver::String)
    N = js.N
    ix = js.dynamics.ix
    iu = js.dynamics.iu
    iq = js.joint_dynamics.iq
    iy = js.joint_dynamics.iy

    xbar = js.solution.x
    ubar = js.solution.u
    Qbar = js.solution.Q
    Ybar = js.solution.Y
    Zbar = js.solution.Z

    Sx = js.scaling.Sx
    iSx = js.scaling.iSx
    Su = js.scaling.Su
    iSu = js.scaling.iSu
    if typeof(js.joint_dynamics) == LinearQY
        Sr = Sx
        iSr = iSx
        ir = ix
    # elseif typeof(fs.funl_dynamics) == LinearQS
    #     Sr = Su
    #     iSr = iSu
    #     ir = iu
    end

    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
    else
        println("You should select Mosek or Clarabel")
    end


    # cvx variables (scaled)
    @variable(model, xcvx[1:ix,1:N+1])
    @variable(model, ucvx[1:iu,1:N+1])
    Qcvx = []
    Ycvx = []
    Zcvx = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Ycvx, @variable(model, [1:iu, 1:ir]))
        push!(Zcvx, @variable(model, [1:ix, 1:ix]))
    end
    @variable(model, vc_traj[1:ix,1:N])
    @variable(model, vc_t_traj[1:N])
    @variable(model, vc_funl[1:iq,1:N])
    @variable(model, vc_t_funl[1:N])

    # Q is PD
    very_small = 1e-12
    for i in 1:N+1
        @constraint(model, Qcvx[i] >= very_small * I(ix), PSDCone())
    end

    # scale reference trajectory
    xbar_scaled = zeros(ix,N+1)
    ubar_scaled = zeros(iu,N+1)
    dtbar_scaled = zeros(N)
    Qbar_scaled = zeros(ix,ix,N+1)
    Ybar_scaled = zeros(iu,ir,N+1)
    Zbar_scaled = zeros(ix,ix,N+1)

    for i in 1:N+1
        xbar_scaled[:,i] .= iSx*xbar[:,i]
        ubar_scaled[:,i] .= iSu*ubar[:,i]
        Qbar_scaled[:,:,i] .= iSx*Qbar[:,:,i]*iSx
        Ybar_scaled[:,:,i] .= iSu*Ybar[:,:,i]*iSr
        Zbar_scaled[:,:,i] .= iSx*Zbar[:,:,i]*iSx
    end

    # boundary condition
    @constraint(model, Sx*xcvx[:,1] == js.solution.xi)
    @constraint(model, Sx*xcvx[:,end] == js.solution.xf)
    @constraint(model, Sx*Qcvx[1]*Sx >= js.solution.Qi, PSDCone())
    @constraint(model, Sx*Qcvx[end]*Sx <= js.solution.Qf, PSDCone())

    # trajectory dynamics
    for i in 1:N
        @constraint(model,xcvx[:,i+1] == iSx*js.solution.A[:,:,i]*(Sx*xcvx[:,i])
            +iSx*js.solution.Bm[:,:,i]*(Su*ucvx[:,i])
            +iSx*js.solution.Bp[:,:,i]*(Su*ucvx[:,i+1])
            +iSx*js.solution.zf[:,i]
            +vc_traj[:,i]
            )
    end

    # funnel dynamics
    for i in 1:N
        xi = Sx*xcvx[:,i]
        ui = Su*ucvx[:,i]
        uip = Su*ucvx[:,i+1]
        Qi = Sx*Qcvx[i]*Sx
        Yi = Su*Ycvx[i]*Sr
        Zi = Sx*Zcvx[i]*Sx
        Qip = Sx*Qcvx[i+1]*Sx
        Yip = Su*Ycvx[i+1]*Sr
        Zip = Sx*Zcvx[i+1]*Sx
        @constraint(model, vec(Qip) == js.solution.Aq[:,:,i]*vec(Qi) + js.solution.Ax[:,:,i]*xi +
            js.solution.Bum[:,:,i]*vec(ui) + js.solution.Bup[:,:,i]*vec(uip) +
            js.solution.Bym[:,:,i]*vec(Yi) + js.solution.Byp[:,:,i]*vec(Yip) +
            js.solution.Bzm[:,:,i]*vec(Zi) + js.solution.Bzp[:,:,i]*vec(Zip) +
            js.solution.zF[:,i] +
            vc_funl[:,i]
            )
    end

    # Block LMI
    for i in 1:N+1
        @constraint(model, 0 <= Zcvx[i], PSDCone())
        # @constraint(model, 0 .== Zcvx[i])
    end
    
    # constraint
    N_constraint = size(js.joint_constraint,1)
    for i in 1:N+1
        for j in 1:N_constraint
            if typeof(js.joint_dynamics) == LinearQY
                Y = Su*Ycvx[i]*Sr
            # elseif typeof(fs.funl_dynamics) == LinearQS
            #     A,B = diff(fs.dynamics,xnom[:,i],unom[:,i])
            #     Y = -0.5*(Su*Ycvx[i]*Sr)*B'
            end
            impose!(js.joint_constraint[j],model,Sx*xcvx[:,i],Su*ucvx[:,i],Sx*Qcvx[i]*Sx,Y,xbar[:,i],ubar[:,i])
        end
    end

    # cost trajectory
    cost_traj = sum([get_cost(js.dynamics,Sx*xcvx[:,i],Su*ucvx[:,i],i,N) for i in 1:N+1])

    # cost funnel
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    # cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = - tr(Sx*Qcvx[1]*Sx)
    # cost_funl = tr(Sx*Qcvx[end]*Sx)
    cost_funl = - log_det_Q

    # virtual control - penalization on dynamics
    for i in 1:N
        @constraint(model, [vc_t_traj[i]; vc_traj[:,i]] in MOI.NormOneCone(1 + ix))
    end
    cost_vc_traj = sum([vc_t_traj[i] for i in 1:N])

    # virtual control - penalization on funnel dynamics
    for i in 1:N
        @constraint(model, [vc_t_funl[i]; vc_funl[:,i]] in MOI.NormOneCone(1 + iq))
    end
    cost_vc_funl = sum([vc_t_funl[i] for i in 1:N])

    # trust region of trajectory
    cost_tr_traj = sum([dot(xcvx[:,i]-xbar_scaled[:,i],xcvx[:,i]-xbar_scaled[:,i]) + dot(ucvx[:,i]-ubar_scaled[:,i],ucvx[:,i]-ubar_scaled[:,i]) for i in 1:N+1])

    # @variable(model, tr_norm[1:3,1:N+1])
    cost_tr_funl = sum([dot(vec(Qcvx[i]-Qbar_scaled[:,:,i]),vec(Qcvx[i]-Qbar_scaled[:,:,i])) +
    dot(vec(Ycvx[i]-Ybar_scaled[:,:,i]),vec(Ycvx[i]-Ybar_scaled[:,:,i])) +
    dot(vec(Zcvx[i]-Zbar_scaled[:,:,i]),vec(Zcvx[i]-Zbar_scaled[:,:,i]))
    for i in 1:N+1])

    # control rate
    cost_rate = sum([dot(ucvx[:,i+1] - ucvx[:,i],ucvx[:,i+1] - ucvx[:,i]) for i in 1:N])

    cost_all = (js.w_traj * cost_traj + js.w_funl * cost_funl +
        js.w_vc_traj * cost_vc_traj + js.w_vc_funl * cost_vc_funl +
        js.w_tr_traj * cost_tr_traj + js.w_tr_funl * cost_tr_funl +
        js.w_rate * cost_rate
        )
   
    @objective(model,Min,cost_all)
    optimize!(model)

    for i in 1:N+1
        js.solution.x[:,i] .= Sx*value.(xcvx[:,i])
        js.solution.u[:,i] .= Su*value.(ucvx[:,i])
        js.solution.Q[:,:,i] = Sx*value.(Qcvx[i])*Sx
        js.solution.Y[:,:,i] = Su*value.(Ycvx[i])*Sr
        js.solution.Z[:,:,i] = Sx*value.(Zcvx[i])*Sx
    end
    return value(cost_all),value(cost_traj),value(cost_funl),value(cost_vc_traj),value(cost_vc_funl),value(cost_tr_traj),value(cost_tr_funl),value(cost_rate)
end

function run(js::JointSynthesis,x0::Matrix,u0::Matrix,dt::Vector,
        Q0::Array{Float64,3},Y0::Array{Float64,3},Z0::Array{Float64,3},
        xi::Vector,xf::Vector,
        Qi::Matrix,Qf::Matrix,solver::String)
    js.solution.x .= x0
    js.solution.u .= u0
    js.solution.dt .= dt

    js.solution.Q .= Q0
    js.solution.Y .= Y0
    js.solution.Z .= Z0

    js.solution.xi .= xi 
    js.solution.xf .= xf
    js.solution.Qi .= Qi 
    js.solution.Qf .= Qf

    for iteration in 1:js.max_iter
        # discretization & linearization
        (
            js.solution.A, js.solution.Bm, js.solution.Bp, js.solution.Aq, js.solution.Ax, 
            js.solution.Bum, js.solution.Bup, js.solution.Bym, js.solution.Byp, js.solution.Bzm, 
            js.solution.Bzp, js.solution.zf, js.solution.zF, ~, ~
        ) = discretize_foh(
            js.joint_dynamics, js.dynamics,
            js.solution.Q[:, :, 1:N], js.solution.Y, js.solution.Z, js.solution.x[:, 1:N], 
            js.solution.u, js.solution.dt
        )
        # js.solution.A,js.solution.Bm,js.solution.Bp,js.solution.Aq,js.solution.Ax,js.solution.Bum,js.solution.Bup,js.solution.Bym,js.solution.Byp,js.solution.Bzm,js.solution.Bzp,js.solution.zf,js.solution.zF,x_prop,q_prop = discretize_foh(js.solution.joint_dynamics,fs.solution.dynamics,
        #     js.solution.Q[:,:,1:N],js.solution.Y,js.solution.Z,js.solution.x[:,1:N],js.solution.u,js.solution.dt)

        # solve subproblem
        cost_all,cost_traj,cost_funl,cost_vc_traj,cost_vc_funl,cost_tr_traj,cost_tr_funl,cost_rate = sdpopt!(js,solver)

        # propagate
        xfwd,Qfwd,js.solution.tprop,js.solution.xprop,js.solution.uprop,js.solution.Qprop,js.solution.Yprop =  propagate_multiple_FOH(DLMI,dynamics,
            js.solution.x,js.solution.u,js.solution.dt,js.solution.Q,js.solution.Y,js.solution.Z)
        traj_shooting_error = maximum(norm.(eachcol(xfwd - js.solution.x), 2))
        funl_shooting_error = maximum([norm(Qfwd[:,:,i] - js.solution.Q[:,:,i],2) for i in 1:N+1])

        if js.verbosity == true && iteration == 1
            println("+------------------------------------------------------------------------------------------------------------------------+")
            println("|                                                 ..:: Joint Synthesis ::..                                              |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+----------+----------+")
            println("| iter. |    cost    |    traj   |   funl    |   rate    |log(vc_t)|log(vc_f)|log(tr_t) |log(tr_f) |log(dyn_t)|log(dyn_f)|")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+----------+----------+")
        end
        @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.1f    |%-5.1f    |%-5.1f    |%-5.1f    | %-5.1e    |%-5.1e   |\n",
            iteration,
            cost_all,cost_traj,cost_funl,cost_rate,
            log10(abs(cost_vc_traj)),log10(abs(cost_vc_funl)),
            log10(abs(cost_tr_traj)),log10(abs(cost_tr_funl)),
            log10(abs(traj_shooting_error)),log10(abs(funl_shooting_error))
            )

        flag_vc::Bool = cost_vc_traj+cost_vc_funl < js.tol_vc
        flag_tr::Bool = cost_tr_traj+cost_tr_funl < js.tol_tr
        flag_dyn::Bool = traj_shooting_error+funl_shooting_error < js.tol_dyn

        if flag_vc && flag_tr && flag_dyn
            println("+------------------------------------------------------------------------------------------------------------------------+")
            println("Converged!")
            break
        end
    end
end