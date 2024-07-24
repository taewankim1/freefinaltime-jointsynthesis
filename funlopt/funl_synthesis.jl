using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel
using BenchmarkTools

include("funl_dynamics.jl")
include("funl_utils.jl")
include("funl_constraint.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")

mutable struct FunnelSolution
    X::Matrix{Float64}
    U::Matrix{Float64}

    A::Array{Float64,3}
    Bm::Array{Float64,3}
    Bp::Array{Float64,3}
    Bt::Matrix{Float64}
    rem::Matrix{Float64}

    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    t::Vector{Float64}

    tprop::Any
    xprop::Any
    uprop::Any
    Xprop::Any
    Uprop::Any
    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,is::Int64)
        iq = ix*ix
        iX = iq
        X = zeros(iX,N+1)
        ik = ix*iu
        iz = ix*ix
        iU = ik+iz+is
        U = zeros(iU,N+1)

        A = zeros(iX,iX,N)
        Bm = zeros(iX,iU,N)
        Bp = zeros(iX,iU,N)
        Bt = zeros(iX,N)
        rem = zeros(iX,N)

        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        
        t = zeros(N+1)
        new(X,U,A,Bm,Bp,Bt,rem,Qi,Qf,t)
    end
end

struct FunnelSynthesis
    dynamics::Dynamics
    funl_dynamics::FunnelDynamics
    funl_constraint::Vector{FunnelConstraint}
    scaling::Any
    solution::FunnelSolution

    N::Int64  # number of subintervals (number of node - 1)
    w_funl::Float64  # weight for funnel cost
    w_vc::Float64  # weight for virtual control
    w_tr::Float64  # weight for trust-region
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    max_iter::Int64  # maximum iteration
    verbosity::Bool

    # flag_type::String
    # funl_ctcs::Union{FunnelCTCS,Nothing}
    function FunnelSynthesis(N::Int,max_iter::Int,
        dynamics::Dynamics,funl_dynamics::FunnelDynamics,funl_constraint::Vector{T},scaling::Scaling,
        w_funl::Float64,w_vc::Float64,w_tr::Float64,tol_tr::Float64,tol_vc::Float64,tol_dyn::Float64,
        verbosity::Bool) where T <: FunnelConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        is = DLMI.is
        solution = FunnelSolution(N,ix,iu,is)
        new(dynamics,funl_dynamics,funl_constraint,scaling,solution,
            N,w_funl,w_vc,w_tr,tol_tr,tol_vc,tol_dyn,max_iter,verbosity,
            )
    end
end

function boundary_initial!(fs,model::Model,Q1)
    @constraint(model, Q1 >= fs.solution.Qi, PSDCone())
end

function boundary_final!(fs,model::Model,Qend)
    @constraint(model, Qend <= fs.solution.Qf, PSDCone())
end


function state_input_constraints!(fs,model::Model,Qi,Ki,Qbar,Kbar,xnom,unom,idx)
    N_constraint = size(fs.funl_constraint,1)
    for j in 1:N_constraint
        impose!(fs.funl_constraint[j],model,Qi,Ki,Qbar,Kbar,xnom,unom,idx)
    end
end

function sdpopt!(fs::FunnelSynthesis,xnom::Matrix,unom::Matrix,dtnom::Vector,Lipschitz::Vector,solver::String,iteration::Int64)
    N = fs.N
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    is = fs.funl_dynamics.is
    iphi = fs.dynamics.iϕ
    G = fs.dynamics.G

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    Su = fs.scaling.Su
    iSu = fs.scaling.iSu
    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false) # Turn off verbosity for Mosek
    else
        println("You should select Mosek or Clarabel")
    end

    # cvx variables (scaled)
    Qcvx = []
    Kcvx = []
    Zcvx = []
    VC = []
    for i in 1:N+1
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Kcvx, @variable(model, [1:iu, 1:ix]))
        push!(Zcvx, @variable(model, [1:ix, 1:ix], PSD))
        if i <= N
            push!(VC, @variable(model, [1:ix, 1:ix], Symmetric))
        end
    end
    @variable(model, vc_t[1:N])
    @variable(model, Scvx[1:is,1:N+1])

    # Q is PD and S >= 0
    very_small = 1e-5
    for i in 1:N+1
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small .* Matrix(1.0I,ix,ix), PSDCone())
        @constraint(model, Scvx[:,i] .>= 0)
    end

    # scale reference trajectory for trust region computation
    Qbar_scaled = zeros(ix,ix,N+1)
    Kbar_scaled = zeros(iu,ix,N+1)
    Zbar_scaled = zeros(ix,ix,N+1)
    Sbar_scaled = zeros(is,N+1)

    for i in 1:N+1
        Q_ = reshape(fs.solution.X[1:ix*ix,i],(ix,ix))
        K_ = reshape(fs.solution.U[1:iu*ix,i],(iu,ix))
        Z_ = reshape(fs.solution.U[iu*ix+1:iu*ix+ix*ix,i],(ix,ix))

        Qbar_scaled[:,:,i] .= iSx*Q_*iSx
        Kbar_scaled[:,:,i] .= iSu*K_*iSx
        Zbar_scaled[:,:,i] .= iSx*Z_*iSx
        Sbar_scaled[:,i] .= fs.solution.U[end-is+1:end,i]
    end

    # boundary condition
    boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Ki = Su*Kcvx[i]*Sx
        Zi = Sx*Zcvx[i]*Sx
        Si = Scvx[:,i]
        xi = xnom[:,i]
        ui = unom[:,i]

        if i <= N
            Qip = Sx*Qcvx[i+1]*Sx
            Kip = Su*Kcvx[i+1]*Sx
            Zip = Sx*Zcvx[i+1]*Sx
            Sip = Scvx[:,i+1]
            xip = xnom[:,i+1]
            uip = unom[:,i+1]
        end

        # Funnel dynamics
        if i <= N
            dti = dtnom[i]
            Ui = vcat(vec(Ki),vec(Zi),Si)
            Uip = vcat(vec(Kip),vec(Zip),Sip)
            @constraint(model, vec(Qip) == (
                fs.solution.A[:,:,i]*vec(Qi) 
                + fs.solution.Bm[:,:,i]*Ui
                + fs.solution.Bp[:,:,i]*Uip
                + fs.solution.Bt[:,i].*dti # * S_sigma
                + fs.solution.rem[:,i]
                + vec(VC[i])
            ))
        end

        # constraint with gamma
        Lip_squared = (Lipschitz[i]^2)
        LMI = [(-Lip_squared).*Zi Lip_squared.*G;
            Lip_squared.*G' (-Si[1]).* Matrix(1.0I,iphi,iphi)]
        @constraint(model, LMI <= 0, PSDCone())

        # constraints
        Qbar = reshape(fs.solution.X[1:ix*ix,i],(ix,ix))
        Kbar = reshape(fs.solution.U[1:iu*ix,i],(iu,ix))
        state_input_constraints!(fs,model::Model,Qi,Ki,Qbar,Kbar,xnom[:,i],unom[:,i],i)
    end

    # cost
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    cost_funl = - log_det_Q 
    # cost_funl = - tr(Sx*Qcvx[1]*Sx)

    # virtual control
    for i in 1:N
        @constraint(model, [vc_t[i]; vec(VC[i])] in MOI.NormOneCone(1 + ix*ix))
    end
    cost_vc = sum([vc_t[i] for i in 1:N])

    # trust region
    cost_tr = 0.0
    for i in 1:N+1
        Qdiff = vec(Qcvx[i]-Qbar_scaled[:,:,i]) 
        cost_tr += dot(Qdiff,Qdiff)
        Kdiff = vec(Kcvx[i]-Kbar_scaled[:,:,i]) 
        cost_tr += dot(Kdiff,Kdiff)
        Zdiff = vec(Zcvx[i]-Zbar_scaled[:,:,i]) 
        cost_tr += dot(Zdiff,Zdiff)
        Sdiff = Scvx[:,i]-Sbar_scaled[:,i]
        cost_tr += dot(Sdiff,Sdiff)
    end
    cost_all = fs.w_funl * cost_funl + fs.w_vc * cost_vc + w_tr * cost_tr
   
    @objective(model,Min,cost_all)
    optimize!(model)
    time_solver_time = solve_time(model)
    println("The elapsed time of solver is: $time_solver_time seconds")

    for i in 1:N+1
        Q = Sx*value.(Qcvx[i])*Sx
        K = Su*value.(Kcvx[i])*Sx
        Z = Sx*value.(Zcvx[i])*Sx
        S = value.(Scvx[:,i])
        fs.solution.X[:,i] .= vec(Q)
        fs.solution.U[:,i] .= vcat(vec(K),vec(Z),S)
    end

    return value(cost_all),value(cost_funl),value(cost_vc),value(cost_tr)
end

function run(fs::FunnelSynthesis,X0::Matrix{Float64},U0::Matrix{Float64},Lipschitz::Vector{Float64},
        Qi::Matrix,Qf::Matrix,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String)
    fs.solution.X .= X0
    fs.solution.U .= U0

    fs.solution.Qi .= Qi 
    fs.solution.Qf .= Qf

    for iteration in 1:fs.max_iter
        # discretization & linearization
        time_discretization = @elapsed begin
            fs.solution.A,fs.solution.Bm,fs.solution.Bp,fs.solution.Bt,fs.solution.rem,_,_ = discretize_foh(fs.funl_dynamics,
                fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.X[:,1:N],fs.solution.U)
        end
        println("The elapsed time of discretization is: $time_discretization seconds")

        # solve subproblem
        time_cvxopt = @elapsed begin
        c_all, c_funl, c_vc, c_tr = sdpopt!(fs,xnom,unom,dtnom,Lipschitz,solver,iteration)
        end
        println("The elapsed time of subproblem is: $time_cvxopt seconds")

        # propagate
        time_multiple_shooting = @elapsed begin
        (
            Xfwd,
            fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,
            fs.solution.Xprop,fs.solution.Uprop
        ) =  propagate_multiple_FOH(fs.funl_dynamics,fs.dynamics,
            xnom,unom,dtnom,fs.solution.X,fs.solution.U,flag_single=false)
        end
        println("The elapsed time of multiple shooting is: $time_multiple_shooting seconds")
        dyn_error = maximum(norm.(eachcol(Xfwd - fs.solution.X), 2))

        if fs.verbosity == true && iteration == 1
            println("+--------------------------------------------------------------------------------------------------+")
            println("|                                   ..:: Penalized Trust Region ::..                               |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
            println("| iter. |    cost    |    tof    |   funl    |   rate    |  param  | log(vc) | log(tr)  | log(dyn) |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
        end
        @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.3f    |%-5.1f    | %-5.1f    |%-5.1e   |\n",
            iteration,
            c_all,-1,c_funl,-1,
            -1,
            log10(abs(c_vc)), log10(abs(c_tr)), log10(abs(dyn_error)))

        flag_vc::Bool = c_vc < fs.tol_vc
        flag_tr::Bool = c_tr < fs.tol_tr
        flag_dyn::Bool = dyn_error < fs.tol_dyn

        if flag_vc && flag_tr && flag_dyn
            println("+--------------------------------------------------------------------------------------------------+")
            println("Converged!")
            break
        end
    end
end