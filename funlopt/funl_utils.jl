
include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
# include("funl_dynamics.jl")
using LinearAlgebra

function create_block_diagonal(right::Matrix, n::Int)
    blocks = [right for _ in 1:n]
    return BlockDiagonal(blocks)
end

function get_index_for_upper(n::Int)
    matrix = reshape([i for i in 1:n^2], n, n)
    return vec_upper(matrix,n)
end

function vec_upper(A::Matrix{T},m::Int64) where T
    v = Vector{T}(undef, div(m*(m+1),2))
    # v = zeros(m*(m+1))
    k = 0
    @inbounds for j in 1:m
        @inbounds for i = 1:j
            v[k + i] = A[i,j]
        end
        k += j
    end
    return v
end

function inv_vec_upper(v::Vector{T},n::Int64) where {T}
    M = zeros(T, n, n)
    @assert size(M, 2) == n
    @assert length(v) >= n * (n + 1) / 2
    k = 0
    @inbounds for j in 1:n
        @inbounds @simd for i = 1:j
            M[i,j] = v[k+i]
            M[j,i] = v[k+i]
        end
        k += j
    end
    return M
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

function XU_to_QKZS(X::Matrix{Float64},U::Matrix{Float64},ix::Int,iu::Int)
    N = size(X,2)
    iq = size(X,1)
    # Q = reshape(X,(ix,ix,N))
    Q = Array{Float64}(undef,ix,ix,N)
    Z = Array{Float64}(undef,ix,ix,N)
    for i in 1:N
        Q[:,:,i] .= inv_vec_upper(X[:,i],ix) 
        Z[:,:,i] .= inv_vec_upper(U[ix*iu+1:ix*iu+iq,i],ix) 
    end
    # Z = reshape(U[ix*iu+1:ix*iu+ix*ix,:],(ix,ix,N))

    K = reshape(U[1:ix*iu,:],(iu,ix,N))
    S = U[ix*iu+iq+1:end,:]
    return Q,K,Z,S
end

function QKZS_to_XU(Q::Array{Float64,3},K::Array{Float64,3},Z::Array{Float64,3},S::Matrix{Float64})
    N = size(Q,3)
    ix = size(Q,1)
    iu = size(K,1)
    iq = div(ix*(ix+1),2)
    # q = reshape(Q,(ix*ix,N))
    q = Matrix{Float64}(undef,iq,N)
    z = Matrix{Float64}(undef,iq,N)
    for i in 1:N
        q[:,i] .= vec_upper(Q[:,:,i],ix)
        z[:,i] .= vec_upper(Z[:,:,i],ix)
    end
    k = reshape(K,(iu*ix,N))
    # z = reshape(Z,(ix*ix,N))
    return q,vcat(k,z,S)
end