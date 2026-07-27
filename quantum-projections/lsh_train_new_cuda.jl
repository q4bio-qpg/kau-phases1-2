using Yao #, Yao.EasyBuild
using Zygote
using CUDA
using LinearAlgebra
using Random
using Ket
using Serialization
using Optimisers    
using Distributions 
using Chain

import Yao: mat, niparams, getiparams, setiparams!, setiparams, print_block, nqudits, iparams_eltype, cache_key, render_params


# cpu(x) = x

⊗ = kron
cx = cnot
int(x) = Int(floor(x))
CUDA.allowscalar(true)

BitMask = Matrix{Int} # target measurement bits. 1 for |0>, -1 for |1>, 0 for ignore 

# Custom Yao blocks https://github.com/QuantumBFS/Yao.jl/blob/cc9eace41a6add6da26ba3c7ea0c3250c119d419/docs/src/man/blocks.md?plain=1#L82

# mutable struct TK2{T <: Real} <: PrimitiveBlock{2}
#     α::T
#     β::T 
#     γ::T 

# end

mutable struct U1q{T <: Number} <: PrimitiveBlock{2}
    α::T
    β::T 
end
nqudits(g::U1q) = 1
print_block(io::IO, block::U1q) = print(io, "U1q(α=$(block.α), β=$(block.β))")

function Base.:(==)(g1::U1q, g2::U1q)
    return g1.α == g2.α && g1.β == g2.β
end

function mat(::Type{T}, gate::U1q) where T 
    # M = Matrix{Complex{T}}(I, 2, 2)

    c = cos(gate.α/2)
    s = sin(gate.α/2)
    e = -im*exp(im*gate.β)

    # M[1,1] = c
    # M[1,2] = -s/e
    # M[2,1] = e*s
    # M[2,2] = c
    M = T[c -s/e; e*s c]

    return M     
end    

iparams_eltype(::U1q{T}) where T = T

niparams(::Type{<:U1q}) = 2
getiparams(g::U1q) = (g.α, g.β)
setiparams!(g::U1q, α::Number, β::Number) = ((g.α, g.β) = (α, β); g)
setiparams(g::U1q, α::Number, β::Number) = U1q(α, β)
# render_params(::U1q{T}, ::Val{:random}) where {T} = (rand(T,2)*2π,)


Base.adjoint(g::U1q) = U1q(-g.α, g.β)

cache_key(g::U1q) = (g.α, g.β)


function U1(α, β)
    M = zeros(ComplexF64, 2, 2)

    M[1,1] = cos(α/2)
    M[1,2] = -im*exp(-im*β)*sin(α/2)
    M[2,1] = -im*exp(im*β)*sin(α/2)
    M[2,2] = M[1,1]

    return M 
end

# test
# M1 = U1(0.1, 0.2)
# M2 = Matrix(mat(Rz(0.2-pi/2))*mat(Ry(0.1))*mat(Rz(pi/2-0.2)))
# M3 = Matrix(mat(Rz(0.2))*mat(Rx(0.1))*mat(Rz(-0.2)))
# M1 ≈ M2 ≈ M3

function TK2(α, β, γ)
    M = zeros(ComplexF64, 4, 4)

    M[1,1] = exp(-im*γ/2)*cos((α-β)/2)
    M[1,4] = -im*exp(-im*γ/2)*sin((α-β)/2)
    M[4,1] = M[1,4]
    M[4,4] = M[1,1]

    M[2,2] = exp(im*γ/2)*cos((α+β)/2)
    M[2,3] = -im*exp(im*γ/2)*sin((α+β)/2)
    M[3,2] = M[2,3]
    M[3,3] = M[2,2]
    
    return M
end

@const_gate S = mat(Rz(-pi/2))

function TK2_decomposed(α, β, γ)
    circ = chain(2)
    (α, β, γ) = [α, β, γ]*(-1/2)



    # push!(circ, put(1=>Rz(-pi/2)))
    push!(circ, put(1=>S))
    push!(circ, cx(2,1))
    push!(circ, put(2=>Ry(2β-pi/2)))
    push!(circ, cx(1,2))
    push!(circ, put(1=>Rz(2γ-pi/2)))
    push!(circ, put(2=>Ry(pi/2-2α)))
    push!(circ, cx(2,1))
    # push!(circ, put(2=>Rz(pi/2)))
    push!(circ, put(2=>S'))

    # return exp(im*pi/4)*circ'
    return circ
end

function TK2_decomposed!(circ, i, j, α, β, γ)
    (α, β, γ) = [α, β, γ]*(-1/2)

    # push!(circ, put(1=>Rz(-pi/2)))
    push!(circ, put(i=>S))
    push!(circ, cx(j,i))
    push!(circ, put(j=>Ry(2β-pi/2)))
    push!(circ, cx(i,j))
    push!(circ, put(i=>Rz(2γ-pi/2)))
    push!(circ, put(j=>Ry(pi/2-2α)))
    push!(circ, cx(j,i))
    # push!(circ, put(2=>Rz(pi/2)))
    push!(circ, put(j=>S'))

    # return exp(im*pi/4)*circ'
    return circ
end

# test
# M1 = exp(-im*pi/4)*TK2_decomposed(0.1,0.2,0.3) |> mat
# M2 = TK2(0.1, 0.2, 0.3)
# M1 ≈ M2

function brickwork(nbit, nlayer; entangler =:tk2, start_layer=true, parameters=:zero)
    @assert nbit%2 == 0    
    circuit = chain(nbit)

    if start_layer == true
        single_chain = chain(nbit)
        for j = 1:nbit
            push!(single_chain, put(j=>Rz(0.0)))
            push!(single_chain, put(j=>Rx(0.0)))
            push!(single_chain, put(j=>Rz(0.0)))
        end
        push!(circuit, single_chain)
    end

    if nlayer == 0 
        circuit = dispatch(circuit, parameters)
        return circuit 
    end 

    echain1 = chain(nbit)
    echain2 = chain(nbit)
    # for i = 1:2:nbit-1
    #      push!(echain1, put((i,i+1)=>TK2_decomposed!(0,0,0)))
    #      push!(echain2, put((i+1,(i+1)%nbit+1)=>TK2_decomposed(0,0,0)))
    # end
    for i = 1:2:nbit-1
        if entangler in (:cnot, :cx)
            push!(echain1, cnot(i,i+1))
            push!(echain2, cnot(i+1,(i+1)%nbit+1))
        elseif entangler == :tk2
            TK2_decomposed!(echain1, i, i+1, 0,0,0)
            TK2_decomposed!(echain2, i+1,(i+1)%nbit+1, 0,0,0)
        end 
    end

    ent_chain = [
        echain1,
        echain2
    ]

    for l = 1:nlayer
        push!(circuit, ent_chain[(l-1)%2+1])    
        # single_chain = chain(nbit, [put(j=>U1q(0.0, 0.0)*Rz(0.0)) for j=1:nbit])
        # single_chain = chain(nbit, [put(j=>Rz(0.0)*Rx(0.0)*Rz(0.0)) for j=1:nbit])
        single_chain = chain(nbit)
        for j = 1:nbit
            push!(single_chain, put(j=>Rz(0.0)))
            push!(single_chain, put(j=>Rx(0.0)))
            push!(single_chain, put(j=>Rz(0.0)))
        end
        push!(circuit, single_chain)
    end

    circuit = dispatch(circuit, parameters)
    return circuit
end

"""
Defines a circuit with the brick-like pattern

Arguments:  
nbit - number of qubits  
nlayer - number of thick layers. Each thick layer is an entangling layer plus a layer of single qubit gates. The total depth is thus 2*nlayer.  
paulis - an array of numbers in 1:3 that denote Rx, Ry, or Rz gates

Returns:  
a constructed circuit 
"""
function brick_pattern(nbit, nlayer, paulis)
    @assert nbit%2 == 0    
    circuit = chain(nbit)

    if nlayer == 0 
        return circuit 
    end 

    ent_chain = [
        chain(cx(i,i+1) for i in 1:2:nbit-1),
        chain(cx(i+1,(i+1)%nbit+1) for i in 1:2:nbit-1)
    ]
    R = [Rx, Ry, Rz]

    i = 1
    # # use a starting single-qubit layer
    # single_chain = chain(nbit)
    # for j = 1:nbit
    #     gate = R[paulis[i]]
    #     i+=1
    #     push!(single_chain, put(j=>gate(0.0)))        
    # end
    # push!(circuit, single_chain)

    for l = 1:nlayer
        push!(circuit, ent_chain[(l-1)%2+1])

        single_chain = chain(nbit)
        for j = 1:nbit
            gate = R[paulis[i]]
            i+=1
            push!(single_chain, put(j=>gate(0.0)))
        end
        push!(circuit, single_chain)
    end

    return circuit
end

"""
Defines a random brick-pattern circuit 

Arguments:  
nbit - number of qubits  
nlayer - number of thick layers. Each thick layer is an entangling layer plus a layer of single qubit gates

Note that it adds a starting single-qubit layer

Returns:  
a constructed circuit 
"""
function random_brick_pattern(nbit, nlayer)
    npar = nbit + nbit*nlayer
    paulis = rand(1:3, npar)
    params = rand(npar)*2pi

    R = [Rx, Ry, Rz]

    circuit = chain(nbit)

    start_layer = chain(nbit)
    i = 1
    for j = 1:nbit
        gate = R[paulis[i]]
        i+=1
        push!(start_layer, put(j=>gate(0.0)))        
    end
    push!(circuit, start_layer)

    bp = brick_pattern(nbit, nlayer, paulis[nbit+1:end])
    push!(circuit, bp) 

    circuit = dispatch(circuit, params)

    return circuit
end

"""
Extends a variational circuit with random brick pattern

Arguments:  
vqc - a given circuit  
nlayer - number of thick layers to add. Each thick layer is an entangling layer plus a layer of single qubit gates  

Returns:  
a constructed circuit 
"""
function extend_circuit(vqc, nlayer)
    nbit = nqubits(vqc)
    npar = nbit*nlayer
    paulis = rand(1:3, npar)    
    params = rand(npar)*2pi        

    circuit = copy(vqc)
    bp = brick_pattern(nbit, nlayer, paulis)
    bp = dispatch(bp, params)

    push!(circuit, bp)
    return circuit
end

"""
Extends a variational circuit with a dual (time-symmetric) random brick pattern

Arguments:  
vqc - a given circuit  
nlayer - number of thick layers to add (in one part). Each thick layer is an entangling layer plus a layer of single qubit gates  
params - either :rand or :zero

Returns:  
a constructed circuit 
"""
function extend_circuit_sym(vqc, nlayer; params = :zero)
    nbit = nqubits(vqc)    

    circuit = copy(vqc)

    bp1 = random_brick_pattern(nbit, nlayer)
    bp2 = random_brick_pattern(nbit, nlayer)

    if params == :zero 
        npar = nbit + nbit*nlayer
        bp1 = dispatch(bp1, zeros(Float64, npar))
        bp2 = dispatch(bp2, zeros(Float64, npar))
    end

    push!(circuit, bp1)
    push!(circuit, bp2')
    return circuit
end


"""
Generates a database of random states

Arguments:  
q - number of qubits  
n - number of states on q qubits

Returns:  
a matrix of size 2^q × n 
"""
function generate_db(q, n)::Matrix{ComplexF64} 
    d = 2^q
    
    db = zeros(ComplexF64, d, n)
    for i=1:n
        db[:, i] = random_state_ket(d)
    end

    return db    
end

"""
Saves serialized object to a file 

Arguments:  
data - object to save  
file - path to a file (string)
"""
function save(data, file)
    open(file, "w") do f
        serialize(f, data)
    end
end

"""
Loads deserialized object from a file 

Arguments:  
file - path to a file (string)
"""
function load(file)
    data = open(file, "r") do f
        deserialize(f)
    end
    return data
end

""" 
Computes probabilities of measuring 0 for each qubit in each state in a database

Arguments:  
db - database of states. Either a matrix of size 2^q × n, or a BatchedArrayReg  

Returns:  
a matrix of size q × n
"""
function zero_probs(db)::Matrix{Float64}    
    z = [1; 0]    

    if typeof(db) <: BatchedArrayReg # db is a BatchedArrayReg
        dev = cpu
        if typeof(db).parameters[3] <: CuArray
            dev = cu
        end

        q = nqubits(db)
        n = nbatch(db)
        ret = zeros(Float64, q, n) 
        ret = dev==cu ? dev(ret) : ret
        pdb = Yao.probs(db) #|> dev

        for j in 1:q
            v = ones(Float64, 2^(j-1)) ⊗ z ⊗ ones(Float64, 2^(q-j))
            v = dev==cu ? dev(v) : v
            ret[j,:] = v'*pdb 
        end
    else # db is a regular matrix
        dev = cpu
        if typeof(db) <: CuArray
            dev = cu
        end

        d, n = size(db)
        q = d |> log2 |> round |> Int
        ret = zeros(Float64, q, n)            
        ret = dev==cu ? dev(ret) : ret
        pdb = abs2.(db) #|> dev

        for j in 1:q
            v = ones(Float64, 2^(j-1)) ⊗ z ⊗ ones(Float64, 2^(q-j)) 
            v = dev==cu ? dev(v) : v
            ret[j,:] = v'*pdb 
        end
    end

    return ret
end

"""
Pick tau in the LSH method based on the quantile

Arguments:  
q - number of qubits  
qle - quantile. qle=0.1 means tau will cut 5% from both ends of the distribution

Returns:  
tau 
"""
function pick_tau(q, qle) 
    distr = Beta(2^(q-1),2^(q-1))
    τ = 0.5 - quantile(distr, qle/2) 
    # same: τ = quantile(distr, 1-qle/2) - 0.5

    # It should be cdf(distr, 0.5-tau) == qle/2 and cdf(distr, 0.5+tau) == 1-qle/2 
    return τ
end

"""
For each qubit select target indices from a (rotated) db

Arguments:  
db - database of states  
focus_indices - list of indices to focus on.  
qle - quantile (portion of elements to focus on from both ends). 
qle=0.1 means that, for each qubit, 5% of states with zero_prob closest to 0 and 5% of states closest to 1 will be in focus. 
qle is applied on top of focus_indices to produce target indices.

Returns:
pair (best0, best1), where best0 and best1 have size q × n*qle/2 and contain target indices
"""
function target_indices(db; qle=0.1, focus_indices=:all, show=false) 
    if typeof(db) <: BatchedArrayReg 
        q = nqubits(db)
        n = nbatch(db)
    else 
        d, n = size(db)
        q = d |> log2 |> round |> Int
    end    
    probs = zero_probs(db)

    qnf2 = q*n*qle/2 |> int
    best0 = zeros(Int, qnf2)
    best1 = zeros(Int, qnf2)
    # if focus_indices == :all
    #     focus_indices = collect(1:n)
    # end
    focus_indices = collect((i, j) for i = 1:q, j = 1:n) |> vec

    sorted = sort(focus_indices, lt=(a,b)->probs[a[1], a[2]]<probs[b[1], b[2]])
    best1 = sorted[1:qnf2]
    best0 = sorted[end-qnf2+1:end]
    
    if show 
        a = sorted[qnf2]
        b = sorted[end-qnf2+1]
        @show probs[a...], probs[b...]
        @show probs[sorted[1]...], probs[sorted[end]...]
    end
    return best0, best1 
end

"""
Generates bitmask based on found best0, best1 target indices

Arguments:  
n - number of elements in a db
b0 - best0 target indices
b1 - best1 target indices

Returns: 
mask of size q × n (with values -1,1,0)
"""
function _gen_bitmask(q, n, b0, b1) 
    bitmask = zeros(Int, q, n)
    
    for a in b0
        bitmask[a...] = 1 
    end
    for a in b1
        bitmask[a...] = -1 
    end

    return bitmask
end


function gen_bitmask(db, vqc, qle=0.1; show=false) 
    d, n = size(db)
    q = d |> log2 |> round |> Int
    
    bitmask = zeros(Int, q, n)
    
    db_reg = BatchedArrayReg(db)
    db_p = apply(db_reg, vqc)

    b0, b1 = target_indices(db_p, qle = qle, show=show)
    bitmask = _gen_bitmask(q, n, b0, b1)

    return bitmask
end


""" 
Train the given variational circuit on the database to amplify non-zero lsh values according to a bitmask

Arguments:  
db - database of states  
vqc - variational circuit  
bitmask - given bitmask  
step - number of optimisation steps  
dev - device to run on (BatchedArrayReg is super slow on CPU for some reason, ArrayReg is much faster)
lr - learning rate 

Returns:  
vqc with trained parameters
"""
function qml(db, vqc, bitmask, step; dev=cu, lr=0.01, lsx2 = false) 
    d, b = size(db)
    q = d |> log2 |> round |> Int
    qa = nqubits(vqc) 

    param = parameters(vqc)
    
    bstate = BatchedArrayReg(db)
    bstate = dev==cu ? dev(bstate) : bstate # superslow on CPU for some reason
    append_qubits!(bstate, qa-q) # in case of using ancilla qubits

    z = [1; 0]
    V = zeros(Float64, 2^qa, qa)
    for j in 1:qa
        v = ones(Float64, 2^(j-1)) ⊗ z ⊗ ones(Float64, 2^(qa-j))
        V[:,j] = v 
    end
    V = dev==cu ? dev(V) : V

    cubm = dev(bitmask)
    bm1 = (abs.(cubm) + cubm)/2
    bm2 = cubm-bm1
    function loss(param)
        r = 0
        vqc_p = dispatch(vqc, param)
        bstate_p = apply(bstate, vqc_p)
        # pr = probs(bstate_p)
        pr = abs2.(state(bstate_p))
        pr0 = V'*pr
        # @show size(pr0)
        if lsx2 
            r -= sum((bm1.*pr0).^2)
            r -= sum((bm2.*pr0 - bm2).^2)
        else 
            r -= sum(cubm.*pr0)
        end

        return r
    end        

    rule = Adam(lr)
    train_state = Optimisers.setup(rule, param)
    
    # start training
    for i = 1:step
        if step > 100
            if i%div(step,100)==0
                l = loss(param)
                @show Int(ceil(i/step*100)), l
            end
        else
            l = loss(param)
            @show l
        end
        grad = Zygote.gradient(loss, param)[1]
        train_state, param = Optimisers.update!(train_state, param, grad)
    end

    vqc_p = dispatch(vqc, param)     
    return vqc_p
end

function gen_ms(q, nm)
    ms = [[] for i = 1:nm]
    for i=1:nm
        m = []
        while true 
            m = rand([0,0,1,-1], q)
            if sum(abs.(m))==q÷2 
                break 
            end
        end
        ms[i] = m
    end

    return ms
end

function ssqml(v, vqc, ms, step; dev=cu, lr=0.01) 
    d = length(v)
    q = @chain d log2 Int
    nm = length(ms)

    param = parameters(vqc)
    
    reg = ArrayReg(v)
    reg = dev==cu ? dev(reg) : reg 

    V = zeros(Int64, d, nm)
    for k = 1:nm
        w = 1
        for j in 1:q
            if ms[k][j] == '0'
                w = w ⊗ [1; 0]
            elseif ms[k][j] == '1'
                w = w ⊗ [0; 1]
            else # '+'
                w = w ⊗ [1; 1]
            end
        end
        V[:, k] = w
    end
    V = dev==cu ? dev(V) : V 

    function loss(param)
        vqc_p = dispatch(vqc, param)
        reg_p = apply(reg, vqc_p)
        # pr = probs(reg_p)
        pr = abs2.(state(reg_p))
        r = V'*pr
        # @show size(r)
        # @show size(V)
        # @show size(pr)

        return -sum(r)
    end        

    rule = Adam(lr)
    train_state = Optimisers.setup(rule, param)
    
    l = 0
    # start training
    for i = 1:step
        if step > 100
            if i%div(step,100)==0
                l = loss(param)
                pcnt = Int(round(i/step*100))
                @show pcnt, l
            end
        else
            l = loss(param)
            @show l
        end
        grad = Zygote.gradient(loss, param)[1]
        train_state, param = Optimisers.update!(train_state, param, grad)
    end

    vqc_p = dispatch(vqc, param)     
    return vqc_p, l
end

function state_prep(v, depth, step, iter)
    d = length(v)
    q = @chain d log2 Int
    sp = ['+' for i=1:q]
    ms = [ (spi = copy(sp); spi[i]='0'; spi) for i=1:q]
    
    vqc = chain(q)
    vi = copy(v) |> cu
    for i = 1:iter 
        @info "Iteration $i"

        if i==1 
            vqc = brickwork(q, depth; start_layer=true, parameters=:random)
        else 
            vqc = brickwork(q, depth; start_layer=false, parameters=:zero)*vqc 
        end

        @show nparameters(vqc)

        vqc, l = ssqml(vi, vqc, ms, step, lr=0.001*(i==1 ? 10 : 1))
        # @show l
        # push!(vqcs, vqc)
        # vqcs = deepcopy(vqc)

        # if i < iter
        #     vi = apply(ArrayReg(vi), vqc) |> state
        # end
    end
    return vqc, v
end

"""
Adaptive training of a circuit in the LSH method

Arguments:  
db - database of states  
vqc - variational circuit (with a fixed structure)   
qle - quantile (portion of elements to focus on from both ends). 
qle=0.1 means that, for each qubit, 5% of states with zero_prob closest to 0 and 5% of states closest to 1 will be in focus. 
qle is applied on top of focus_indices to produce target indices.
step - number of optimizer steps in each iteration  
iter - number of optimizer iterations (bitmask adaptations)
focus_indices - list of indices to focus on.  

Returns:  
vqc with trained parameters
"""
function adaptive_qml(db, vqc, qle, step=10, iter=10; focus_indices=:all, wiggle=false, dev=cu)
    d, n = size(db)
    q = d |> log2 |> round |> Int
    db_reg = BatchedArrayReg(db) |> dev

    vqc_p = copy(vqc)

    if focus_indices == :all
        focus_indices = collect(1:n)
    end

    f = qle
    bitmask = zeros(Int, q, n)
    for i in 1:iter
        @info "Iteration $i"
        # vqc_p = dispatch(vqc, param)        
        db_p = apply(db_reg, vqc_p) # |> cpu |> state
        
        fi = f
        if wiggle # change fraction f for more variability (could give some advantage)
            if (i-1) % 2 == 0
                fi = f 
            elseif (i-1) % 4 == 1
                fi = 2f
            elseif (i-1) % 4 == 3
                fi = f/2
            end 
            if i == iter 
                fi = f 
            end
        end

        b0, b1 = target_indices(db_p, qle = fi, focus_indices=focus_indices)
        bitmask = _gen_bitmask(n, b0, b1)
        
        @time vqc_p = qml(db[:, focus_indices], vqc_p, bitmask[:, focus_indices], step, dev=dev)    
    end
    return vqc_p, bitmask
end


function adaptive_simple(db, vqc, qle=0.1, step=10, iter=10; dev=cu, lr=0.01, lsx2=false)
    d, n = size(db)
    q = d |> log2 |> round |> Int

    vqc_p = copy(vqc)

    for i in 1:iter
        bitmask = gen_bitmask(db, vqc_p, qle)
        @info "Iteration $i"
        
        @time vqc_p = qml(db, vqc_p, bitmask, step, dev=dev, lr=lr, lsx2=lsx2)    
    end
    return vqc_p
end

function mqml(cudb; f = 0.1/4, num = 2000, nk = 20, lsx2=false)
    d, n = size(db)
    q = d |> log2 |> round |> Int

    vqc = brickwork(q, q)
    vpn = dispatch(vqc, :random)
  
    vpn = adaptive_simple(cudb, vpn, f, nk, nk, lr=0.01, lsx2=lsx2)
    vpn = adaptive_simple(cudb, vpn, f, nk, nk, lr=0.001, lsx2=lsx2)
    vpn = adaptive_simple(cudb, vpn, f, num, 1, lr=0.001, lsx2=lsx2)

    return vpn
end

function mqml2(cudb; f = 0.1/4, num = 2000, nk = 20)
    d, n = size(db)
    q = d |> log2 |> round |> Int

    vqc = brickwork(q, 2q)
    vpn = dispatch(vqc, :random)
  
    vpn = adaptive_simple(cudb, vpn, f, nk, nk, lr=0.01)
    vpn = adaptive_simple(cudb, vpn, f, nk, nk, lr=0.001)
    vpn = adaptive_simple(cudb, vpn, f, num, 1, lr=0.001)

    return vpn
end

function fullmqml(db; nproj=10, num=2000, nk = 20)
    d, n = size(db)
    q = d |> log2 |> round |> Int

    tsum = zeros(Int64, n)
    projs = []
    idb = db

    # first projection is more "free"
    vqc = mqml(idb, f=0.1/4, num=num, nk=nk)
    push!(projs, deepcopy(vqc))

    for i=2:nproj
        mask = gen_bitmask(db, vqc, 0.1/4)
        isum = sum(abs.(mask), dims=1)[:]
        tsum += isum
        @show tsum |> sort

        sorted = sort(collect(1:n), lt=(x,y)->tsum[x]<tsum[y])
        focus = sorted[1:n÷2]
        idb = db[:, focus]

        @info "Projection $i:"
        vqc = mqml(idb, f=0.1/2, num=num, nk=nk)
        push!(projs, deepcopy(vqc)) 
    end

    mask = gen_bitmask(db, vqc, 0.1/4)
    isum = sum(abs.(mask), dims=1)[:]
    tsum += isum
    @show tsum |> sort

    return projs
end



"""
Computes how well a variational circuit rotates a database towards the end goal (shifting qubit zero probs away from 0.5 average).  

For each qubit, probability of measuring 0 is computed and db elements are sorted over this value. They split in two halves around ≈0.5 value. 
In each half, we look at the portion of size n*qle/2 of best values (which are closer either to 0 or 1 respectively). 
Then two mean values, mv0 and mv1, are picked (one in each half). It should be mv0 ≈ 1-mv1. 
The average mv of (mv0+(1-mv1))/2 over qubits is computed and the corresponding quantile is returned. 
For the empty circuit and random db it should return a value close to qle/4.

Arguments: 
db - database of states  
vqc - variational circuit to apply on db  
qle - specifies the portion of best values to look at. It makes sence to take qle used in training of the circuit  
factor - return the improving factor instead

Returns:  
either quantile (averaged over qubits and halfs) or the improving factor
"""
function mean_qle(db, vqc, qle; factor = true)
    d, n = size(db)
    q = d |> log2 |> round |> Int
    nf2 = n*qle/2 |> int

    distr = Beta(2^(q-1),2^(q-1)) # the distribution 

    db_reg = BatchedArrayReg(db)
    db_rot = apply(db_reg, vqc)
    probs = zero_probs(db_rot)

    mvs = zeros(Float64, q) # mean value avgs per each qubit
    for j in 1:q
        sorted = sort(collect(1:n), lt=(x,y)->probs[j, x]<probs[j, y])
        mv0 = probs[j, sorted[nf2÷2+1]] 
        mv1 = probs[j, sorted[end-nf2÷2]] 

        mvs[j] = (mv0+(1-mv1))/2
    end

    mv = sum(mvs)/q
    ret = cdf(distr, mv) 

    if factor 
        return qle/4/ret
    end

    return ret
end

# todo: can be done in parallel on a split database db 
function find_proj(db, f=0.1; nproj=12, mode = :adaptive) # find nproj projections, f is the fraction of target non-zeros in a bitmask. Three modes - :notrain is no training, :simple is simply the qml, :adaptive is adaptive qml
    d, n = size(db)
    q = d |> log2 |> round |> Int
    db_reg = BatchedArrayReg(db) |> cu

    vqc = random_brick_pattern(q, q) 

    m = nparameters(vqc)

    param_list = [] # list of parameters for each projection
    all_mask = zeros(Int, n, nproj, q)
    pick_list = :all
    stm = zeros(Int, n) 
    n2 = n/2 |> floor |> Int        
    for i = 1:nproj
        @info "Projection $i"
        param = 4pi*rand(Float64, m)

        if mode == :adaptive
            param, mask = adaptive_qml(db, vqc, f, 10, 13, param=param, pick_list=pick_list)
        else 
            vqc_p = dispatch(vqc, param)        
            db_p = apply(db_reg, vqc_p) |> cpu |> state
            b0, b1 = select_best(db_p, f, pick_list=pick_list)
            mask = gen_bitmask(b0, b1, n)

            if mode == :simple
                param = qml(db, vqc, 10, bitmask=mask, param=param, dev=:gpu)
            elseif mode == :notrain 
                # skip training
            end
        end

        push!(param_list, param) 
        all_mask[:,i,:] = mask'        

        # update pick list - see by how many non-zeros in bitmasks each vector is covered and focus on the half of the database that is less covered
        stm += sum(abs2.(mask), dims=1) |> vec 
        sorted = sort(collect(1:n), lt=(x,y)->stm[x]<stm[y])        
        pick_list = sorted[1:n2]
    end
    
    # TODO return circuits not params
    return param_list, all_mask
end

function lsh(db2, param_list; τ = 1e-5) # compute LSH for test vectors collected into a db2 database, param_list are projection parameters 
    d, n2 = size(db2)
    q = d |> log2 |> round |> Int
    nproj = length(param_list)
    db_reg = BatchedArrayReg(db2) |> cu

    vqc = variational_circuit(12, 12)
    # m = nparameters(vqc)

    hashes = zeros(Int, nproj, q, n2)
    for i = 1:nproj
        # @info "Projection $i"
        param = param_list[i]
        vqc_p = dispatch(vqc, param)        
        db_p = apply(db_reg, vqc_p) # |> cpu |> state
        probs = zero_probs(db_p)
        function cut(x)
            # τ = 1e-1 # threshold
            if abs(x-0.5) < τ return 0 end 
            return x>0.5 ? 1 : -1
        end
        hashes[i,:,:] = cut.(probs)
    end
    return hashes
end

function compare_hash(hashes, masks) # compute Hamming distances between hashes of all vectors in the test database db2
    n, nproj, q = size(masks)
    n2 = size(hashes, 3)
    dist = zeros(Int, n, n2)
    for j = 1:n2
        for i = 1:n 
            dist[i,j] = hashes[:,:,j].*masks[i,:,:] |> sum
        end
    end
    return dist 
end

"""
Generates Haar random state with a given fidelity f with a state v

Arguments:  
v - given state  
f - target fidelity 

Returns:  
random state w that has fidelity f with v
"""
function rand_fid(v, f) 
    d = length(v)
    q = d |> log2 |> round |> Int
    ort = random_state_ket(d)
    ort = ort - (v' * ort) * v # orthogonalize to v
    ort /= norm(ort) 
    w = sqrt(f)*v + sqrt(1-f)*ort
    return w
end

function test(db, param_list, masks; num=1, fid=0.5, top=10, τ=1e-5) # test the algorithm. num - number of test vectors, fid - fidelity between the query and some database vector, top - the number of vectors with best Hamming distance against query to consider: each test is passed if the correct index is in the top best indices
    d, n = size(db)
    q = d |> log2 |> round |> Int
    nproj = length(param_list)

    passed = 0
    best = zeros(Int, n)

    db2 = zeros(ComplexF64, d, num)
    inds = zeros(Int, num)
    for t = 1:num
        ind = rand(1:n) # select random index of an element in the database
        inds[t] = ind
        v = db[:, ind]
        w = rand_fid(v, fid) # shift selected database vector
        db2[:, t] = w # collect test (query) vectors
    end

    hashes = lsh(db2, param_list, τ=τ) # compute hashes for test vectors
    dist = compare_hash(hashes, masks)

    for j=1:num
        sorted = sort(collect(1:n), lt=(x,y)->dist[x,j]>dist[y,j])
        passed += (inds[j] in sorted[1:top]) ? 1 : 0
    end
    

    @show passed/num, " - portion of passed tests"
    return passed/num
end

function main_test(db, params, masks)
    ret = []
    for i in 0:10 
        fid = 0.5 + i*0.05
        @time p = test(db, params, masks, num = 1000, fid = fid, top=10, τ=1e-2)
        push!(ret, p)
    end
    return ret
end

function full_test() # generate db, projections, test vectors and check results
    q = 12 # number of qubits
    n = 10^3 # number of database vectors
    db = generate_db(q, n) 
    println("Database generated with $n vectors on $q qubits")

    param_list, masks = find_proj(db, nproj=20, mode=:adaptive)
    println("20 projections found!")

    num = 100 # number of test vectors
    fid = 0.8 # fidelity with a random vector in the database
    top = 10 # number of best indices to consider

    test(db, param_list, masks; num=num, fid=fid, top=top)
end
