using LinearAlgebra
using Random
using Ket
using Yao 
using Distributions
using CUDA
using Serialization
using Base.Threads 

⊗ = kron
cx = cnot
int(x) = Int(floor(x))
# CUDA.allowscalar(true)

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
dev - device to use. CUDA by default

Returns:  
a matrix of size q × n
"""
function zero_probs(db; dev = cu)::Matrix{Float64}    
    z = [1; 0]    

    if typeof(db) <: BatchedArrayReg # db is a BatchedArrayReg
        q = nqubits(db)
        n = nbatch(db)
        ret = zeros(Float64, q, n) |> dev
        pdb = Yao.probs(db) |> dev

        for j in 1:q
            v = ones(Float64, 2^(j-1)) ⊗ z ⊗ ones(Float64, 2^(q-j)) |> dev
            ret[j,:] = v'*pdb 
        end
    else # db is a regular matrix
        d, n = size(db)
        q = d |> log2 |> round |> Int
        ret = zeros(Float64, q, n) #|> dev
        pdb = abs2.(db) #|> dev

        for j in 1:q
            v = ones(Float64, 2^(j-1)) ⊗ z ⊗ ones(Float64, 2^(q-j)) #|> dev
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
Computes LSH given a database of states and a single circuit

Arguments:  
db - database of states  
circ - circuit to use on states  
τ - tau to use  
dev - device to use

Returns:  
an array of hashes of size q × n (with values -1, 0, or 1)
"""
function lsh(db, circ, tau; dev = cu, true_val=false)
    d, n = size(db)
    q = nqubits(circ)
    db_reg = BatchedArrayReg(db) |> dev

    hashes = zeros(Int, q, n)

    db_p = apply(db_reg, circ) |> cpu |> state
    probs = zero_probs(db_p)

    if true_val 
        return probs 
    end

    function cut(x)
        if abs(x-0.5) < tau return 0 end 
        return x>0.5 ? 1 : -1
    end
    hashes = cut.(probs)

    return hashes
end

"""
Computes aggregated LSH given a database of states and an array of circuits (projections)

Arguments:  
db - database of states  
projs - array of circuits to use on states  
qle - quantile for tau calculation  
dev - device to use

Returns:  
an array of hashes of size nprojs × q × n (with values -1, 0, or 1), 
"""
function lsh_total(db, projs, qle; dev = cu, true_val=false) 
    d, n = size(db)
    q = nqubits(projs[1])
    nprojs = length(projs)

    τ = pick_tau(q, qle)
    total_hashes = zeros(Int, nprojs, q, n)
    if true_val
        total_hashes = zeros(Float64, nprojs, q, n)
    end
    for p in 1:nprojs
        hashes = lsh(db, projs[p], τ, dev = dev, true_val=true_val) 
        total_hashes[p,:,:] = hashes
    end

    return total_hashes
end

"""
Picks random circuits (projections) given a database of states

Arguments:  
db - database of states  
nproj - number of circuits to pick  
reps - number of reselections of each circuit  
projs - list of already selected circuits  
fdiv - fraction of db states to focus on in the process. fdiv=3 means 1/3 of states will be in focus (with least non-zero hashes).  
qle - quantile for tau selection  
nlayer_mul - used for deduction the number of layers in circuits. nlayer = q*nlayer_mul

Returns:  
an array of circuits (including previously selected projs) 
"""
function pick_random_projections(db, nproj, reps=10; projs=[], fdiv = 3, qle = 0.1, nlayer_mul = 1) 
    d, n = size(db)
    q = d |> log2 |> round |> Int

    np = length(projs)
    ret = copy(projs) 

    nfocus = n÷fdiv # number of selected indices to focus on
    τ = pick_tau(q, qle)
    sum_hashes = zeros(Int, 1, n) # accumulate (over projections) the number of non-zero lsh for each db state
    focus_indices = []
    
    skip1 = 1
    if length(projs)==0 # first proj is fully random 
        circ = random_brick_pattern(q, q*nlayer_mul) 
        push!(ret, circ)
        skip1 = 2
        hashes = lsh(db, circ, τ)        
        sum_hashes = sum(abs.(hashes), dims=1)
    else 
        # calculate sum_hashes       
        for i=1:np
            hashes = lsh(db, projs[i], τ)
            sum_hashes += sum(abs.(hashes), dims=1)
        end
    end

    sorted = sort(collect(1:n), lt=(x,y)->sum_hashes[x]<sum_hashes[y])        
    focus_indices = sorted[1:nfocus]

    for i=skip1:nproj
        @info "Projection $i:"
        best = -1 # best value of total non-zero hashes on focus indices
        bsums = [] # per qubit sums of non-zero hashes for best circuit
        bcirc = Nothing # best circuit so far
        for r = 1:reps
            @info "  retry $r"
            circ = random_brick_pattern(q, q*nlayer_mul) # pick random circuit
            hashes = lsh(db, circ, τ) # compute hashes       
            sums = sum(abs.(hashes), dims=1) # compute sums of non-zero hashes
            focus_sums = sum(sums[focus_indices]) # look at focus indices 
            if focus_sums > best
                best = focus_sums
                bsums = sums
                bcirc = circ
            end
        end
        push!(ret, bcirc) # add best circuit to return array
        sum_hashes += bsums # update totals 
        sorted = sort(collect(1:n), lt=(x,y)->sum_hashes[x]<sum_hashes[y])                
        focus_indices = sorted[1:nfocus] # focus on new indices
    end

    return ret
end


"""
Linear db search for the best match between hashes

Arguments:  
lsh_t - lsh_total values found for a database and projections  
hash - lsh_total of the query state  
top - number of best matches to return. Default 10. 

Returns:  
array of top number of pairs (index, dist) where dist is the distance between hashes, sorted from best to worst 
"""
function linear_db_search_hash(lsh_t, hash; top = 10, true_val=false)
    nprojs, q, n = size(lsh_t)

    best = [(0, 0.0) for i in 1:top] # collect best matches
    for i=1:n
        if true_val
            dist = (hash.-0.5).*lsh_t[:,:,i] |> sum # Euclidean dist works much worse
        else
            dist = hash.*lsh_t[:,:,i] |> sum # Euclidean dist works much worse
        end
        if dist > best[top][2]
            push!(best, (i, dist))
            best = sort(best, by = x->x[2], rev = true)[1:top]
        end
    end

    return best
    # return first.(best)
end

"""
Linear db search for the best match given a query state

Arguments:  
lsh_t - lsh_total values found for a database and projections  
projs - array of projections  
top - number of best matches to return. Default 10.

Returns:  
array of top number of pairs (index, dist) where dist is the distance between hashes, sorted from best to worst 
"""
function linear_db_search(lsh_t, projs, state; top = 10, true_val=false)
    nprojs, q, n = size(lsh_t)
    db_s = reshape(state, 2^q, 1) # db with one state
    hash = lsh_total(db_s, projs, 0.1, true_val=true_val) 

    ret = linear_db_search_hash(lsh_t, hash, top = top, true_val=true_val)
    return ret
end

"""
Simple linear db search for the best match 

Arguments:  
db - database of states  
v - query state 

Returns:  
index of the best match under the fidelity
"""
function linear_ss(db, v)
    d, n = size(db)

    best_dist = 0
    best = 0 # best index
    for i=1:n
        dist = abs2(v'*db[:,i])
        # dist = abs2.(v.-db[:,i]) |> sum # Euclidean dist
        if dist >= best_dist
            best_dist = dist 
            best = i
        end
    end

    return best
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



function test(db, projs, lsh_t, num; fid=0.5, top=10, true_val=false) # num - number of query states 
    nprojs, q, n = size(lsh_t)
    passed = 0

    db_s = zeros(ComplexF64, 2^q, num)
    inds = zeros(Int, num)
    for t = 1:num
        ind = rand(1:n) # select random index of an element in the database
        inds[t] = ind
        v = db[:, ind]
        w = rand_fid(v, fid) # shift selected database vector
        db_s[:, t] = w # collect test (query) vectors
    end

    test_lsh_t = lsh_total(db_s, projs, 0.1, true_val=true_val)
    for t = 1:num    
        ret = linear_db_search_hash(lsh_t, test_lsh_t[:,:,t], top=top, true_val=true_val)
        # @show  inds[t], ret
        # fids = sort([abs2(db[:,inds[t]]'*db[:,ret[i][1]]) for i=1:top], rev=true)
        # @show fids[1]
        if inds[t] in first.(ret)
            passed += 1
        end
    end

    @info "passed: $passed/$num"   
    return passed
end

function big_test(db, projs, lsh_t; true_val=false)
    ret = [0 for i=0:10]
    @threads for i in 0:10 
        fid = 0.5 + i*0.05
        @show fid
        @time p = test(db, projs, lsh_t, 1000, fid = fid, top=10, true_val=true_val)
        ret[i] = p
    end
    save(ret, "ret.bin")
    return ret
end
