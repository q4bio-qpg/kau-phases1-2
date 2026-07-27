### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 39c52222-4e8e-11ef-0a00-b5808f067508
begin
	using LinearAlgebra
	using Optim
	using SparseArrays
	using RandomMatrices
	using Memoize
end

# ╔═╡ 2915a690-6f0f-45a6-b5dd-a7cc01241aa7
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 504f61bd-e407-4282-bb5f-abc27c462d36
begin
	⊗ = kron
	rcx = Haar(2) # complex	
	
	@memoize Dict{Tuple{Int64,Int64}, Any} function ket(dim, i::Int64)::SparseMatrixCSC{ComplexF64, Int64}
		ret = spzeros(ComplexF64,dim,1)
		i = mod(i,dim)
		ret[i+1,1] = 1
		
		return ret
	end
	bra(dim, i::Int64) = ket(dim, i::Int64)'

	X = spzeros(ComplexF64,2,2)
	X[1,2] = 1
	X[2,1] = 1
	
	function Rx(θ::Real)
	  return [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
	end
	
	function Ry(θ::Real)
	  return [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
	end
	
	function Rz(θ::Real)
	  return [exp(-im*θ/2) 0; 0 exp(im*θ/2)]
	end

	function cx(a,b,n) # a,b < n, 0-based
		if a < b
			c = (ket(2,0)*bra(2,0)) ⊗ I(2^(b-a)) + (ket(2,1)*bra(2,1)) ⊗ I(2^(b-a-1)) ⊗ X # cx(0,b-a,b-a+1)
			return I(2^a) ⊗ c ⊗ I(2^(n-b-1))
		else 
			a, b = b, a
			c = I(2^(b-a)) ⊗ (ket(2,0)*bra(2,0)) + X ⊗ I(2^(b-a-1)) ⊗ (ket(2,1)*bra(2,1))
			return I(2^a) ⊗ c ⊗ I(2^(n-b-1))
		end
	end
end

# ╔═╡ d937c8dd-436b-49f1-af12-d70fbc062ac1
cx(1,0,2)

# ╔═╡ cf48bf13-1a31-450e-bc4f-192329f335ed
q = 4

# ╔═╡ b2efdb0f-9ac2-4145-a5ff-3918abda52af
# generate random database with 2^q states
function genDB(q=4) 
	ret = zeros(ComplexF64, 2^q, 2^q)
	for i=1:2^q
		ret[:,i] = rand(rcx, 2^q)*ket(2^q,0)
	end
	return ret
end

# ╔═╡ 77e3476c-1b06-4a2e-9540-160d037d4a23
@time DB = genDB(q)

# ╔═╡ b16d67d4-a87a-4455-a595-2065f207096b
function layer(x) # x is an array n x 3 of parameters. 2n gates 
	n = size(x)[1]
	r = 1
	for i = 1:n
		r = r ⊗ (Rz(x[i,1])*Rx(x[i,2])*Rz(x[i,3]))
	end
	# c = 1
	for i=1:n-1
		r = cx(i-1,i,n) * r
	end
	return cx(n-1,0,n) * r 
end

# ╔═╡ 7889445c-6660-4d6d-892b-4ff447da47f5
function model(p) 
	l = size(p)[1]
	ret = layer(p[1,:,:])
	for i = 2:l
		ret = layer(p[i,:,:]) * ret
	end
	return ret
end

# ╔═╡ 00a5a293-4b8d-4e59-bfc0-4bf7408a4ab8
function loss(DB, U)
	-sum(norm(ket(2^q,i-1)'*U*DB[:,i])^2 for i=1:2^q)/2^q
end

# ╔═╡ 07a3a8c1-5799-4dc1-8023-c792d9624484
# Loss for a unitary U that rotates DB, concerning the first qubit
function loss1(DB, U)
	q = Int(log2(size(DB)[1]))
	n = size(DB)[2]
	P0 = kron( (ket(2,0)*ket(2,0)'), I(2^(q-1)) ) 
	P1 = I - P0
	UDB = U*DB
	ret = 0
	for i in 1:Int(n/2)
		ret += -norm(P0*UDB[:,i])^2 
		# same as probability of measuring 0 at first qubit of UDB[:,i])
  	end
  	for i in Int(n/2)+1:n
    	ret += -norm(P1*UDB[:,i])^2 
		# same as probability of measuring 1 at first qubit of UDB[:,i])
	end
	return ret/n
end

# ╔═╡ 49abd1cc-eb97-40af-a640-62ac79f59db8
loss1(DB[:, 1:2^(q-1)], I)

# ╔═╡ af36189a-243c-4695-9b81-8e74d0742f99
function target(p)
	M = model(p)
	return loss1(DB, M)
end

# ╔═╡ 74acad20-dc7b-48b2-981e-e5d075bd8c92
# Compute best rotating U, concerning first qubit 
function bestU(DB)
	q = Int(log2(size(DB)[1]))
	M = sum(DB[:,i]*DB[:,i]' for i in 2^(q-1)+1:2^q) - sum(DB[:,i]*DB[:,i]' for i in 1:2^(q-1))
	U = eigvecs(Hermitian(M))
	return U'
end

# ╔═╡ 7507e661-1a89-41a2-99d4-a7079adea063
function runOptim(func, ini)
	manif = Optim.Flat()
	optimizer = Optim.LBFGS(manifold=manif)
	res = Optim.optimize(func, ini, optimizer, Optim.Options(iterations = 100, allow_f_increases=false, g_abstol = 1e-12, show_trace=false))

	bestmn = Optim.minimum(res)
	bestmnz = Optim.minimizer(res)
	println(bestmn)

	return (bestmn, bestmnz)
end

# ╔═╡ a69d389d-8716-47d7-ac95-5506e9cf8aef
function incrementalOptim(DB)
	q = Int(log2(size(DB)[1]))
	n = size(DB)[2]
	n2 = Int(n/2)
	cDB = hcat(DB[:,n2],DB[:,n2+1])
	# para = rand(Int(floor(q/2)),q,3)*2pi  # q=5 => 20+16+6+4+1=47  # 8%
	# para = rand(Int(ceil(q/2)),q,3)*2pi  # q=5 => 30+16+12+4+1=63  # 11%
	para = rand(q,q,3)*2pi  # q=5 => 50+32+18+8+1=109  # q=4 26%, fid=0.7 => 42%, fid=0.82 => 0.5%; q=5 19%
	# para = rand(1,q,3)*2pi # 10+8+6+4+1 # q=4 fid=1.0 => 20%
	for i = 1:n2
		@show size(cDB)
		function tar(p)
			M = model(p)
			return loss1(cDB, M)
		end
		r = runOptim(tar, para)
		@show r[1]
		para = r[2]
		if i < n2 
		  cDB = hcat(DB[:,n2-i], cDB, DB[:,n2+1+i])
		end
	end
	return model(para)
end

# ╔═╡ 03bb44f1-01fa-4aec-b9a7-3b7c8dcda9e2
# p = incrementalOptim(DB)

# ╔═╡ 994cc9de-482b-4bdf-93cd-9b1dcb9f755f
# Compute best U recursively for all qubits
function bestUrec(DB)	
	q = Int(log2(size(DB)[1]))
	U = bestU(DB)
	if q == 1 
		return U
	end
	P0 = kron( (ket(2,0)*ket(2,0)'), I(2^(q-1)) )
	P1 = I-P0

	DB0 = (P0*U*DB[:, 1:2^(q-1)])[1:2^(q-1),:]
	DB1 = (P1*U*DB[:, 2^(q-1)+1:2^q])[2^(q-1)+1:2^q,:]
	
	for i=1:2^(q-1)
		DB0[:,i]/=norm(DB0[:,i])
		DB1[:,i]/=norm(DB1[:,i])
	end
		
	U0 = bestUrec(DB0)
	U1 = bestUrec(DB1)
	
	V0 =  kron( (ket(2,0)*ket(2,0)'), U0 )
	V1 =  kron( (ket(2,1)*ket(2,1)'), U1 )

	return (V0+V1)*U
end

# ╔═╡ 822752c3-1173-422b-9c69-29a1c673adc7
@time BUR = bestUrec(DB)

# ╔═╡ 00d2e683-258b-4599-a8f3-75a9460b0a40
loss1(DB, BUR)

# ╔═╡ 56e2b464-503c-4662-8b83-2f2523c1dfc5
loss(DB, BUR)

# ╔═╡ c0a32ac7-3b29-497a-b0ec-6928d8484091
# Optim find of best U recursively for all qubits
function bestUrecOptim(DB)	
	q = Int(log2(size(DB)[1]))
	if q == 1 
		return bestU(DB)
	end
	U = incrementalOptim(DB)
	
	P0 = kron( (ket(2,0)*ket(2,0)'), I(2^(q-1)) )
	P1 = I-P0

	DB0 = (P0*U*DB[:, 1:2^(q-1)])[1:2^(q-1), :]
	DB1 = (P1*U*DB[:, 2^(q-1)+1:2^q])[2^(q-1)+1:2^q, :]
	
	for i=1:2^(q-1)
		DB0[:,i]/=norm(DB0[:,i])
		DB1[:,i]/=norm(DB1[:,i])
	end
		
	U0 = bestUrecOptim(DB0)
	U1 = bestUrecOptim(DB1)
	
	V0 =  kron( (ket(2,0)*ket(2,0)'), U0 )
	V1 =  kron( (ket(2,1)*ket(2,1)'), U1 )

	return (V0+V1)*U
end

# ╔═╡ 4cf9bde7-4e46-4181-a88d-ec28387d8f0f
@time BURO = bestUrecOptim(DB)

# ╔═╡ 5ae9eae2-7db9-4d92-be9a-c3f38585fb07
function tar(p)
	loss(DB,model(p))
end

# ╔═╡ 00dafd6c-34b6-45b4-8f95-37b648900fc9
r = runOptim(tar, rand(q,q,3)*2pi)

# ╔═╡ 287834a1-9afe-49b9-8474-a3c9ce83426f
loss1(DB, BURO)

# ╔═╡ fa7af3d5-e495-4ab1-b152-4d6519eafce5
loss(DB, BURO)

# ╔═╡ c10d6d76-7548-41cf-bc03-40e79b5860be
loss(DB, I)

# ╔═╡ c17be2c4-aa5a-41da-b406-98f8ba4e2501
# Generate random query state w
function genw(q)
	a = rand(1:2^q) # pick an index 
	f = DB[:,a]
	# V = exp(log(rand(rcx, 2^q))/2) 
	V = I
	w = V*f # the query
	fid = norm(f'*w)^2
	return (a,w,fid)
end

# ╔═╡ 883f7938-e022-4b9e-9ad7-b07d640e5706
# average fidelity between query and the answer
@time sum(genw(q)[3] for i=1:10^2)/10^2

# ╔═╡ c2e1ab59-a6b7-43f4-84f4-6e6f7ca9d372
# test QBS method
function test_QBS()
	a, w, fid = genw(q)
	wb = BURO*w
	# i = argmax(abs2.(wb)) # an index with highest probability
	p = abs2(wb[a]) # probability of getting the correct answer
	return p
end

# ╔═╡ d9e57b5a-db4d-4898-972a-a79c62d094ff
# average success probability
@time sum(test_QBS() for i = 1:10^3)/10^3

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Memoize = "c03570c3-d221-55d1-a50c-7939bbd78826"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
RandomMatrices = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Memoize = "~0.4.4"
Optim = "~1.9.4"
RandomMatrices = "~0.5.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "d26063b74121f787828b8edfdce7488b202f9d16"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5c9b74c973181571deb6442d41e5c902e6b9f38e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.12.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d8a9c0b6ac2d9081bf76324b39c78ca3ce4f0c98"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.6"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GSL]]
deps = ["GSL_jll", "Libdl", "Markdown"]
git-tree-sha1 = "3ebd07d519f5ec318d5bc1b4971e2472e14bd1f0"
uuid = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
version = "1.0.1"

[[deps.GSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "56f1e2c9e083e0bb7cf9a7055c280beb08a924c0"
uuid = "1b77fbbe-d8ee-58f0-85f9-836ddc23a7a4"
version = "2.7.2+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "e237232771fdafbae3db5c31275303e056afaa9f"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.10.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomMatrices]]
deps = ["Combinatorics", "Distributions", "FastGaussQuadrature", "GSL", "LinearAlgebra", "Random", "SpecialFunctions", "Test"]
git-tree-sha1 = "359d601ae45b05ea9a53d303dfbf477fcaca4195"
uuid = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
version = "0.5.5"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─2915a690-6f0f-45a6-b5dd-a7cc01241aa7
# ╠═39c52222-4e8e-11ef-0a00-b5808f067508
# ╠═504f61bd-e407-4282-bb5f-abc27c462d36
# ╠═d937c8dd-436b-49f1-af12-d70fbc062ac1
# ╠═cf48bf13-1a31-450e-bc4f-192329f335ed
# ╠═b2efdb0f-9ac2-4145-a5ff-3918abda52af
# ╠═77e3476c-1b06-4a2e-9540-160d037d4a23
# ╠═b16d67d4-a87a-4455-a595-2065f207096b
# ╠═7889445c-6660-4d6d-892b-4ff447da47f5
# ╠═00a5a293-4b8d-4e59-bfc0-4bf7408a4ab8
# ╠═07a3a8c1-5799-4dc1-8023-c792d9624484
# ╠═49abd1cc-eb97-40af-a640-62ac79f59db8
# ╠═af36189a-243c-4695-9b81-8e74d0742f99
# ╠═74acad20-dc7b-48b2-981e-e5d075bd8c92
# ╠═7507e661-1a89-41a2-99d4-a7079adea063
# ╠═a69d389d-8716-47d7-ac95-5506e9cf8aef
# ╠═03bb44f1-01fa-4aec-b9a7-3b7c8dcda9e2
# ╠═994cc9de-482b-4bdf-93cd-9b1dcb9f755f
# ╠═822752c3-1173-422b-9c69-29a1c673adc7
# ╠═00d2e683-258b-4599-a8f3-75a9460b0a40
# ╠═56e2b464-503c-4662-8b83-2f2523c1dfc5
# ╠═c0a32ac7-3b29-497a-b0ec-6928d8484091
# ╠═4cf9bde7-4e46-4181-a88d-ec28387d8f0f
# ╠═5ae9eae2-7db9-4d92-be9a-c3f38585fb07
# ╠═00dafd6c-34b6-45b4-8f95-37b648900fc9
# ╠═287834a1-9afe-49b9-8474-a3c9ce83426f
# ╠═fa7af3d5-e495-4ab1-b152-4d6519eafce5
# ╠═c10d6d76-7548-41cf-bc03-40e79b5860be
# ╠═c17be2c4-aa5a-41da-b406-98f8ba4e2501
# ╠═883f7938-e022-4b9e-9ad7-b07d640e5706
# ╠═c2e1ab59-a6b7-43f4-84f4-6e6f7ca9d372
# ╠═d9e57b5a-db4d-4898-972a-a79c62d094ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
