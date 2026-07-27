# using CUDA 
using Ket
using Base.Threads
using Combinatorics
using Chain 

# CUDA.allowscalar(true)

"""
This is basically a recursive tensor contraction of a vector. Can be useful to traverse many different combinations, 
such as <ϕ|⋅|0>|+>|0>...|1>|+>|0> for all possible combinations of 0,1,+ with a fixed number of '+'s. 

Arguments:
p - array of numbers of size 2^q
i - current index to fill in the mask
k - number of indices left to fill with 0 or 1
m - current mask (init to ['+', '+', ..] before use!)

Returns: 
The largest contracted value for all possible combinations of k indices left to fill, 
along with the best mask that achieves it.
"""
function ss(p, i, k, m; top = 1)
    n = length(p)
    Q = length(m)

    # @show i, k, m
    if k == 0 
        for j=i:Q
            m[j] = '+' # needed to fix the mask
        end
        return sum(p), copy(m)
    end 



    b1,b2,b3 = [],[],[]

    # set 0
    m[i]='0'
    # T1 = @spawn ss(view(p, 1:n÷2), i+1, k-1, copy(m))

    # t = Q÷2-2
    t = 8
    if k>t
        T1 = @spawn ss(p[1:n÷2], i+1, k-1, copy(m), top = top)
    else 
        b1 = ss(view(p, 1:n÷2), i+1, k-1, m, top = top)
    end
    # b[1], bm[1] = ss(p[1:n÷2], i+1, k-1, m)

    # set 1
    m[i]='1'
    # T2 = @spawn ss(view(p, n÷2+1:n), i+1, k-1, copy(m))
    if k>t 
        T2 = @spawn ss(p[n÷2+1:n], i+1, k-1, copy(m), top = top)
    else 
        b2 = ss(view(p, n÷2+1:n), i+1, k-1, m, top = top)
    end
    # b[2], bm[2] = ss(p[n÷2+1:n], i+1, k-1, m)

    if 2^k<n # you have to select either 0 or 1 if k==q
        # set +
        m[i]='+'
        b3 = ss(p[1:n÷2]+p[n÷2+1:n], i+1, k, copy(m), top = top)
        # b[3], bm[3] = ss(p[1:n÷2]+p[n÷2+1:n], i+1, k, m)
    end

    if k>t 
        b1 = fetch(T1) 
        b2 = fetch(T2) 
    end
    # b[3], bm[3] = T3
    
    b = vcat(b1,b2,b3)
    @chain b sort!(lt=(x,y)->x[1]<y[1], rev=true)
    return b[1:min(top, length(b))]
end



# function genV(q)
#     q2 = q÷2
#     m = binomial(q, q2)
#     ret = zeros(2^q, m)
    
#     for i = 1:m 
#         v = 1
#         for j = 1:q
#             w =  
#             v = kron(v, )

#         end
#     end
#     return ret 
# end
# function ssc(p)
#     n = length(p)
#     q = @chain n log2 Int
#     q2 = q÷2

#     combs = combinations(1:q, q2)
#     m = length(combs)

#     V = zeros(Int, )
#     for c in combs

#     end
# end
