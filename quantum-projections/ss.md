# Description of ss.jl

# `ss` — Recursive Product-State Search

This is a Julia implementation of a recursive search over quantum product states, using a binary-tree traversal of a state vector.

## What it computes

Given a quantum state vector `p` (size `2^Q`), it searches over all product states of the form `|b₁⟩⊗|b₂⟩⊗...⊗|b_Q⟩` where each `bᵢ ∈ {0, 1, +}` and **exactly `k` of the qubits are constrained to be `0` or `1`** (the rest are `+`). It returns the **top** product states that maximize the overlap `⟨ϕ|product_state⟩`.

## How the recursion works

The state vector is treated as a binary tree. At each level `i` (one qubit), three branches are explored:

| Choice | Mask char | Action on vector | Decrement `k`? |
|--------|-----------|------------------|----------------|
| qubit = `0` | `'0'` | take first half `p[1:n÷2]` | yes (`k-1`) |
| qubit = `1` | `'1'` | take second half `p[n÷2+1:n]` | yes (`k-1`) |
| qubit = `+` | `'+'` | merge halves: `p[1:n÷2]+p[n÷2+1:n]` | no (stays `k`) |

The `'+'` branch is only taken when `2^k < n`, i.e. there's still "room" — if every remaining qubit *must* be 0/1 (`k` equals remaining qubits), `'+'` is skipped.

## Base case (`k == 0`)

All remaining qubits are forced to `'+'`, and the contraction value is just `sum(p)` (since `'+'` = summing both halves). Returns `(value, mask)`.

## Threading

- When `k > t` (`t = 8`): the `'0'` and `'1'` branches are dispatched to other threads via `@spawn`, and use **copies** of `p` and `m` (to avoid races).
- When `k ≤ t`: everything runs synchronously using zero-cost **views** and in-place mask mutation.

## Aggregation

The three branches' results are concatenated, sorted descending by contraction value, and trimmed to the best `top` entries — so each call returns an array of up to `top` `(value, mask)` tuples.

## Use case

Finding optimal measurement bases or product-state approximations that maximize overlap with a given state, where you fix how many qubits are measured (0/1) vs. left in superposition (+).

## Note

The base case returns a bare tuple `(sum(p), copy(m))` while the recursive case returns an array. If `top > 1`, the `vcat(b1,b2,b3)` call could behave unexpectedly since one branch may yield a tuple instead of an array. Wrapping the base-case return in `[ (sum(p), copy(m)) ]` would ensure consistency.
