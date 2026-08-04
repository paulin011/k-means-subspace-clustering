# KERNEL.md — the matmul trick, explained slowly

This is the one piece of the codebase that reads as if it does something different from
what it actually does. The code at `subspace_kmeans.py:148-166` (and its verbatim copies in
`holdout_eval.py` and `file_signature.py`) *looks* like it only measures distance to
centroids. It is in fact the exact orthogonal residual to each cluster's subspace. This
file explains why, without assuming much maths.

## First: what `U` actually is

A recurring first guess is that a subspace means "keep 64 of the 2048 coordinates" — 64
slots set to 1, the rest 0. It does not. Here are the real numbers from v6, cluster 0,
first basis vector:

    first 8 values : [0.0023, 0.0332, 0.0005, 0.0099, 0.0255, -0.0819, 0.0062, 0.0023]
    min / max      : -0.3065 / 0.4224
    exactly zero   : 0 of 2048
    length         : 1.000000

So each basis vector is a **dense list of 2048 real numbers**, positive and negative, none
of them zero. The only constraints are that each has length exactly 1 and that the 64 are
mutually perpendicular (measured on v6: `U^T U` has diagonal 1.000000 and off-diagonal at
most 1e-6).

Shape bookkeeping, since two numbers get confused:

    U shape = [128, 2048, 64]
               |     |    |
               |     |    +-- d = 64  basis vectors per cluster
               |     +------- 2048    length of each vector (embedding dim)
               +------------- K = 128 clusters

**Why "any direction" rather than "pick coordinates" matters.** Suppose data lies exactly
along a 45-degree line in 2D. Restricted to coordinate axes you need *both* x and y to
describe it — neither alone captures the line. Allowed to choose any direction, one
suffices and the residual is zero. In 2048 dimensions this is the whole ball game: real
structure is essentially never aligned with the raw coordinate axes. A basis vector here is
a **blend** — "0.0023 of dim 0, plus 0.0332 of dim 1, minus 0.0819 of dim 5, ..." — a new
direction built out of all 2048 originals.

The directions are not uniform across coordinates, though: on v6 the top 4 coordinates
carry 50% of a direction's weight and 221 carry 90%. So the latent space does have
dominant dimensions — but an axis-aligned direction would have **1** coordinate at 100% and
2047 exact zeros, which is nothing like what is there.

Geometrically the subspace is `mu_j + span(U_j)`: every point reachable by starting at the
anchor and moving any amount along any mix of those 64 directions — a 64-dimensional flat
slab living inside 2048-dimensional space. A token's residual is how far it sits *off* that
slab.

## What we need to compute

For every token `x` (a list of 2048 numbers) and every cluster `j` (of which there are
K=128), we want one number: **how far is this token from cluster j's flat?**

The flat is "anchor point `mu_j`, plus every direction you can build from the 64 basis
vectors in `U_j`". The distance to it is

    R_j(x) = ||x - mu_j||^2  -  ||U_j^T (x - mu_j)||^2

In words: *take the whole distance from the token to the anchor, then subtract off the part
that runs along the flat's own directions.* What survives is the part sticking out
perpendicular to the flat — the residual. If a token lies exactly on the flat, the two
terms are equal and R = 0.

## Why the obvious way is impossible

The obvious way is to literally follow the formula: compute `x - mu_j`.

Do that for a batch of B = 262,144 tokens against all K = 128 clusters and you have created
262,144 x 128 x 2048 numbers. At 4 bytes each that is **275 GB** — for one batch, on a 40 GB
GPU. Dead on arrival.

Notice the actual answer we want is tiny: one number per (token, cluster) pair, so
`[262144, 128]` = 134 MB. The 275 GB is entirely scratch space we throw away. The trick is
to never create it.

## The key algebraic fact

Everything follows from one identity you already know in the one-dimensional case:

    (a - b)^2 = a^2 - 2ab + b^2

The vector version is identical:

    ||x - mu||^2 = ||x||^2 - 2 (x . mu) + ||mu||^2

where `x . mu` is the dot product (multiply the 2048 pairs, add them up).

**Why this is the whole trick:** look at what each of the three pieces depends on.

| piece | depends on | how many values for the whole batch |
|---|---|---|
| `\|\|x\|\|^2` | the token only | B — one per token |
| `\|\|mu\|\|^2` | the cluster only | K = 128 — one per cluster, computed once |
| `x . mu` | both | B x K — but this is exactly a matrix multiply |

The left-hand side `||x - mu||^2` mixes token and cluster together, which is why computing
it directly forces you to build a separate vector for every pair. The right-hand side
**separates** them: two cheap lists, plus one cross-term. And the cross-term — "dot every
token with every cluster mean" — is the literal definition of a matrix multiply. GPUs do
nothing faster.

    X @ means.T      # [B, 2048] x [2048, K]  ->  [B, K]

One matmul, and we have all B x K dot products. No 275 GB tensor was ever created, because
`||x - mu||^2` was never formed as a vector — only its *length*, which is all we wanted.

## A tiny worked example

Two dimensions, so you can check it by hand. Token `x = (3, 4)`, anchor `mu = (1, 2)`.

Direct: `x - mu = (2, 2)`, so `||x - mu||^2 = 4 + 4 = 8`.

Expanded:

    ||x||^2   = 9 + 16 = 25
    x . mu    = 3*1 + 4*2 = 11
    ||mu||^2  = 1 + 4 = 5
    25 - 2*11 + 5 = 25 - 22 + 5 = 8      <- same answer

Same number, but the second route never built `(2, 2)`. With 128 anchors you would reuse
that one `||x||^2 = 25` for all of them, and get all 128 dot products from a single matrix
multiply.

## Now the second term

The same move works on the projection term `||U_j^T (x - mu_j)||^2`.

`U_j^T (x - mu_j)` is the token's coordinates *inside* the flat — 64 numbers instead of
2048. Call it `z`. Distributing the multiplication:

    z = U_j^T x  -  U_j^T mu_j

The second half, `U_j^T mu_j`, doesn't involve the token at all — it is a fixed `[K, 64]`
table computed once per iteration (called `c` in the code). The first half, `U_j^T x`, is
again just a matrix multiply — and here is the neat part: all K bases are **glued
side-by-side** into one wide matrix so a single multiply handles every cluster at once:

    U_cat = [U_0 | U_1 | ... | U_127]      # [2048, K*64] = [2048, 8192]
    P = X @ U_cat                          # [B, 8192], then viewed as [B, K, 64]

Then apply `(a-b)^2 = a^2 - 2ab + b^2` one more time, to `z = P - c`:

    ||z||^2 = ||P||^2  -  2 (P . c)  +  ||c||^2
              \_ pe _/     \_ pc _/     \ cnorm /

## Putting it together

That is every line of the kernel:

    R = ( xnorm - 2*xm + mnorm )  -  ( pe - 2*pc + cnorm )
        \___ ||x - mu_j||^2 ___/     \__ ||U_j^T(x-mu_j)||^2 __/

Each of the six terms is either a `[B]` list (token-only), a `[K]` list (cluster-only,
computed once), or a `[B, K]` table produced by a matrix multiply. Peak memory is the
`[B, K, 64]` projection `P`, not `[B, K, 2048]` — a **32x** reduction, which is what makes
the whole thing fit.

## The point that causes confusion

Read the code quickly and you see `X @ means.T`, squared norms of means, and a `min()`.
That is the visual signature of **k-means** — distance to centroids. It is easy to conclude
the subspaces are not being used in the assignment at all.

They are. The `pe - 2*pc + cnorm` group is the subspace's contribution, and subtracting it
is what turns "distance to the anchor point" into "distance to the whole flat". The
subspace is in there; it just does not look like it, because it has been algebraically
flattened into the same matmul-plus-lookup shape as everything else.

The `d = 0` case makes this concrete: with no basis, `pe`, `pc` and `cnorm` are all zero,
the second group vanishes, and the formula collapses to exactly k-means. K-means is not a
different algorithm here — it is this same kernel with the subspace term switched off.

## Where else this appears

The identical kernel is reused verbatim in three more places, which is why a change here
must be mirrored (or better, why none of them re-derive it):

- `holdout_eval.py` — same formula, tokens from unseen files.
- `file_signature.py` — same formula, all 13,021 files, reduced per file instead of per set.
- the `--soft` MPPCA path — reuses `R` and `z^2` directly, which is exactly why the
  likelihood costs only two extra einsums (see `docs/PIPELINE.md` §2).
