# PCA.md — the update step, from the ground up

`docs/KERNEL.md` explains the **assignment** half of each iteration: given the subspaces,
which cluster does each token belong to. This file explains the **update** half: given the
tokens each cluster now owns, what are the best anchor and the best directions to describe
them. Together they are one iteration of `subspace_kmeans.py`.

Assumes no linear algebra background beyond "a vector is a list of numbers".

---

## 0. The three ideas, with small numbers

Sections 1-10 below are the full version. This section is the same three ideas at their
smallest, for a reader who wants them concrete first.

### (a) Why the anchor is the average

Three points on a line: **1, 2, 6**. Pick one number representing them so the total of
*squared* distances is smallest:

| candidate | squared distances | total |
|---|---|---|
| 2 | 1 + 0 + 16 | 17 |
| **3** | 4 + 1 + 9 | **14**  <- best |
| 4 | 9 + 4 + 4 | 17 |
| 3.5 | 6.25 + 2.25 + 6.25 | 14.75 |

The winner is 3 = `(1+2+6)/3`, the average. Always. Picture the candidate on a see-saw with
a weight at each point: squaring means a point twice as far pulls four times as hard, so the
only place every pull cancels is the balance point. The mean is not an approximation here —
it is the exact minimiser, which is why no iteration is needed for the anchor.

### (b) Why "capture most" and "leave least" are one goal

A token sits 5 units from the anchor. Drop a perpendicular onto the flat:

                        * token
                       /|
            5 units   / |  3 units   <- sticks out sideways (RESIDUAL)
                     /  |
        anchor *----+----            <- the flat
                4 units              <- runs along the flat (CAPTURED)

A 3-4-5 right triangle, so `25 = 16 + 9`.

**The 25 never changes** — it is just the token's distance from the anchor, and the flat's
orientation has nothing to do with it. So 16 and 9 must always sum to 25: rotate the flat to
capture more (16 -> 20) and the leftover shrinks by exactly as much (9 -> 5). Capturing the
most spread and leaving the least residual are the same goal, not a trade-off. This matters
because "capture the most" has an exact solution and "minimise residual" directly does not.

### (c) What an eigenvalue is

Picture a cluster's tokens as a **cigar-shaped cloud** — long in one direction, thin across.
Pick any direction and ask "how spread out is the cloud along this direction?": flatten all
the points onto it and measure. Along the cigar's length, a big number. Across its width, a
small one.

The covariance matrix is nothing more than a **compact device that answers that question for
every possible direction**. Direction in, spread out. (`v^T C v` is just the mechanics of
the lookup.)

Every such cloud has natural axes — the direction it is longest, the direction it is
thinnest — and they are perpendicular.

> **The eigenvectors are those natural axes. The eigenvalues are how much spread sits on
> each one.**

Concretely, six points placed exactly on a 45-degree line:

    (1,1)  (2,2)  (3,3)  (-1,-1)  (-2,-2)  (-3,-3)

- spread along the 45-degree direction: **9.33**
- spread along the perpendicular (135-degree) direction: **0** — every point is exactly on
  the line, so they all flatten onto the same spot

Eigenvalues 9.33 and 0. Which says something real: this "2-dimensional" cloud is actually
**1-dimensional**, and describing it with the single 45-degree direction loses *nothing*.
That is precisely what the algorithm does at 2048 dimensions instead of 2.

Two consequences make the method work:

1. **The spreads sum to the total** (9.33 + 0 = 9.33). Think of a fixed budget of spread
   divided into buckets, one per perpendicular axis.
2. **You cannot beat taking the biggest buckets.** Choosing 64 directions collects 64
   buckets, so the maximum possible is the 64 largest — achieved exactly by the 64
   eigenvectors with the largest eigenvalues.

By (b), collecting the most spread *is* leaving the smallest residual, so the leftover is
just the buckets not taken: `lambda_65 + ... + lambda_2048`.

---

## 1. What the update step has to solve

After assignment, cluster *j* owns some set of tokens. We must produce two things:

- **`mu_j`** — one anchor point, 2048 numbers.
- **`U_j`** — 64 directions ("basis vectors"), each 2048 numbers.

"Best" means the same thing it meant during assignment: make the leftover residual as small
as possible. If assignment picks the nearest flat, the update moves each flat so it fits
its own tokens as tightly as it can.

## 2. The anchor is just the average

Minimise the total squared distance from the tokens to a single point:

    sum over tokens of ||x - mu||^2

The answer is the plain **average** of those tokens. This is the same fact as "the number
minimising the sum of squared differences to a list is the mean" — the balance point. No
iteration needed; the mean is exactly optimal.

In code: `means = msum / n`, where `msum` is the running sum of the cluster's tokens.

## 3. What we want from the directions

Now centre the data — replace each token `x` by `x - mu`, so the cluster sits around the
origin. We want 64 directions that describe this cloud as well as possible.

Here is the fact that makes "as well as possible" precise. Because the 64 directions are
**orthonormal** (mutually perpendicular, each of length 1), Pythagoras applies exactly:

    ||x - mu||^2   =   ||U^T (x - mu)||^2   +   residual
    \_ total ___/      \_ captured ______/       \_ left over _/

The total on the left does **not depend on U at all** — it is fixed once the anchor is
fixed. So the two goals

- *capture as much as possible*, and
- *leave as little residual as possible*

are the **same goal**. Maximise one and you have minimised the other. This matters because
"capture as much spread as possible" turns out to have a clean, exactly solvable answer.

## 4. The covariance matrix, concretely

The covariance matrix `C` is a 2048 x 2048 table built from the centred tokens:

    C = (1/n) * sum over tokens of  (x - mu)(x - mu)^T

Entry `(a, b)` answers: *when dimension a is above its average, does dimension b tend to be
above its average too?* The diagonal entries are just the spread (variance) of each
individual dimension.

The single property that matters here: **for any direction `v` of length 1, the spread of
the data along `v` is `v^T C v`.** So think of `C` not as a table but as a *machine*: feed
it a direction, it returns how spread out the cloud is along that direction. Finding good
directions means finding directions where this machine returns big numbers.

## 5. Eigenvectors: the directions where the machine is simplest

Most directions get scrambled by `C` — you put in `v` and get back some unrelated vector.
But some special directions come back **pointing the same way**, only stretched:

    C v = lambda v

Such a `v` is an **eigenvector** and the stretch factor `lambda` its **eigenvalue**.

For those special directions the spread formula collapses beautifully:

    spread along v  =  v^T C v  =  v^T (lambda v)  =  lambda * (v^T v)  =  lambda

since `v` has length 1. So:

> **An eigenvalue is literally the amount of spread along its eigenvector.**

Because `C` is symmetric (entry `(a,b)` equals `(b,a)` by construction), two further facts
hold, and they are what make the whole method work:

1. Its eigenvectors are **mutually perpendicular** — so they can be used directly as an
   orthonormal basis, no adjustment needed.
2. There are a full 2048 of them, and the total spread splits cleanly across them:
   `trace(C) = lambda_1 + lambda_2 + ... + lambda_2048`.

That second point is the useful mental image: the cloud's total spread is a fixed budget,
divided into 2048 independent buckets, each bucket belonging to one perpendicular
direction.

## 6. Why the top-d eigenvectors are optimal — not just a good heuristic

With the bucket picture, the answer is almost obvious. Choosing 64 orthonormal directions
means choosing how much of the budget to collect. The most you can possibly collect with
64 directions is the **64 largest buckets** — and you achieve exactly that by choosing the
64 eigenvectors with the largest eigenvalues.

This is a theorem (Courant–Fischer / Ky Fan), not an approximation: no other choice of 64
orthonormal directions captures more variance. That is why the code calls a full exact
`eigh` and takes the top 64, rather than running power iterations or a randomised SVD. The
optimal answer is available in closed form, so it takes it.

The residual left over is then the sum of the buckets you did *not* take:

    residual = lambda_65 + lambda_66 + ... + lambda_2048

## 7. What the code actually does

```python
C = S / n - mu mu^T                       # covariance from streamed moments
evals, evecs = torch.linalg.eigh(C)       # exact, ascending
U      = evecs[..., DIM-d:].flip(-1)      # top-d, flipped to descending
eigvals = evals[...,  DIM-d:].flip(-1)
```

Two implementation points worth understanding:

**Streaming moments instead of gathering tokens.** The code never collects cluster *j*'s
tokens into one place. Each GPU accumulates three running totals per cluster while it
sweeps its chunks:

    S    = sum of x x^T       [K, 2048, 2048]
    msum = sum of x           [K, 2048]
    cnt  = number of tokens   [K]

and then uses the standard identity `Var = E[X^2] - E[X]^2` in matrix form:
`C = S/n - mu mu^T`. This is the same spirit as the assignment kernel — **the memory cost
does not depend on how many tokens a cluster owns.** `S` is 2048 x 2048 whether the cluster
holds 200,000 tokens or 2 million: 2.1 GB for all 128 clusters, fixed. Gathering tokens
instead would cost ~12 GB for a single large cluster.

It is also **additive**, which is exactly why two GPUs work: each accumulates its own
partial `S`, and the partials are summed. (They are summed on CPU because peer-to-peer
copies between the two GPUs on this machine silently return zeros — see `CLAUDE.md`.)

The `E[X^2] - E[X]^2` form can lose precision when the mean is large relative to the
spread, since it subtracts two big numbers to get a small one. Here the accumulation is
fp32 and the latents are roughly centred, so it is not a problem — but it is the reason
this formula is not a universally safe default.

**`DIM-d:` rather than `-d:`.** `eigh` returns eigenvalues ascending, so the largest are at
the end. The obvious slice `evecs[..., -d:]` breaks silently for `d = 0` (plain k-means),
because Python reads `-0` as `0` and hands back the *entire* array instead of nothing.
Slicing from `DIM-d` is correct in both cases.

## 8. Why the whole loop is guaranteed to converge

Each half of the iteration can only make the objective smaller or leave it alone:

- **Assignment**: moving a token to the flat with the smallest residual cannot increase
  that token's residual (it kept its old cluster as an option).
- **Update**: for the partition as it now stands, the mean and the top-d eigenvectors are
  exactly optimal, so the objective cannot go up.

The objective decreases monotonically and can never drop below 0, so it must settle. This
is why `report.md`'s convergence table should show a curve that falls and flattens; a rise
would indicate a bug.

**Important caveat:** this guarantees a *local* minimum, not the best possible one. Which
one you reach depends on initialisation. That is precisely why runs v6/v7/v8 repeat the
same configuration with seeds 0/1/2 — the objectives agree to 0.24% and the partitions
agree at NMI 0.72 (against 0.13 for chance), which is the evidence that d=64/K=128 is a
stable basin rather than a lucky start.

## 8b. Is it really optimising the subspace, or just tracking the centre?

A fair suspicion: the kernel is full of centroid-shaped terms, so maybe the subspaces are
decoration on what is essentially k-means. **Measured on v6, they are not.**

Taking 24,576 real tokens (files 5000 and 9000) and assigning them two ways under v6's
trained model — once by nearest centre `||x - mu_j||^2`, once by the full subspace residual:

    agreement between the two rules : 43.0%
    they DISAGREE on                 57.0% of tokens

    mean subspace residual achieved:
      assign by subspace residual : 1883.3    <- what the algorithm does
      assign by nearest centre    : 2262.5    <- the proxy
      the proxy is 20.1% worse

    where the subspace winner ranks by centre distance:
      nearest centre           43.0%
      top-3 nearest centres    68.1%
      top-10 nearest centres   88.0%     (so 12% are not even in the nearest 10 of 128)

The comparison is **generous to the proxy**: it reuses v6's converged means, which were
themselves shaped by subspace-aware assignment. A genuine k-means run would place its
centres elsewhere and do worse still.

*Why they diverge:* think of a cluster as a city (the centre) with roads radiating out (the
64 directions). A token can be far from city A yet sit right on one of A's roads (small
residual, A wins), or close to city B yet out in the fields beside every road (large
residual, B loses). The roads run a long way — the subspaces capture ~70% of each cluster's
spread. This is the same fact as the variance decomposition: only **8.4%** of total variance
separates the centres from one another, while **60.1%** lives in the within-cluster
directions. Judging by centre alone keeps the 8% and discards the informative 60%.

*Why no step is a proxy:* there is one objective, `J = sum over tokens of R_{j(x)}(x)`, and
each iteration minimises **that same J** in two different variables — labels (holding the
flats fixed; exact, by checking all K residuals) then flats (holding labels fixed; exact,
by mean + top-d eigenvectors). This is block coordinate descent on the true objective, which
is why J can only fall and why a rise means a bug.

The contrasting design — run plain k-means, then fit a PCA per cluster afterwards — is a
legitimate algorithm; it is just worse, and 2262.5 is roughly what it buys. The difference
is that here the subspaces **feed back into** assignment, so clusters reshape around shared
directions instead of around proximity to a point.

Honest exception: **iteration 1 genuinely is the proxy**, since the initialisation sets
`U = 0` and the first sweep is literally nearest-centre k-means. The proxy is the starting
point, not the method.

## 9. The re-seed guard, and why the obvious fix fails

A cluster that shrinks below `max(2d, 64)` tokens cannot support a 64-dimensional basis —
you cannot fit 64 meaningful directions through 30 points.

The obvious repair is to restart it from a random token with `U = 0`. **At large d this
provably cannot recover.** A cluster with no basis scores the full `||x - mu||^2` for every
token, while its rivals — each carrying 64 directions — have already subtracted most of
that away. It loses every single contest and stays empty forever. Observed directly: 1
stranded cluster at d=64 (v4), 2 at d=128 (v5).

§8b quantifies exactly how hopeless this is: a `U = 0` cluster is judged by the centre rule
while all 127 rivals use the subspace rule, and the centre rule is **20.1% worse** on the
objective. The handicap is structural, not a matter of tuning — which is why the fix has to
hand the new cluster a real subspace rather than a bare point.

The working fix **splits the largest healthy cluster along its top eigenvector**. Both
halves inherit the parent's basis and are offset by +/-1 standard deviation along that
direction, so they start on opposite sides of the parent's widest axis and the next
assignment divides the parent's tokens between them. Both are immediately competitive
because both already own a real 64-dimensional subspace. v6 keeps all 128 clusters healthy,
minimum size 227,886.

## 10. What the eigenvalues tell you afterwards

They are saved in `model.pt` and drive several reported numbers:

- **`EVR`** = `sum(eigvals) / trace` — the fraction of a cluster's own spread its subspace
  captures. v6 median ~0.70.
- **`d80`** — how many leading directions are needed to reach 80% of the captured spread.
  v6 sits at 28-40 for every cluster, i.e. the spectrum is genuinely flat-ish; there is no
  cluster secretly describable in 5 dimensions.
- **`sigma2`** (soft mode only) = `(trace - sum eigvals) / (DIM - d)` — the *average*
  discarded bucket. This is exactly the maximum-likelihood noise estimate for probabilistic
  PCA, which is what lets the same eigendecomposition serve as the MPPCA M-step with no
  extra work. See `docs/PIPELINE.md` §2.
