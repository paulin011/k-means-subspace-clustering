What the algorithm actually computes (needed to read anything below)
The model is K-subspaces clustering — k-means where each "centroid" is not a point but an affine subspace. Cluster j is defined by a mean μ_j (a point in 2048-dim space) plus a d=32-dimensional flat through it, spanned by orthonormal basis U_j (2048×32).

Assignment: each token x goes to the cluster whose subspace it sits closest to, measured by orthogonal residual — the part of x−μ_j that the 32 basis directions can't reconstruct: ‖x−μ_j‖² − ‖U_jᵀ(x−μ_j)‖².
Update: for each cluster, recompute μ_j and the 2048×2048 covariance of its tokens, then take the top-32 eigenvectors (exact PCA via eigh) as the new basis U_j.
So the deliverable is genuinely subspace bases per cluster, not just labels — which is the stated project requirement.

The two binary files in detail
model.pt — the result
A dict (load with torch.load(..., weights_only=False)):

U [128, 2048, 32] — the subspace bases. U[j] has orthonormal columns ordered by descending eigenvalue (1st column = direction of most variance within cluster j). This is the headline output.
means [128, 2048] — μ_j, the affine offset of each subspace (this run is affine, linear=False).
eigvals [128, 32] — variance captured along each of the 32 basis directions, per cluster.
trace [128] — total within-cluster variance (sum of all 2048 eigenvalues). eigvals.sum(1) / trace = fraction of cluster variance the 32-dim subspace captures = the EVR column.
counts [128] — tokens assigned to each cluster.
explained_var_ratio [128] — that EVR per cluster, precomputed.
config — the argparse namespace (see the caveat below).
history — list of per-iteration dicts (objective, fraction of labels changed, sizes) → the Convergence table.
sampled_files + sample_fingerprint — the 7000 file ids and the 82ca602ed7e7 hash that marks comparable runs.
To use a basis: project a token with (x - model['means'][j]) @ model['U'][j] → a 32-dim coordinate.

assignments.pt — the labels
Three parallel int32 arrays, one entry per sampled token (86,016,000 total):

file_id — which latent file the token came from (0…13020).
cell_id — which of the 12288 HEALPix cells (NESTED ordering) — i.e. its geographic location.
label — which cluster (0…127) it was assigned to.
This is what lets the report tie clusters to geography (via cell_id) and time (via file_id, since file index ≈ time step).

sample.json — the manifest
{fingerprint, num_files: 7000, tokens_per_file: 12288, seed: 0, src, files: [...7000 ids...]}. Its only job is reproducibility: --files-from subspace_kmeans_runs/v3_subspace_big_i100/sample.json re-clusters the identical token set under a different K or d for a fair comparison.

Reading the report section by section
Header & Configuration
86,016,000 tokens = 7000 files × 12288 cells. One caveat to flag: the Configuration table shows num_files = 1500, but that's the stale argparse default — this run was driven by --files-from, so the real count is the 7000 in the Token-sample section (and the token total confirms it: 7000×12288 = 86M, not 1500×12288). Trust the Token-sample section, not the num_files row.

Convergence table
Four things to watch as iterations climb:

objective/token — mean squared orthogonal residual. Drops from 7659 → 3093 in one step (the first real subspace update), then crawls: 2698 at iter 10, 2672 at iter 100. The extra 90 iterations bought almost nothing (2698→2672, <1%). This is the key finding of the i100 experiment: the default run length was already essentially converged.
labels changed — fraction of tokens that switched cluster. Falls to 5% by iter 10 and 0.37% by iter 100 — never quite zero, so a small set of boundary tokens keeps oscillating between near-equidistant subspaces (normal, harmless).
min/max size — largest cluster is ~6× the smallest at convergence (1.39M vs 0.23M). Moderately balanced; no cluster collapsed to empty.
Global variance decomposition
Total token spread (5998) splits three ways:

10.2% between clusters — how much is explained just by which cluster a token belongs to (the means). Low, which is expected: weather tokens don't separate into far-apart blobs.
45.3% within clusters, top-32 directions — what the subspace bases capture. This is where the model does its real work.
44.5% residual — unexplained. Large, but expected for 32 dims out of 2048.
The line "Dimensions needed for 80%… max 22 (close to 32 ⇒ flat spectrum, consider larger --dim)" is the actionable hint: each cluster's spectrum is fairly flat, so d=32 is truncating real structure — bumping --dim would capture more. (This is not something more iterations fixes, which is why i100 didn't help here.)

Clusters table (sorted by size)
Per cluster:

share — % of tokens.
EVR(top-32) — fraction of that cluster's variance its 32-dim basis captures. Higher = the cluster is genuinely lower-dimensional / more coherent (e.g. cluster 123 at 0.657).
d80 — dims needed for 80% of captured variance. Near 32 = flat spectrum (under-dimensioned).
cells@50% — how many distinct HEALPix cells hold half the cluster's tokens. Low = geographically localized (e.g. cluster 28 at 18 cells); high = spread across the globe.
owned — cells where this cluster is the most common label (its "territory" on the map).
files — share of the 7000 time steps where the cluster appears. ~100% = present at all times.
tCV — coefficient of variation of the cluster's share across 10 time-bins. 0 = perfectly steady in time; high = bursty/seasonal.
Interpretation pattern (also in the report's footer): low cells@50% + files≈100% + tCV≈0 ⇒ a stable geographic regime (a region or surface type that's always present). High tCV ⇒ a temporal/seasonal signature.

Subspace affinity between clusters
Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / 32 ∈ [0,1] = mean squared cosine of principal angles between two subspaces. 1 = same subspace, 0 = orthogonal. Off-diagonal median 0.352 means clusters are moderately distinct, not redundant. The top pairs (e.g. 17↔50 at 0.733) are the most overlapping clusters — candidates for merging if you wanted a smaller K. The "mean-vector cosine" column separates two reasons clusters can be similar: similar orientation (subspace affinity) vs. similar location (mean cosine). Two clusters can share a subspace orientation while sitting in different places.

Most time-varying clusters
The decile table shows each cluster's enrichment across the dataset's time axis (D0…D9, file index 0→13020). 1.00 = the cluster's own average rate. A value like cluster 126's D9=1.33 means it's 33% over-represented in the last tenth of the timeline. These smooth-ish profiles are the candidate seasonal/trend signatures. The tCV values here (0.26 max) are modest — no cluster is wildly seasonal, consistent with ERA5 tokens being dominated by stable geography rather than time.

World map (dominant_cluster_map.png)
Each of the 12288 cells is colored by its dominant cluster, in NESTED HEALPix ordering (confirmed correct because continents read as coherent regions; RING ordering would produce stripes). Colors come from a spectral ordering of the affinity matrix, so subspace-similar clusters get similar hues — real geographic regions appear as smooth color gradients, while genuine noise stays speckled. At K=128 some apparent "scatter" is just colormap aliasing (too many clusters, too few distinguishable hues), not a sign of bad clustering — per the README, neighbor-agreement metrics actually improve with K.

Bottom line for this specific run
v3_subspace_big_i100 exists to answer one question: does running longer help? The convergence table says no — the objective is flat after ~10–15 iterations (2698 → 2672 over 90 more iterations). The real lever for capturing more structure is larger --dim, flagged by the flat per-cluster spectra (d80 ≈ 20–22 against d=32), not more iterations. Compare it directly against v2_subspace_big/ (same sample, fewer iters) to confirm the two converged to essentially the same solution.

Want me to actually load model.pt/assignments.pt and verify any of these numbers, or diff this run against v2_subspace_big to confirm they converged to the same place?

