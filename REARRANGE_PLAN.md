# Rearrange plan — repo layout

Branch `rearrange`. Goal: get the repository root from **27 entries down to 8** by moving
scripts, docs, logs and results into folders. Working document — delete it when the move
lands.

## Target layout

```
.
├── README.md              # stays: GitHub landing page
├── CLAUDE.md              # stays: Claude Code auto-loads it from the root only
├── requirements.txt
├── .gitignore
├── src/                   # 10 .py + run_seed_sweep.sh
├── docs/                  # METRICS.md, wgen_architecture.md,
│                          # INTERPRETATION_i100.md, ideas/
├── runs/                  # all generated results
│   ├── clustering/        # was subspace_kmeans_runs/   (7.5 G)
│   ├── signatures/        # was file_signatures/        (23 M)
│   ├── persistence/       # was persistence/            (614 M)
│   └── forecast_error/    # was forecast_error/         (2.2 M)
├── logs/                  # 6 .log/.out files
├── assets/                # ne_110m_coastline.geojson
├── latents_2/             # data — untouched, gitignored
├── latents_downscaled/    # legacy — untouched, gitignored
└── JL-Downscaling/        # legacy — untouched, gitignored
```

`README.md` and `CLAUDE.md` deliberately stay at the root: GitHub renders the former as the
landing page, and Claude Code only auto-loads the latter from the repo root.

## Why `src/` is flat

The import graph crosses any role-based grouping:

```
cluster_io  <- subspace_kmeans, holdout_eval, file_signature, analyze_clusters
worldmap    <- analyze_clusters, temporal_spatial, analyze_forecast_error
```

`analyze_clusters` needs both, so `src/clustering/` + `src/reporting/` would break all seven
edges and force a package conversion (`__init__.py`, rewritten imports, `python3 -m …` in
every documented command). A **flat `src/`** keeps every `from cluster_io import …` working
untouched, because Python puts the invoked script's own directory on `sys.path`.

**Invariant this creates: scripts are still run from the repo root** —
`python3 src/subspace_kmeans.py`, never `cd src`. Every data-path default is root-relative
(`latents_2`, `runs/…`, the coastline asset), so a different cwd breaks them. This is the
one rule a future contributor has to know; it goes in README + CLAUDE.md.

## Phase 1 — moves + path edits (one commit, so HEAD stays runnable)

### 1a. Moves

All via `git mv`. Verified: `git mv` on a *directory* renames it on the filesystem, so the
gitignored `model.pt` / `assignments.pt` / `err_persist.npy` payloads (~8 G) ride along —
same-filesystem rename, instant, no copy.

| from | to |
|---|---|
| 10 `*.py`, `run_seed_sweep.sh` | `src/` |
| `METRICS.md`, `wgen_architecture.md`, `INTERPRETATION_i100.md` | `docs/` |
| `ideas/` | `docs/ideas/` |
| `seed_sweep.log`, `subspace_run.log`, `subspace_i100_run.log` (tracked) | `logs/` |
| `nohup.out`, `output.log`, `verify_script.out` (ignored) | `logs/` (plain `mv`) |
| `ne_110m_coastline.geojson` | `assets/` |
| `subspace_kmeans_runs/` | `runs/clustering/` |
| `file_signatures/` | `runs/signatures/` |
| `persistence/`, `forecast_error/` | `runs/persistence/`, `runs/forecast_error/` |

`.gitignore` needs no path changes: `nohup.out`, `output.log`, `verify_script.*`, `*.pt` and
`*.npy` are all unanchored patterns, so they keep matching at the new depth.

### 1b. Code edits — argparse defaults

| file:line | from | to |
|---|---|---|
| `worldmap.py:28` | `ne_110m_coastline.geojson` | `assets/ne_110m_coastline.geojson` |
| `temporal_spatial.py:79` | `subspace_kmeans_runs/v6_subspace_big_d64` | `runs/clustering/v6_subspace_big_d64` |
| `file_signature.py:80` | `subspace_kmeans_runs/v6_subspace_big_d64` | `runs/clustering/v6_subspace_big_d64` |
| `persistence_error.py:99` | `persistence/v6` | `runs/persistence/v6` |
| `analyze_forecast_error.py:124-130` | `persistence/v6`, `forecast_error/persist_v6`, 2× `subspace_kmeans_runs/v6…`, 2× `file_signatures/v6_d64/…`, `forecast_error/static_labels.npy` | the `runs/…` equivalents (7 defaults) |
| `extract_forecast_error.py:195` | `err_forecast/run0` | `runs/forecast_raw/run0` |
| `run_seed_sweep.sh:18,21,36,58` | `subspace_kmeans_runs/…` | `runs/clustering/…` |
| `run_seed_sweep.sh:52,53,61,66,67` | `$PY analyze_clusters.py`, `$PY temporal_spatial.py`, `$PY subspace_kmeans.py` | `$PY src/…` |

**Two stale defaults to fix while here** — `subspace_kmeans.py:45 --out` and
`analyze_clusters.py:46 --dir` both default to `subspace_out`, a directory that no longer
exists. Point them at `runs/clustering/`.

### 1c. Code edits — emitted and documented commands

- `analyze_clusters.py:354` writes a `python3 temporal_spatial.py --dir …` line **into every
  generated report** → must become `python3 src/temporal_spatial.py`. Same for the
  `subspace_kmeans.py --files-from …` reproduce command it emits.
- Module docstring usage examples in all 10 scripts (~12 lines, e.g.
  `persistence_error.py:22-23`, `file_signature.py:54-57`, `holdout_eval.py:41`) → prefix
  `src/`.

## Phase 2 — documentation (second commit)

- `README.md` — 55 script refs, 21 output-dir refs; add the "run from the repo root" rule and
  a layout tree near the top.
- `CLAUDE.md` — 33 script refs, 5 output-dir refs; update the `## Code` bullets and the runs
  list, and add the same cwd rule.
- `METRICS.md` — 7 script refs.
- `INTERPRETATION_i100.md` — 1 output-dir ref.

## Phase 3 — regenerate reports (third commit)

The committed `report.md` files carry absolute old paths in their headers and emit the old
reproduce commands, so regenerate them from the moved tree:

```bash
python3 src/analyze_clusters.py --dir runs/clustering/v6_subspace_big_d64 \
    --out runs/clustering/v6_subspace_big_d64/report.md
python3 src/temporal_spatial.py --dir runs/clustering/v6_subspace_big_d64
```

`v9_seed2_d64` has no `model.pt` of its own (it re-reports v8), so regenerate it against
`v8_seed2_d64` or leave it — note which in the commit message.
`runs/signatures/v6_d64/manifest.json` has a baked absolute `model_dir`; it is a provenance
record, not a load path, so it can be left stale or hand-patched.

## Verification (run from repo root, after Phase 1)

| check | cost | proves |
|---|---|---|
| `python3 -c "import ast; …"` over `src/*.py` | instant | nothing syntactically broken |
| `python3 src/analyze_clusters.py --dir runs/clustering/v6_subspace_big_d64 --out /tmp/r.md` then diff vs committed | ~10 s | sibling imports, `model.pt`/`assignments.pt` paths, affinity |
| `python3 src/temporal_spatial.py --dir runs/clustering/v6_subspace_big_d64 --out /tmp/t.md` | ~10 s | `worldmap` import **and** the moved coastline asset |
| `python3 src/analyze_forecast_error.py --out /tmp/fe` | ~1 min | all 7 rewritten forecast defaults resolve |
| `python3 src/file_signature.py --limit 200 --out /tmp/fs` | ~1 min | `latents_2` streaming + GPU path still resolve |
| `bash -n src/run_seed_sweep.sh` | instant | sweep driver parses (a real run is hours — not re-run) |

Not verified by any of the above: `holdout_eval.py` (needs a GPU pass over held-out files —
smoke it with `--num-files 5`) and `extract_forecast_error.py` (supercomputer-only, no
checkpoint on this box — syntax check is all that's possible).

## Risk and rollback

Low. Nothing is deleted, no history is rewritten, and every payload move is a same-filesystem
rename. The whole branch is discardable with `git checkout master && git branch -D rearrange`
— though note that would leave the *working tree* dirs renamed, so recover with
`git checkout master` **before** deleting, or `git mv` them back.

The one thing that can silently break is a path default missed in Phase 1b: it fails at
runtime, not import time, which is exactly what the verification table above is sized to
catch.
