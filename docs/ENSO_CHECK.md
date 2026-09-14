# ENSO visibility check (Niño 3.4 box) — 2026-09-14

One-off measurement backing the El Niño claim in `report/report.tex` §3.5. No
script ships for this; the recipe below reproduces it in ~2 min from the
existing signature-run arrays (no pass over `latents_2`).

## Method

- Region: **Niño 3.4** = 5°S–5°N, 170°W–120°W → **157 NESTED cells** at
  nside=32 (`worldmap.healpix_nest2ring` + `healpix_ring_lonlat`, lon wrapped
  to ±180).
- Per-timestep regional anomaly: `z = (residual_map[t, cells] −
  cell_mean_residual[cells]) / cell_std_residual[cells]`, averaged over the
  box (arrays from `runs/signatures/v6_d64_margin/`; this is the
  mean/std normalisation stored in `signatures.npz`, not the probe's
  median/MAD z — fine here because only winters are compared to each other).
- Winters = DJF, December grouped with the following Jan/Feb (repo
  convention); dates reconstructed as `2014-01-01 + idx×6h`.
- Occupancy = share of the box's cells carrying the box's overall most common
  label (cluster 105, 25.8% of box tokens), from `label_map.npy`.

## Result

| winter | mean z | occ(c105) | ONI phase |
|---|---|---|---|
| 2014/15 | +0.16 | 0.227 | +0.7 weak El Niño |
| 2015/16 | **+0.58** | **0.185** | **+2.6 strong El Niño** |
| 2016/17 | +0.12 | 0.269 | −0.4 weak La Niña |
| 2017/18 | +0.18 | 0.243 | −0.9 La Niña |
| 2018/19 | **+0.55** | 0.271 | +0.8 weak El Niño |
| 2019/20 | **+0.60** | 0.224 | +0.5 weak El Niño |
| 2020/21 | +0.12 | 0.235 | −1.3 La Niña |
| 2021/22 | +0.52 | 0.225 | −1.0 La Niña |

Highest-z single months: 2020-01 (+0.86), 2016-01 (+0.84), 2019-01 (+0.75),
2020-02 (+0.72), 2019-02 (+0.63), 2018-02 (+0.62).

## Reading, and the honest limits

- Three of the four most anomalous winters are exactly the three later
  El Niño winters (2015/16, 2018/19, 2019/20) at z 0.55–0.60, vs 0.12–0.18
  for four of the five others. During the record 2015/16 event the box's
  usual dominant cluster drops to its nine-year occupancy minimum (0.185 vs
  0.224–0.271 elsewhere).
- **Caveat kept out of the 2-page report:** the La Niña winter 2021/22 also
  scores +0.52, so the separation is strong but not perfect — an unsigned
  anomaly score can flag either ENSO phase. The weak 2014/15 El Niño does not
  stand out.
- A label-composition test (total-variation distance of the winter regime mix
  from the 9-yr mean) is noisier: 2019/20 highest (0.087), 2015/16 second
  tier (0.064), no clean phase separation. The z signal, not the mix shift,
  is what carries the ENSO claim.
