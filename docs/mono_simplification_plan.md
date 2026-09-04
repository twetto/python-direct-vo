# Simplification Plan: Monocular VO

**Status:** proposed (2026-09-03)
**Owner:** monocular (`--mono`) mode

## Guiding principle: subtract, then measure

The three symptoms being fought in mono mode —

1. trajectory drifts / scale wrong,
2. loses tracking / diverges,
3. too slow / can't debug —

are all downstream of one root cause: **five competing pose estimators + three
pose-only BAs + ~40 hand-tuned gates**, so no single component is ever provably
correct. When it drifts you can't tell which estimator handed off the bad pose,
so the response has been to add another metric and another gate (see the ~50
`diagnostics_mono_*.jsonl` scratch files and the ~70-field per-frame JSONL
record — the signature of debugging-by-instrumentation).

The fix is to collapse to **one backbone with one joint BA**, and to build a
scale-aware evaluation harness *first* so every later change is measured, not
guessed.

---

## Target architecture (what "done" looks like)

```
bootstrap (once)      ->  two-view essential (and homography) + triangulate -> initial (metric-arbitrary) map
track (every frame)   ->  ONE method: direct photometric frame-to-map
depth of new points   ->  ONE estimator: Sparse3D filters IMMATURE points only, then hands off on promotion
map (on keyframe)     ->  ONE joint local BA (pose + inverse-depth together, sliding window)
keyframes             ->  ONE insertion rule + ONE marginalization rule (no rescue / no usefulness-EMA)
```

Two rules make the whole thing tractable:

1. **Single owner per quantity.** Pose is owned by the tracker + BA. Structure of
   a *mapped* point is owned by BA. Structure of an *immature* (not-yet-mapped)
   point is owned by Sparse3D. Nothing is estimated by two subsystems at once.
2. **One estimator, one job, one handoff.** Sparse3D converges a depth ->
   promotes the point into the map -> BA owns it from then on. No feedback loops.

Delete: the essential-per-frame guess, the flow x assumed-depth fallback as a
*runtime* estimator, the SVO patch-match + PnP candidate, the 6x
recovery-rotation guesses, 2 of the 3 BAs, the KLT landmark re-triangulation, the
keyframe usefulness-EMA layer, and the landmark-rehosting rescue.

---

## Phase 0 - Evaluation harness (do this BEFORE touching estimators)

You can't fix drift/scale you can't measure. Alignment today is done only for
*visualization* (`umeyama_alignment` forces `s = 1.0`). Add a real metric.

- New `src/mono_eval.py`: after a run, compute **Sim(3)-aligned ATE** (Umeyama
  *with* scale free) and report the recovered **scale factor `s`** and its drift
  over time.
- Reduce the ~70-field JSONL to ~6: `ate_rmse`, `scale_ratio_vs_gt`,
  `tracked_inliers`, `ba_cost_before`, `ba_cost_after`, `n_keyframes`.
- Add one integration test: "mono on first N frames of V1_01 keeps
  `scale_ratio` in [0.7, 1.3] and ATE < X." This becomes the regression gate.

**Targets symptom #3:** one number tells you if a change helped.

**Status: DONE (2026-09-03).** `src/mono_eval.py` (Sim(3) Umeyama, scale-corrected
ATE, scale-ratio, scale-drift, stall count) + `tests/test_mono_eval.py` (9 tests).
`main.py` now logs `gt_position` and prints an end-of-run `[mono-eval]` summary;
`python -m src.mono_eval <run.jsonl>` evaluates a saved run.

**Baseline to beat** (current unrefactored mono, V1_01_easy, 250 frames):

```
ATE(sim3) = 0.337 m over a 2.20 m GT path  (~15%)
scale_vs_gt = 0.414                          (~2.4x under-scale — matches SS X systematic under-scale)
scale_drift_spread = 23x, log_std = 0.96     (scale is not remotely consistent)
stall_windows = 7 / 25                        (tracking stalls — "loses tracking / diverges")
```

Every later phase is judged against these four numbers.

## Phase 1 - Single bootstrap, scale fixed once

- Keep `_anchor_essential_pose_guess`'s essential-matrix logic, but
  **triangulate** the inlier correspondences into the initial map right there,
  and **normalize scale once** (e.g. median scene depth = 1.0, or median
  baseline).
- **Delete** the ongoing `image_motion_fallback_depth = 4.0` /
  `_point_flow_step_scale` / `_bootstrap_step_scale` machinery as *runtime*
  scale sources. Assumed-depth flow is fine as a one-time bootstrap seed, never
  as a per-frame scale injector.

**Targets symptom #1:** scale enters the system exactly once.

**Status: DONE (2026-09-04).** Replaced the incremental essential+Sparse3D bootstrap
with a single `_try_two_view_bootstrap` (anchor<->current essential + `cv2.triangulatePoints`,
cheirality, median-parallax gate, scale normalized once so median anchor depth = 1.0),
seeding KF0 directly. Gates aligned to **SVO Pro's `initialization.cpp`** contract:
median-disparity gate `bootstrap_min_flow_px = 30` (their EuRoC `init_min_disparity`)
and a `bootstrap_min_triangulated = 40` floor (their `init_min_inliers`). Removed
`_anchor_essential_pose_guess`; the assumed-4m-depth scale is no longer in the live
path (`_point_flow_step_scale`/`_bootstrap_step_scale`/`_image_motion_pose_guess`
remain only for the Phase-2 fallbacks + their unit tests, dead in the pipeline).
Rewrote the 2 pinned bootstrap tests to the new contract; 87 tests pass.

Results (V1_01_easy, 250 frames): bootstrap fires at frame 92 with **185 landmarks**
(was 38), **77%** of post-bootstrap frames track directly (was 0%). ATE 0.337 -> **0.217 m**
(-36%), scale-drift spread 23x -> **5.2x**, log_std 0.96 -> **0.46**.

Newly exposed bottleneck (deferred to 3B/3C, as predicted): `mono_keyframes` stays
at 1 and the map is frozen at 185 landmarks — no keyframes are inserted post-bootstrap,
so the map goes stale as the camera moves (residual ~5x drift + 8 stalls).

## Phase 2 - One tracker: direct photometric frame-to-map

- Keep `_track_direct_candidates` but reduce `guesses` to a **single** entry:
  the constant-velocity motion model (`self.T_W_C @ self.last_direct_delta`),
  falling back to hold. Drop `direct_recovery` (6 guesses), `svo_match`, and the
  essential/flow guesses.
- Delete `_matched_observation_pose_guess`, `_match_visible_reference_patches`
  (the Python `du/dv` x `map_coordinates` loop - the **perf killer**), and the
  `_keyframe_klt_*` pose-guess paths.
- Collapse the gate thicket to two checks: `inliers >= min_direct_inliers` and
  `_direct_step_is_plausible`. Delete the KLT-residual / flow-cos scoring, the
  per-candidate `reject_reason` bookkeeping, and the ~15 associated thresholds.

**Targets symptoms #2 and #3:** one code path to reason about; recovery comes
from BA + relocalization later, not from spraying guesses.

## Phase 3A - One *joint* local BA

This is the real fix for drift. Today all three BAs freeze half the state:

- `optimize_mono_pose_window` - pose-only, landmarks fixed
- `optimize_mono_geometric_pose_window` - pose-only, points fixed
- `optimize_mono_inverse_depth_window` - depth-only, **poses fixed**

Alternating pose-only and depth-only over a 5-KF window with no joint solve
**is** a drift generator. Replace all three with **one** local BA that optimizes
poses *and* inverse-depths jointly (reprojection residuals, oldest KF as gauge,
Huber). Keep `optimize_mono_inverse_depth_window`'s ray parameterization as the
structure block; run pose and structure in the same normal equations instead of
ping-ponging.

**Targets symptom #1:** joint structure+motion is what keeps mono
self-consistent.

### Covariance-preserving handoff (2026-09-04) — the missing ingredient for 3A

The Stage-1 regression traced to this: on promotion the filter's 5x5 EKF covariance
is **discarded** and the map landmark gets a uniform 2-D pixel cov
(`np.eye(2)*sigma_pixel^2`); the only BA consumer (`optimizer.py:598`) weights
reprojection by that uniform cov *with points fixed*. So a shaky just-promoted point
and a converged one are trusted equally -> growing the map injects scale-inconsistent
structure -> drift regressed 5.2x -> 27x.

Fix (DSO activation model): carry the filter's uncertainty across the handoff and let
BA weight by it.
1. **Smooth transfer.** On promotion compute the filter's 3-D point covariance
   (`J = anchor_point_and_jacobian()`; `P3_anchor = J @ P[:3,:3] @ J^T`; rotate to
   world) and store it on the map landmark as a **prior**, not a fixed truth.
2. **Joint, covariance-weighted BA.** Optimize poses + points together; each point is
   regularized by its transferred prior — tight prior for well-triangulated points,
   loose for fresh ones, which get gently incorporated instead of corrupting scale.

This also softens the retirement sawtooth: points can be promoted earlier with honest
(loose) covariance and tightened by BA, so promotion no longer needs tight EKF
convergence. **Coupling note:** BA must run **every keyframe** (not every 3) for this
to take effect while the window is small.

**Progress (2026-09-04):**
- **Step 1 DONE — covariance-preserving handoff.** `MonoLandmarkTrack.point_covariance`
  (world 3x3), set at promotion in `_store_promotion_covariances` from the filter's
  `anchor_point_and_jacobian` + state covariance, before retire. Verified sane
  (0.0003 vs 0.77 world-cov trace for high- vs low-parallax). 88 tests pass. No
  behavior change on its own (nothing consumes it yet).
- **Step 2 TODO — the joint covariance-weighted BA is a real subsystem, not a tweak.**
  Confirmed the cheap routes fail: sweeping BA frequency (`ba_every_keyframes` 3->1)
  regressed ATE 0.202 -> 0.304 (the alternating pose-only/depth-only BAs shuffle error
  rather than jointly solve); parallax threshold (2-8 deg) and insertion motion
  (0.12->0.015) leave KFs pinned at 2. A naive scipy joint BA over ~450 points is too
  slow (numerical Jacobian, ~1400 params). Correct implementation: analytic-Jacobian
  Gauss-Newton with **Schur complement** over poses + points, points regularized by
  the transferred prior, run every keyframe. Deferred to a focused, carefully-verified
  pass (a subtle bug here corrupts everything).
- **Default pipeline is a WIP regression** until Step 2 lands (map grows but drift
  5.2x -> ~27x because promotion is on without the stabilizing joint BA).

**Step 2 DONE (2026-09-04) + covisibility — all verified, but blocked by keyframing.**
- Covisibility graph: `MonoMap.add_covisibility_observations` records existing map
  points into new keyframes from the tracker's data association -> **0 -> 164**
  multi-view landmarks (previously every landmark was single-view, so any joint BA
  was futile).
- Joint BA: `PhotometricBA.optimize_mono_joint_window` (scipy TRF, sparse Jacobian,
  poses + covisible points, per-point covariance priors), replaces the three
  alternating pose/depth-only BAs, runs every keyframe. Synthetic test recovers a
  perturbed pose. 89 tests pass.
- **Still no metric win, root cause finally pinned:** the map won't grow past **2
  keyframes** because `should_insert` fires **only twice in 250 frames**. Not a
  promotable-count problem (158 promotable when it fired). The `min_motion_from_latest
  = 0.12` criterion is measured in the **under-scaled estimate frame** (scale_vs_gt ~=
  8), so ~0.3 m of real motion registers as ~0.04 and never re-triggers. The joint BA
  has nothing to optimize with only 2 keyframes.

**Next (real Phase 3C): scale-invariant keyframing.** Replace metric-motion insertion
with SVO's `needNewKf` criteria (`frame_handler_base.cpp`): **median pixel disparity
vs last keyframe** and **tracked-landmark count dropping below a threshold** — both
scale-invariant, so under-scale can't suppress keyframing. This is the piece that
finally lets the (already-built) covisibility + joint BA earn their keep.

## Phase 3B - Structure ownership: one depth estimator

Today, a landmark's depth is touched by **three** subsystems: the Sparse3D EKF,
`optimize_mono_inverse_depth_window`, and `_update_map_landmarks_from_klt`
(per-frame two-ray re-triangulation). They disagree and fight -> depth jitter and
drift. Establish a strict lifecycle with a single owner at each stage:

```
new feature -> Sparse3D immature filter -> (converged) promote to map -> BA owns depth
```

- **Sparse3D keeps exactly one job:** initialize/converge depth for *immature*
  points not yet in the map (this is the DSO/SVO "depth filter" role, and it is
  legitimate). The existing birth (`_try_birth`, two-ray + parallax gate) and
  maturation (`>=3` obs, inlier-ratio) logic stays.
- **On promotion, Sparse3D releases the point.** `retire()` already exists and is
  called at keyframe creation; make that the *only* handoff. After promotion the
  point's depth is never written by Sparse3D again - only by BA.
- **Delete `_update_map_landmarks_from_klt`** entirely. Mapped-point depth is
  refined by BA (Phase 3A), not by a parallel per-frame KLT re-triangulation.
- **Do NOT strip the filter internals as a "simplification."** The 5-D state
  (`state[3:5]` = a 2-D correspondence-*position* bias with a random walk) is a
  deliberate, validated covariance-consistency device from the ECHO-LI Sparse3D
  formulation (`../ECHO-LI-notes/docs/sparse3d/filter_formulation.md`, SS III-IV,
  Fig. 1: depth-NEES 27.7 -> 2.29 on exact poses). It models temporally-correlated
  tracker drift (aperture / occlusion / repetitive texture) and is provably *not*
  separately observable from along-bearing depth error (SS VIII), so it restores
  the honesty of the reported *covariance*, not the point estimate. Removing it
  would not improve the point estimates this pipeline consumes and would silently
  re-collapse the covariance. Keep the filter faithful to the port.
- **Do not gate promotion on the filter's depth variance.** The mono pipeline
  promotes points using `conv_depth_variance` (the filter's reported depth
  variance). But ECHO-LI's honesty apparatus is bias-RW **+** pose-covariance
  propagation (lambda ~= 5) **+** the per-landmark range-process noise `q_k`
  (SS V-D, Eq. 10-11), and this repo ports **only the bias-RW**. Per SS V-D/SS X,
  without the range-process term the *recursive* depth variance is radial-blind and
  pinned (~3 m) regardless of inputs -> the promotion gate is reading a number that
  cannot be honest in this port. Replace it with **geometric** promotion criteria
  (parallax angle, track length, reprojection agreement across the window); keep
  the variance only as a soft secondary signal.

### Depth-filter architecture decision (DSO-style, 2026-09-04)

Measurement drove this: post-bootstrap the map froze at 1 keyframe / 185 landmarks
because `mature_landmarks(require_converged=True)` yielded too few points. The
breakdown at frame 130 (755 filters): **441 killed by `few_obs`, 242 by the Beta
`low_inlier` gate**, 0 by variance -> only 72 converged, and the map cannot grow.

Decision: keep the Sparse3D filter (per-frame immature depth is required for the
obstacle-avoidance layer, unlike triangulate-on-keyframe / ORB which gives none),
but move to a **DSO-style pure-Gaussian formulation**:

- **Drop the Gauss-*Beta* mixture -> pure Gaussian EKF.** The Beta inlier model is
  exactly the `low_inlier` throttle and (per the ECHO-LI analysis) is not buying
  accuracy in this port. Outlier rejection moves to **fundamental/homography RANSAC
  on the KLT tracks** (upstream) plus the per-update **Mahalanobis chi2 gate**
  (in-filter). RANSAC is two-view; the chi2 gate catches temporally-drifting tracks
  RANSAC misses — keep both.
- **Keep the correspondence-bias state, zeroable.** Default live (preserves the
  ECHO-LI covariance-consistency result for the avoidance bound); can be frozen via
  `bias_walk_sigma = 0` if a run does not want it.
- **Promote to the VO map on parallax**, not the (radial-blind, pinned) filter
  variance. Depth-variance stays only as a loose non-binding check.
- **Depth measurement source = DSO epipolar search** (`ImmaturePoint::traceOn`):
  along the epipolar line, weight the depth update by gradient energy *along* the
  line (`a = dir^T gradH dir`, `errorInPixel = 0.2 + 0.2*(a+b)/a`) — aperture points
  (edge parallel to the line) widen rather than reject. This replaces/augments KLT
  as the per-point measurement and densifies coverage for avoidance.

**Staging:**
- **Stage 1 (in progress):** drop Beta -> pure Gaussian; fundamental-RANSAC KLT
  prefilter feeding `sparse3d.update`; promote to map on parallax. Keeps KLT as the
  depth measurement. Small, and directly testable against the 0.217 m / 5.2x baseline.
- **Stage 2:** gradient-selected immature set + **vectorized** epipolar-search
  measurement into the same Gaussian filter. Bigger; this is where the Python
  real-time cost lives (a naive per-point epipolar loop is the same perf trap as the
  old patch-matcher — must be batched). **Perf path (user):** port hot functions to
  Rust `std::arch` AVX2 via PyO3, or prototype with numba AVX2 first; premature until
  Stage 2 exists and is profiled, but that's the intended route to real-time.

**Stage 1 result (2026-09-04): 3B landed, 88 tests pass, but no metric win alone —
3A/3B/3C are genuinely coupled (as this plan warned).** Changes: Beta removed (pure
Gaussian + chi2 gate + consecutive-failure cleanup); the redundant RANSAC prefilter
was deleted once we found `FeatureTracker.update` **already** runs fundamental-RANSAC
on the KLT tracks (`feature_tracker.py:60`); promotion switched from `require_converged`
to **parallax-gated, using the filter's own birth-triangulated depth** (one estimator).

Effect: the map now **grows** (2 keyframes / ~452 landmarks vs frozen 1 / 185) — the
promotion gate was indeed the blocker. But metrics **regress**: ATE ~0.20 m (flat),
scale-drift spread **5.2x -> ~27x**. Sweeps show neither the promotion parallax
threshold (2-8 deg) nor the keyframe-insertion motion threshold (0.12 -> 0.015) moves
it: KFs stays at **2**. Root causes, both coupling 3B to the rest:
1. **BA never runs** (`ba_every_keyframes = 3`, only 2 KFs) — nothing refines the
   freshly promoted immature points, so the new keyframe injects scale-inconsistent
   structure. A growing map with no BA drifts *more* than a frozen one.
2. **Retire-on-promotion sawtooth** — promoting retires those landmarks from Sparse3D
   and `feature_tracker.remove_ids`, so each KF resets a chunk of tracks that must
   re-accrue baseline; sustained >=3-KF growth never gets going in 250 frames.

Conclusion: 3B is correct and necessary but **cannot show a win in isolation**. The
next unit of work is landing **3A (joint BA that runs every keyframe) + 3C (sustained
keyframing)** together with 3B, measured as a whole.

**Consequent fix — ONE pipeline, two consumers (not two parallel estimators).**
The stall was caused by the *promotion gate*, not by lacking a second system.
Keep a single point lifecycle:

```
KLT feature -> immature point (per-frame Gaussian depth)   <- avoidance reads this every frame
     '-> at keyframe, if parallax sufficient:
         triangulate with the TRACKED poses -> map point -> BA   <- VO reads this
```

The only change is the **promotion rule**: replace `require_converged` (waits for
slow EKF convergence, which lags motion) with **triangulate-at-keyframe on
parallax** — finalize the immature point's depth from the poses tracking already
provides, immediately. Triangulate-on-keyframe is the *promotion step* of the
existing pipeline (DSO/ORB lifecycle), NOT a parallel structure estimator. Sparse3D
keeps serving avoidance with per-frame immature depth; nothing is duplicated.

### Honest pose uncertainty *is* available here (unlike EqVIO)

ECHO-LI cannot recover the anchor->current cross-covariance from a current-state
filter (SS V-C, SS X: it would need a "kept-pose window"). **This project already
has that window** in the keyframe BA, so `Sigma_ac` falls out of the reduced
pose Hessian for free:

1. at the GN optimum `H = J^T Sigma^-1 J`;
2. Schur-marginalize landmarks -> `H_pp' = H_pp - H_pl H_ll^-1 H_lp`;
3. fix the gauge and invert -> `Sigma_pp = (H_pp')^-1`; the block `Sigma_ac` gives
   the anchor->current *relative* covariance (gauge-invariant, so scale-*drift*
   uncertainty is well-defined even though absolute scale is not).

At a ~7-10 KF window this is a tiny dense inverse, not the O(N K^2) that made it
prohibitive in the filter. This is the honest source that can later feed the
range-process depth term (Eq. 10-11) with a *real* `q_k`.

**Known limit (accepted):** `Sigma_pp = H^-1` is a CRB/Laplace covariance - it
bounds an *unbiased* estimator, so it reports the *random* part of pose
uncertainty but is blind to the **systematic** scale lean SS X identifies as the
dominant end-to-end error. Local BA cannot remove that bias. The direct backbone
mitigates it at the source (photometric residuals do not carry the KLT
aperture/edge-slide correspondence bias of SS X Fig. 10), and the residual
systematic drift is **deferred to loop closure / global optimization** (out of
scope for this plan; see Future Work).

**Targets symptom #1:** promotion stops depending on a dishonest variance, depth
stops oscillating between the filter / KLT re-triangulation / BA, and pose
uncertainty comes from a source that is honest about what it can know.

## Phase 3C - Keyframe management

The current policy is three heuristic layers stacked on a 5-keyframe window:
scored redundancy discard + usefulness-EMA (`keyframe_usefulness_ratio`,
`keyframe_low_usefulness_frames`) + landmark rescue (`_transfer_visible_landmarks`).
It is opaque, and the rescue path **re-hosts** landmarks onto a new anchor, which
breaks the anchored-inverse-depth assumption. Replace with two explicit rules.

- **Insertion (one rule, two conditions).** Insert a keyframe when *both*:
  (a) parallax/translation since the last KF exceeds a threshold expressed as a
  ratio of median scene depth (meaningful only once Phase 1 fixes scale - note
  `min_motion_from_latest = 0.12` is currently in arbitrary units), **and**
  (b) tracking health dropped (tracked-inlier ratio below a floor). Delete the
  `low_inlier_ratio` / `low_coverage` extra branches.
- **Marginalization (one rule).** Grow the window (5 -> ~7-10 for a less
  forgetful local map), then drop the keyframe with the fewest landmarks covisible
  with the current frame. Delete the redundancy score and the usefulness-EMA
  layer entirely.
- **No rescue.** Delete `_transfer_visible_landmarks`. A dropped keyframe's
  points either survive because another window keyframe still hosts them, or they
  die and are re-triangulated fresh by Sparse3D. Do **not** re-host anchors.
  (Later, marginalization can convert the dropped pose into a linear prior for
  BA; out of scope for the first pass.)

**Targets symptoms #1 and #3:** a larger, comprehensible window reduces drift,
and two rules replace ~150 lines of interacting heuristics.

## Phase 4 - Structural cleanup

- Split the 1500-line `ExperimentalMonocularVO` into `MonoBootstrap`,
  `MonoTracker`, `MonoMapper` - each independently testable (existing unit tests
  mostly survive, retargeted).
- Fix the fragile `main.py:246-256` branch: give mono its own top-level
  `if/else` that doesn't rely on the stereo `window_pts` being coincidentally
  empty.
- `git rm` the ~50 `diagnostics_mono_*.jsonl` scratch files (add
  `diagnostics_*.jsonl` to `.gitignore`).

---

## Deletion checklist (concrete)

| Delete | Why |
|---|---|
| `_essential_pose_guess`, `_image_motion_pose_guess`, `_flow_rotation_guess` | competing runtime estimators |
| `_matched_observation_pose_guess`, `_match_visible_reference_patches` | SVO path + Python perf killer |
| `_keyframe_klt_pose_guess` + KLT-residual/flow-cos gating | redundant estimator + gate sprawl |
| `optimize_mono_pose_window`, `optimize_mono_geometric_pose_window` | replaced by one joint BA |
| `image_motion_fallback_depth`, `_point_flow_step_scale`, `_bootstrap_step_scale` | ad-hoc scale injection |
| `_update_map_landmarks_from_klt` | third structure estimator; BA owns mapped-point depth |
| `_transfer_visible_landmarks` | re-hosts anchors, breaks anchored inverse-depth |
| `keyframe_usefulness_ratio`, `keyframe_low_usefulness_frames`, redundancy score | opaque heuristic layers on keyframe discard |
| ~30 threshold attrs in `__init__` | belong to deleted code |

**Explicitly NOT deleted:** the Sparse3D 5-D filter state / correspondence-bias
random walk. It is a validated port from ECHO-LI (see Phase 3B); the fix is to
stop *consuming* its recursive depth variance for promotion, not to alter the
filter.

Rough impact: `monocular.py` goes from ~1520 lines to an estimated 400-500, most
of the mono JSONL fields disappear, and the per-frame Python patch loop is gone
(big speedup -> symptom #3).

---

## Sequencing

Phases 0 -> 3 are ordered by dependency; **do Phase 0 first** so later phases are
measured. Phases 3A/3B/3C are tightly coupled (BA, structure ownership, and
keyframe policy all touch the same window) and should land together behind the
Phase-0 gate. Each phase must keep the test suite green and the Phase-0
integration test as the gate.

**Recommended first move:** Phase 0 (build the Sim(3)-scaled eval harness + slim
regression test). It is non-destructive and gives a measuring stick before
anything is cut.

---

## Pose tracking rebuild (2026-09-04) — KLT-PnP anchor, and the DSO-style direction

**The breakthrough.** ATE 0.337 -> **0.078 m** (official main.py; 0.048 standalone),
drift-spread 23x -> ~5x, 5 keyframes, and the KLT-vs-reprojection displacement stops
diverging (was 147->350 px, now ~2 px). The direct tracker was **not** broken: seeded
with a good PnP pose it stays within 0.3 px (measured). The bug was the **seed** — the
pipeline seeded direct tracking from the drifting motion-model / candidate-selection
soup, which inherited prior drift with no absolute anchor. Fix (`_pnp_from_klt` +
rewired `process()`): every frame, **PnP-RANSAC from the pure-2D KLT map-point
correspondences** (pose-drift-free absolute anchor) -> seed `DirectTracker.track_map`
photometric refine -> keep output. Ripped out `_direct_pose_guesses` / `svo_match` /
`_track_direct_candidates` from the live path.

**Diagnostic that mattered:** the **KLT-vs-reprojection displacement** (KLT pixel vs
`project(T, point_w)`) is the definitive probe for pose divergence and for map-point
health — use it.

**Known residual: KLT edge-drift (aperture) ratchet.** KLT slides along edges by
1-3 px, *inside* the PnP `reprojectionError` tolerance so it is never rejected; and
because KLT re-seeds from its own previous (drifted) position (`FeatureTracker.update`
overwrites `prev_points` with the raw KLT output, no pose feedback), the error
**accumulates**. Direct refinement keeps the *pose* photometrically correct, so KLT and
the reprojection end up agreeing *photometrically* but disagreeing *geometrically*.
This is why the residual displacement grows monotonically (0.4 -> 2.8 px over 250
frames) and would erode longer sequences. "KLT is drift-free" is only true w.r.t.
*pose* (it can't self-reinforce a wrong pose); it is **not** feature-drift-free.

**The fix: re-anchor the KLT chain to the pose-refined reprojection — NOT a 2-D
re-alignment search, and independent of the BA choice.** A per-point 2-D search
(SVO-style feature alignment) is *aperture-limited exactly like KLT* — it slides along
the same edge and buys nothing. The correct correction is by **pose perturbation with
each point constrained to its epipolar line** (1-DOF): the point's along-edge position
is set by the *reprojection from the pose*, and the pose is pinned by the
well-conditioned (gradient-across-epipolar) points — so an edge point that KLT lets
slide is nailed to the globally-consistent geometry. We already have this refine
(`DirectTracker.track_map` after PnP; the point moves only as the pose moves = along
its epipolar locus). **The missing piece is small: after the refine, reproject the map
points and write those pixels back into `FeatureTracker.prev_points`, so next frame KLT
seeds from the epipolar-correct location, not its own accumulated drift.** This
re-anchors the KLT front-end to geometry every frame; it is aperture-immune
(pose-constrained) and orthogonal to whether the joint BA is reprojection- or
photometric-based.

Three **composable** pieces (not either/or, as an earlier draft wrongly implied):
1. **KLT** — cheap wide-radius guess, seeded from the *prior frame's reprojection*.
2. **Pose-perturbation photometric refine (epipolar-constrained) + reprojection
   write-back** — corrects the pose AND re-anchors the KLT chain. Kills the ratchet.
   **DONE (2026-09-04):** `_reanchor_klt_to_reprojection` +
   `FeatureTracker.set_positions`, gated by `pnp_reproj_thresh` (re-anchor exactly the
   points PnP trusts as inliers). Displacement ramp 0.4->2.8px flattened to ~0.1px
   (no accumulation); ATE 0.078 -> **0.045 m** (official), drift-spread 1.9x.
3. **Joint BA** — reprojection now; **photometric/DSO** later (poses + affine +
   inverse depths, points 1-DOF on their host ray). Orthogonal accuracy/density
   upgrade; the **Rust `std::arch` AVX2 / PyO3** perf plan lands here.

**DSO-gaps in the current tracker** (for the photometric upgrade, piece 3):
`DirectTracker.track_map` uses only a **Huber** weight on residual magnitude — it has
**no gradient-magnitude weighting** (`w ~ c²/(c²+|∇I|²)`, which DSO uses to stop edge
pixels from dominating and biasing the pose) and **no structure tensor** (`gradH`,
DSO's along-/across-epipolar observability weight). It is also **single-scale** (5-px
pattern, no coarse-to-fine pyramid).

*Not* a gap: the raw-vs-equalized split (KLT on histogram-equalized frames, direct/
photometric on raw) is fine — equalization is a monotonic intensity remap so it does
not move pixels; KLT output is equalization-invariant *coordinates*; each subsystem is
internally consistent and the two exchange only positions, never intensities. Keep the
invariant "all-KLT equalized, all-direct raw" (a future photometric BA must also use
raw).

## Future work (out of scope)

- **Loop closure / global optimization.** Local BA covariance is a CRB bound and
  is blind to the systematic scale lean (Phase 3B, SS X). Bounded *systematic*
  drift is a global-consistency problem, not a local-covariance one; the intended
  remedy is loop closure (pose-graph / Sim(3) global optimization) added after the
  single-backbone pipeline is stable. Deferred deliberately - do not try to fix
  systematic scale drift with local heuristics in the meantime.
- **Range-process depth term (Eq. 10-11) with a real `q_k`** fed from the BA
  anchor->current relative covariance (Phase 3B). Only worth doing if a downstream
  consumer actually needs honest depth covariance.
