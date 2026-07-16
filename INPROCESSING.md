# Theory-Level Inprocessing: Phase Probing

GPU-batched two-phase probing of unstable ReLUs before branch-and-bound.
Every fact the probes prove is **entailed by the specification** — sound by
construction — and is delivered through amortized channels: intermediate-bound
hull refinements written back into the verifier state that flows to BaB,
spec-conditional forced phases, phase-implication edges, and implied-bound
GCP-CROWN cuts.

Feature lives in `complete_verifier/phase_probing.py` (`_PhaseProber`,
`_SpecChecker`, entry point `probe_and_refine`), with hooks in
`incomplete_verifier_func.py` (pre-`build()` alpha-retention gate +
post-incomplete probe call), `beta_CROWN_solver.py` (`alpha_drop_unused`
retention), `state/alpha.py` (extraction filtering), and
`cuts/infered_cuts.py` (pending-cut install + BICCOS vivification).

## Theory

Unstable ReLUs are ranked by instability score `|lb·ub|/(ub−lb)`; the top
`max_neurons` are probed in both phases (pre-activation clamp to `>= 0` /
`<= 0`), batched over the probe dimension and chunked per pinned layer.
Per feasible pin:

- **Hull refinement**: downstream intermediate bounds are recomputed under
  the pin; the elementwise hull over the two phases is box-sound and is
  written back into `ret` (flows into BaB's starting bounds). Never-loosen
  asserts guard every write-back.
- **Forced phases** (spec-conditional): if one phase proves the *negated*
  spec region infeasible, the other phase is forced — kept clearly separated
  from unconditional facts in code. Both-phases-verified ⇒ instance verified
  (early exit `safe-incomplete`).
- **Implication edges**: pin ⇒ other-ReLU-phase facts harvested from the
  probe bounds (feeds BICCOS-clause vivification).
- **Implied-bound cuts**: `v ≤ ub_I + (ub_A−ub_I)·z_i` in exact GCP-CROWN
  wire format, installed via a pending list that survives BICCOS's
  destructive cut-pool rebuilds (capped by `implied_cuts_max`, gated by
  `implied_cuts_gap_frac`).

### Oracle ladder

Probes start at the cheapest oracle and escalate near-misses
(`escalate_margin_frac`, `escalate_hull_frac`):

`crown` → `alpha` (reuses build-time alphas; **requires the retention fix
below**) → `beta` (one-split BaB domain via `update_bounds`) → `gurobi`
(exact per-clause MILP; needs a valid license via `GRB_LICENSE_FILE`).

**Measured saturation point: `alpha`.** On cifar100 idx0 the beta rung adds
≈0 output-margin over alpha, and the gurobi rung (12-probe pilot, 30s
TimeLimit each) closed nothing, changed no hull/margin stat, and cost 14×
total wall time (454.7s vs 31.8s). Use `oracle: alpha`; `beta`/`gurobi`
remain available but are not recommended (see graveyard).

### The alpha-retention fix

`build()` calls `alpha_drop_unused()`, which discards intermediate-start-node
alphas — without them the alpha rung can only strengthen output margins and
hulls stay crown-quality. When probing is enabled with `oracle != crown`,
a gate (`phase_probing_keep_alpha_nodes`) retains them through `build()`;
the prober then trims retention to layers downstream of the earliest probed
pin, and a `finally` wrapper restores the normal post-build alpha state on
every exit path (BaB sees the usual footprint; alpha extraction is filtered
so `ret['alphas']` is byte-identical to the non-probing case). Effect on
cifar100 idx0: hull avg tightening 0.001844 → 0.002055 (+11%), implication
edges 3 → 19.

The beta rung additionally stashes non-final-start-node alphas around its
`CROWN-optimized` calls and repeats final alphas to the probe batch — the
retained batch-1 alphas otherwise crash the optimizer's best-alpha indexing
(CUDA device-side assert; the poisoned context then cascades into unrelated
failures).

## Running it

Env: `/home/liam/.local/share/mamba/envs/verifier_pyt280/bin/python`, run
from `complete_verifier/`; vnncomp benchmarks resolve via
`../../vnncomp2024_benchmarks`. (Fresh clones need
`git submodule update --init` for auto_LiRPA.)

## Benchmarking (A/B)

Control and treated commands are **identical except the probing flags** —
same config yaml, instance range, timeout. With probing disabled, all edits
are inert (verified: baseline verdicts and trajectories match pre-edit).

```bash
# control (stock α,β-CROWN)
python abcrown.py --config exp_configs/vnncomp24/cifar100.yaml \
    --start 0 --end 1

# treated (control + inprocessing) — only probing flags added
python abcrown.py --config exp_configs/vnncomp24/cifar100.yaml \
    --start 0 --end 1 \
    --phase_probing --phase_probing_batch_size 128 \
    --phase_probing_max_neurons 64 --phase_probing_oracle alpha
```

### Clause-vivification + SAT-layer arms (BICCOS required)

Joint-pin vivification and the SAT layer act on BICCOS clauses, so BOTH
arms carry the BICCOS flags (cifar100.yaml does not enable cuts) with
otherwise DEFAULT settings; the arms differ ONLY in the phase-probing
flags:

```bash
# control arm: stock α,β-CROWN + BICCOS (default settings)
python abcrown.py --config exp_configs/vnncomp24/cifar100.yaml \
    --start 0 --end 1 \
    --enable_cut --enable_bab_cut --biccos_cuts

# treated arm: control + probing + joint vivification + SAT layer
python abcrown.py --config exp_configs/vnncomp24/cifar100.yaml \
    --start 0 --end 1 \
    --enable_cut --enable_bab_cut --biccos_cuts \
    --phase_probing --phase_probing_batch_size 128 \
    --phase_probing_max_neurons 64 --phase_probing_oracle alpha \
    --phase_probing_vivify_joint --phase_probing_vivify_grade beta \
    --phase_probing_vivify_iterations 20 \
    --phase_probing_sat_layer
```

(`--phase_probing_vivify_iterations 20` per the sensitivity study below:
2x oracle adequacy over the inherited default at 2x GPU, verdicts
preserved.)

(Earlier pilots in this doc used `--tree_traversal breadth_first
--number_cuts 200` per the BICCOS class docstring; the canonical arms
are default-settings-only by user decision, 2026-07-15.)

Inertness re-verified 2026-07-15 after the vivification/SAT edits: the
plain control command (no probing flags, no cuts) gives safe 26.2s on
idx0 with zero phase-probing/SAT/vivification output — matching the
pre-edit baseline (safe 26–29s).

Ablations: drop `--phase_probing_sat_layer` for vivification-only; drop
`--phase_probing_vivify_joint` for SAT-layer-only (the SAT DB then holds
edges/forced phases + unvivified BICCOS clauses). Read the
`Phase probing joint vivification:` lines (cumulative eligible /
shortened / literals_removed / full-pin verified / probes / gpu_time /
len_hist) and the `Phase probing SAT layer:` lines (checked / pruned /
clamped / clauses / time). Instance notes from single-trajectory pilots:
cifar100 idx 1 and 6 close at the incomplete stage, idx 2 is unsafe-pgd
(no BICCOS clauses on any of those); idx 0/3/4 are safe BaB instances
with 2–3-literal clause pools; idx 5/7 are hard (timeout) — idx 7 is the
one with genuinely long clauses (2–6+ literals).

Read the `Phase probing summary:` line (probed/escalated counts, forced
phases, tightened neurons, avg/max tightening, implication edges, implied
cuts, probe time) and the final `Result:`/`Summary` block. Verdict parity
with control is a hard requirement; build-phase bounds have ~1e-6 GPU
nondeterminism jitter, so compare verdicts and probe stats, not bitwise
logs. NOTE: the vnncomp2021 acasxu config uses input split — probing (and
BICCOS) do not engage there; use activation-split benchmarks.

## Configuration reference

yaml block `solver: phase_probing:` (CLI in parentheses):

| knob | default | effect |
|---|---|---|
| `enabled` (`--phase_probing`) | false | master switch |
| `batch_size` (`--phase_probing_batch_size`) | 256 | probes per GPU chunk (OOM-adaptive) |
| `max_neurons` (`--phase_probing_max_neurons`) | 0 = all | top-N unstable neurons to probe |
| `oracle` (`--phase_probing_oracle`) | crown | max ladder rung: crown / alpha / beta / gurobi |
| `apply_forced_phases` (`--no_phase_probing_forced_phases`) | true | apply spec-conditional clamps |
| `mip_confirm` (`--phase_probing_mip_confirm`) | false | Gurobi-confirm each forced phase (inert when no phases force) |
| `escalate_margin_frac` (`--phase_probing_escalate_margin_frac`) | 0.5 | near-miss margin fraction for escalation |
| `escalate_hull_frac` (`--phase_probing_escalate_hull_frac`) | 0.05 | hull-gain fraction for escalation |
| `implied_cuts_max` (`--phase_probing_implied_cuts_max`) | 100 | cap on installed implied-bound cuts |
| `implied_cuts_gap_frac` (`--phase_probing_implied_cuts_gap_frac`) | 0.2 | min relative gap for a cut to qualify |
| `vivify_biccos` (`--no_phase_probing_vivify_biccos`) | true | vivify BICCOS clauses against the probe edge graph |
| `vivify_joint` (`--phase_probing_vivify_joint`) | false | joint-pin descent vivification of BICCOS clauses (GPU entailment probes) |
| `vivify_grade` (`--phase_probing_vivify_grade`) | beta | vivification oracle grade: crown / alpha / beta |
| `vivify_max_lits` (`--phase_probing_vivify_max_lits`) | 32 | max clause length considered by joint vivification |
| `vivify_budget` (`--phase_probing_vivify_budget`) | 8192 | total vivification probes per instance |
| `vivify_iterations` (`--phase_probing_vivify_iterations`) | 0 = inherit beta-crown iteration | optimizer iterations per beta-grade probe chunk (recommended: 20) |
| `sat_layer` (`--phase_probing_sat_layer`) | false | CPU clause DB (PySAT/CaDiCaL): pick_out-time domain filtering + phase clamping |

Env (measurement only): `PHASE_PROBING_GUROBI_MAX=N` caps gurobi-rung
probes; `GRB_LICENSE_FILE` points at the Gurobi license (WLS works).

## Measured results (cifar100 vnncomp24 idx0, 64 neurons, batch 128)

| config | hull avg / max | edges | forced | probe time | verdict |
|---|---|---|---|---|---|
| control | — | — | — | — | safe 26–29s |
| crown rung | 0.001844 / 0.182 | 3 | 0 | 4.4s | safe |
| alpha rung (retention fix) | **0.002055 / 0.183** | **19** | 0 | 7.7s | safe |
| beta rung | 0.002055 / 0.183 | 19 | 0 | 8.3s | safe 31.8s |
| gurobi rung (12-probe pilot) | 0.002055 / 0.183 | 19 | 0 | 429s | safe 454.7s |

Forced phases stay 0 here because a single pin cannot close the ~1.0 output
deficit (best probe margin lift ≈ +0.22) — expect them only on instances
with small deficits. An all-escalate diagnostic showed the near-miss set
already captures all alpha-grade hull gain.

## Joint-pin clause vivification (ported from Marabou 2026-07-15)

BICCOS blocking clauses and CDCL conflict clauses share one disease: they
are conditioned on the whole split path (maximally coarse). The Marabou
side proved the cure — joint-pin descent: for a clause `OR_i L_i`, pin the
negations of a literal PREFIX simultaneously and recompute bounds; if the
pinned region is verified, the prefix disjunction is entailed and replaces
the clause. `ClauseVivifier` in `phase_probing.py` is the abcrown port,
hooked into `BICCOS.update_cut` (an amortized delivery point — clauses are
global, never trail-conditioned) right after the edge-graph pass.

Mechanics: all (clause, prefix) candidates of a cut-inference round are
batched into few GPU calls; pins are multi-neuron clamps in the
`interm_bounds` tensors with everything fixed at the refined root bounds
(a multi-split BaB domain view). Literals are ordered
most-tightening-first (probe hull gain, then root instability). Literals
whose truth value is decided by the root bounds are screened CPU-side
(false ⇒ removed unconditionally; true ⇒ clause is a tautology, skipped).
Each clause additionally gets a `j = n` full-pin probe — the
grade-adequacy diagnostic: a clause whose full pin set does not re-verify
at the oracle's grade can never shorten.

**CONDITIONING is the whole game.** BaB runs once per unverified OR group
with a single-clause spec (`Activation BaB batch k/N`, `c shape [1,1,·]`);
BICCOS clauses are entailed w.r.t. THAT clause only. The probes therefore
use the picked domains' `cs`/`thresholds` and the net's CURRENT
final-start-node alphas (sliced to batch 1 — spec-consistent with the
run). Probing against the full multi-group spec demands entailment the
clauses never had: measured 0/640 (crown+alpha grades) and 14/746
full-pin at beta grade before this fix, vs 296/746 after.

Oracle grades (`vivify_grade`), measured on cifar100 idx0 (BICCOS flags
`--enable_cut --enable_bab_cut --biccos_cuts --tree_traversal
breadth_first --number_cuts 200` added to BOTH A/B arms):

| grade | mechanics | full-pin verified | shortened | cost |
|---|---|---|---|---|
| alpha (backward, reuse_alpha, all interm fixed) | one backward pass | 0/640 | 0 | 0.44 ms/probe |
| beta, mis-conditioned (full spec) | CROWN-optimized + SparseBeta pins | 14/746 | 0 | ~10 ms/probe |
| **beta, per-run conditioning** | + current-run cs/alphas + GCP cut pool | **296/746 (39.7%)** | **102/746 (13.7%), 106 literals** | 5.4 ms/probe, 8.4s GPU total |

The beta grade expresses each pin BOTH as a bound clamp and as a
multi-entry SparseBeta split history (generalizing the prober's beta rung
from one pin to several), optimizes over the probe batch, and runs WITH
the GCP-CROWN cut pool — exactly the grade `biccos_verification` used to
verify the clauses in the first place. Non-final alpha entries are
stashed around the probes (the optimizer's best-alpha indexing crashes on
batch-mismatched entries), and beta/cut state is swapped out and restored
around every call.

Alpha-knob notes: the oracle already does per-probe alpha
re-optimization — the batch-1 parent slice is repeated with
`requires_grad` and CROWN-optimized trains each probe's copy — so the
strength-vs-cost knob is the ITERATION COUNT (`vivify_iterations`,
0 = inherit `solver:beta-crown:iteration`). The probing feature's
retained-alpha-node set (`phase_probing_keep_alpha_nodes`) is
structurally IRRELEVANT here: the vivifier fixes ALL intermediate bounds,
so intermediate-start-node alphas are never consumed by its probes (and
retention is released before BaB starts anyway). "0 iterations" is the
alpha grade — measured dead above. Implementation trap: the iteration
override must be applied AFTER `set_crown_bound_opts('beta')`, which
stamps the config iteration count over `optimize_bound_args` (the first
sweep silently ran every arm at 10 iterations; caught because two "arms"
were byte-identical — incidentally a determinism check).

### `vivify_iterations` sensitivity (single-trajectory runs, canonical default-settings arm, 2026-07-15)

cifar100 idx0 (safe, 2-lit clause pool, 100s timeout):

| iters | shortened | lits removed | full-pin adequacy | probes / GPU s | ms/probe | verdict |
|---|---|---|---|---|---|---|
| 10 (inherit) | 70/687 (10.2%) | 70 | 192/687 (28.0%) | 1385 / 8.0 | 5.8 | safe 68.2s |
| **20** | 74/687 (10.8%) | 74 | **483/687 (70.3%)** | 1385 / 15.7 | 11.3 | safe 67.8s |
| 50 | 74/687 (10.8%) | 74 | 516/687 (75.1%) | 1386 / 40.5 | 29.2 | **unknown 101s — oracle cost blew the timeout** |

cifar100 idx7 (timeout instance, long clauses 2–9+ lits):

| iters | shortened | lits removed | full-pin adequacy | probes / GPU s | ms/probe |
|---|---|---|---|---|---|
| 10 (inherit) | 128/938 (13.6%) | 152 | 180/938 (19.2%) | 4307 / 16.0 | 3.7 |
| **20** | 136/958 (14.2%) | 160 | 258/958 (26.9%) | 4440 / 31.5 | 7.1 |
| 50 | 137/672 (20.4%) | 161 | 270/672 (40.2%) | 2522 / 50.7 | 20.1 |

(idx7 rows are not iso-pool: the oracle GPU time competes with BaB inside
the 100s timeout, so higher iters sample fewer cut-inference rounds —
compare shortened-rate and adequacy, not absolute counts.)

**Recommendation: `--phase_probing_vivify_iterations 20`.** The 10→20
step buys the big adequacy jump (28→70% on idx0, 19→27% on idx7) at 2×
GPU with verdicts preserved; 50 is past the knee — +5pp adequacy on
idx0, ~0 extra literals, 2.6× more GPU, and it cost idx0 its verdict.
The adequacy-vs-shortening gap on idx0 (70% adequate, 10.8% shortened)
is real, not oracle weakness: 2-lit clauses only shorten if a 1-lit
prefix (a forced phase) verifies. On the long-clause idx7 the same step
converts into more removals per adequate clause.

Caveat for idx0: eligible clauses are already near-minimal —
`len_hist={2: ~750, 3: ~38, 4: ~7}` — BICCOS's own constraint
strengthening plus `merge_cuts` grinds this benchmark's clauses down to
1–3 literals, so most shortenings here are 2→1 (a 1-literal blocking
clause = a forced phase, the strongest cut there is). The Marabou-style
89–99% shortening headroom needs instances with longer clauses; on idx0
the ceiling is the 39.7% full-pin-verifiable fraction, of which a third
shorten. Shortened clauses flow into the GCP-CROWN pool automatically
(vivification edits `tmp_cuts` in place BEFORE pool insertion/merging) —
no separate pending-cut path needed.

Per-instance hit rates, cifar100 vnncomp24, full stack (probing + joint
vivify beta + SAT layer), one trajectory each:

| idx | verdict | eligible | shortened (hit rate) | lits removed | full-pin verified | probes / GPU s | len_hist |
|---|---|---|---|---|---|---|---|
| 0 | safe 61.7s | 687 | 70 (10.2%) | 70 | 192 (28%) | 1385 / 7.4 | 2:676, 3:11 |
| 3 | safe 32.5s | 476 | 78 (16.4%) | 78 | 110 (23%) | 952 / 3.5 | 2:476 |
| 4 | safe 25.3s | 6 | 0 | 0 | 2 (33%) | 12 / 0.4 | 2:6 |
| 5 | unknown (timeout) | 1033 | 84 (8.1%) | 84 | 562 (54%) | 2150 / 9.5 | 2:973, 3:38, 4:20, 5:2 |
| 7 | unknown (timeout) | 700 | **124 (17.7%)** | **144** | 173 (25%) | 2646 / 10.9 | 2:243, 3:104, 4:131, 5:97, 6+:rest |

(idx 1/6 close at the incomplete stage, idx 2 is unsafe-pgd — no BICCOS
clauses there.) idx7 is the only sampled instance with genuinely long
clauses, and the only one where multi-literal removals appear (144
removals over 124 clauses). idx5 shows the honest converse: 54% of full
pins re-verify but the 2-literal clauses are TIGHT — no 1-literal prefix
verifies — so vivification correctly certifies them minimal. Cost is
~4–5 ms/probe throughout; the whole-oracle budget of 8192 probes was
never exhausted.

## CPU SAT layer (`sat_layer.py`)

A clausal mirror of every phase fact, in PySAT with the CaDiCaL backend
(`Cadical195.propagate(assumptions=...)` verified working in
verifier_pyt280). Two databases:

- **persistent** — probe implication edges (box-sound) and forced phases
  (spec-conditional, valid for every OR group of this property);
- **per-run** — BICCOS + vivified clauses, keyed by a fingerprint of the
  picked domains' `(cs, thresholds)` and FLUSHED when the BaB run (OR
  group) changes: BICCOS clauses are only entailed within their run. The
  CaDiCaL solver is rebuilt on flush (incremental solvers cannot retract).

Duties, all CPU, zero GPU contention, hooked after `pick_out` in
`bab.act_split_round`:

1. every picked domain's phase assignment (split history) is
   unit-propagated; a conflict ⇒ the domain contains no counterexample ⇒
   pruned before any bound computation (`select_domain_batch` filters the
   whole picked-batch dict, custom objects first for atomicity);
2. propagation-implied phases are clamped into the domain's constructed
   intermediate bounds before bounding;
3. cross-domain transfer is LAZY: every domain passes through pick_out
   before bounding, so a clause learned at iteration k automatically
   filters every domain popped later — including domains queued before
   the clause existed. No queue rescan.

Domains with non-phase splits (nonzero branching points) or heterogeneous
batch specs are passed through unfiltered.

Hooked in BOTH BaB loops: `bab.act_split_round` (the deprecated loop MTS
still uses — where all BICCOS inference happens) and
`activation_split.stage_preprocess.branch_and_bound_preprocess` (the main
loop, reached via a `sat_layer` field on `PreprocessConstArguments`; this
pipeline cannot skip a round, so an all-falsified batch keeps one domain).

Measured, cifar100 idx0 full stack (probing + joint vivify beta + SAT):
verdict safe 61.7s, 2190 domains visited (BICCOS-only control: safe
54.8s, 2750 visited — GPU batching couples into trajectories, compare
verdicts and stats, not timings); SAT layer checked 470 picked domains,
pruned 63 (13.4%) before any bound computation, clamped 510
propagation-implied phases, mirrored 232 clauses, 0.033s total CPU.
Vivification on this trajectory: 70/687 shortened, full-pin 192/687.

## Graveyard / known limits

- **Vivification at crown/alpha grade**: measured dead — 0/640 clauses
  shortened, 0/640 full-pin verified on cifar100 idx0. BICCOS clauses
  were verified at beta grade with per-run alphas and the cut pool; a
  fixed-bounds backward pass with build-time alphas can never re-verify
  them. Code stays behind `vivify_grade: crown|alpha`; use `beta`.
- **Vivification against the full multi-group spec**: measured dead
  (0 shortened at every grade). BaB runs per OR group; clauses are
  entailed per group. Condition on the run's own `cs`/`thresholds`.
- **Intersecting stored single-pin probe bounds** (rung-0 oracle):
  never built here — measured dead on the Marabou side (0 removals in
  ~49k checks). Joint propagation under the pin conjunction is required.
- **Gurobi rung**: measured dead weight at 30s/probe TimeLimit — zero
  closures, zero hull/margin delta, 14× wall time. Dropped from the arc;
  code remains behind the opt-in `oracle: gurobi`.
- **Beta rung**: ≈0 margin over alpha on the calibration instance; kept
  (cheap) but not the default recommendation.
- **Alpha-rung ceiling**: it *reuses* build-time alphas optimized for the
  unpinned box. Next strength rung would be per-probe alpha re-optimization
  of downstream bounds — no auto_LiRPA surgery needed, purely a compute-cost
  question.
- **BICCOS vivification**: 0 removals so far (35–378 edges vs long clauses);
  plumbing is unit-tested, needs denser edge graphs to matter.
- MTS/BICCOS heuristics are batching-sensitive; GPU free-memory couples into
  trajectories via `auto_enlarge_batch_size`. Compare verdicts and stats,
  not exact timings.

## Next steps

1. Cluster A/B of the vivification + SAT-layer arms (commands above) —
   the per-instance numbers in this doc are single-trajectory pilots;
   verdict/stat comparison over the full benchmark is pending.
2. Vivification oracle strength: 60–75% of full pins do NOT re-verify at
   the beta grade on most instances — the remaining gap to
   `biccos_verification` is per-parent-domain alphas (probes use a
   batch-1 slice of the current net alphas). Plumbing the source domain's
   alphas per clause through `constraint_strengthening` is the next rung.
3. Longer-clause benchmarks: idx7-like instances (clauses 4–6+ literals)
   are where multi-literal descent pays; the 2-literal pools of idx0/3/4
   cap the hit rate at "2→1 = forced phase" conversions.
4. SAT-layer clamping currently only writes bounds already constructed in
   `d`; feeding implied phases into the split PROPOSALS (branching) is
   untried.
5. Re-measure hulls + implied cuts on BaB-hard instances (not MTS-closable
   ones) with `oracle: alpha`.
6. Per-probe alpha re-optimization rung, if hull gains justify the compute.
7. Forced-phase hunting on small-deficit instances (where `mip_confirm`
   would finally have a behavioral test case).
