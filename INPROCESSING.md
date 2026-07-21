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

Inertness re-verified 2026-07-15 after the vivification/SAT edits and
again 2026-07-16 after the vivify_use_cuts/vivify_bcp edits: the plain
control command (no probing flags, no cuts) gives safe 25–26s on idx0
with zero phase-probing/SAT/vivification output — matching the pre-edit
baseline (safe 26–29s).

Strongest measured treated variant (see the closed-loop, BCP and
mirror-oracle sections below): add `--phase_probing_vivify_use_cuts pool
--phase_probing_mirror --phase_probing_vivify_dry_rounds 3` to the
treated arm (`--phase_probing_vivify_bcp` is superseded by the mirror).
"Strongest" here means most clauses shortened — on cifar100 that has NOT
been shown to reduce the search (see the GPU-descent graveyard entry);
the ranking is by clause counters, which this doc's protocol section now
warns against trusting alone.

For the next cluster sweep, add `--phase_probing_reprobe` (built
2026-07-19, default-off) as its own arm, ideally with a
`--phase_probing_reprobe_budget 30` variant.

Ablations: drop `--phase_probing_sat_layer` for vivification-only; drop
`--phase_probing_vivify_joint` for SAT-layer-only (the SAT DB then holds
edges/forced phases + unvivified BICCOS clauses);
`--phase_probing_vivify_use_cuts off` for a cut-less oracle. Read the
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
| `vivify_use_cuts` (`--phase_probing_vivify_use_cuts`) | auto | GCP-CROWN cuts in the vivification oracle: auto (net's current module = previous rebuild's pool; round 1 cut-less) / off (ablation) / pool (probe-scoped module rebuilt per round from pool + pending + fresh clauses — the closed loop) |
| `vivify_bcp` (`--phase_probing_vivify_bcp`) | false | unit-propagate each (clause, prefix) pin set through the SAT layer before GPU: conflicts shorten at zero GPU cost, implied literals become extra pins (needs `sat_layer`) |
| `sat_layer` (`--phase_probing_sat_layer`) | false | CPU clause DB (PySAT/CaDiCaL): pick_out-time domain filtering + phase clamping |
| `wide_root` (`--phase_probing_wide_root`) | false | cross-layer batched root probe: clamp injected per batch ROW via a `clamp_interim_bounds` hook instead of through `interm_bounds`, so probes at any depth share one backward pass (see "Wide root probe" section). Makes full coverage affordable. |
| `wide_root_window` (`--phase_probing_wide_root_window`) | -1 | with `wide_root`: a row keeps recomputed bounds only this many INTERMEDIATE layers past its own pin (-1 = unlimited, 0 = all-fixed weak oracle). 1 is the measured knee — cuts saturate at the first layer. |
| `wide_root_interm_only` (`--phase_probing_wide_root_interm_only`) | false | with `wide_root`: skip the full-network output backward pass, compute ONLY the intermediate bounds the pass harvests (hull/edges/cuts). Drops the forced-phase channel; 10–15× faster on deep nets. |
| `wide_root_sparse` (`--no_phase_probing_wide_root_sparse`) | true | with `wide_root`: recompute only neurons unstable at the root (root bounds as `aux_reference_bounds`); off = dense IBP-superset (3.4× slower, identical facts at window 1). |
| `wide_root_time_budget_frac` (`--phase_probing_wide_root_time_budget_frac`) | 0.0 | cap wide-root probe wall clock at this fraction of the per-instance timeout (0 = uncapped); depth-sorted chunking stops adding chunks, so coverage degrades gracefully. |
| `reprobe` (`--phase_probing_reprobe`) | false | conditioned re-probing at depth (see its section); needs `sat_layer` |
| `reprobe_budget` (`--phase_probing_reprobe_budget`) | 15.0 | total wall-clock seconds of re-probing per instance |
| `reprobe_max_neurons` (`--phase_probing_reprobe_max_neurons`) | 64 | top-N still-unstable neurons re-probed per pass |

Env (measurement only): `PHASE_PROBING_GUROBI_MAX=N` caps gurobi-rung
probes; `GRB_LICENSE_FILE` points at the Gurobi license (WLS works).

## Wide root probe + interm-only: affordable FULL coverage (2026-07-20)

The root probe used to issue one `compute_bounds` call per pinned layer, because
`interm_bounds` is a per-CALL dict: a probe pinned at layer L needs L
fixed-with-clamp and everything below it FREE (recomputed under the clamp — the
source of hull refinement / implication edges / implied cuts), and probes at
different depths cannot agree on that. Cost was superlinear (65 → 145 ms/neuron,
64 → 256 neurons), so full coverage (`max_neurons 0`) blew every per-instance
budget — the `probe_alpha_nall` graveyard entry.

**Wide root** (`--phase_probing_wide_root`) stops expressing the clamp through
`interm_bounds`. Every layer from the earliest pin on is left free, and the
clamp is written into `node.lower/upper` for the owning rows by a hook on
`clamp_interim_bounds` (which `BoundedModule` already calls at the end of
`compute_intermediate_bounds`, both fresh and cached paths). Intermediate bounds
compute in topological order, so a row's clamp lands before anything downstream
of it is bounded — each row still gets full downstream recomputation under its
own pin, and rows at different depths share one backward pass. Soundness is the
same restriction the per-layer path makes (the clamp uses max/min not
assignment, so a neuron the recomputation re-stabilised never crosses; freed
layers are intersected back against the alpha-optimized root bounds).

Three levers made cifar100 idx0 (1,447 neurons) go **153.2s → 8.85s (17×)**,
per-probe now CHEAPER than the n=64 arm (2.6 vs 9.3 ms/row):
1. `wide_root_sparse` (default on): `aux_reference_bounds` = root bounds, so a
   freed layer recomputes only neurons unstable AT THE ROOT (BaB's own
   assumption) instead of an IBP-guessed dense superset. ~85% of the win. NOT a
   fact loss at window 1 — dense and sparse give identical edges/cuts there
   (dense only refines root-stable neurons, which yield nothing).
2. `wide_root_window 1`: keep recomputation for one intermediate layer past each
   pin. Cuts saturate at the first layer; layers 2+ cost 15.5s for +61 edges and
   +0 cuts. (Window is measured in INTERMEDIATE-layer positions, not raw graph
   nodes.)
3. Cross-layer batching itself: converts the superlinear per-layer scaling into
   flat (111 → 106 ms/neuron, 64 → 1,447 neurons).

**Interm-only** (`--phase_probing_wide_root_interm_only`): `compute_bounds` is
`check_prior_bounds` (produces the freed-layer bounds behind hull/edges/cuts)
followed by `backward_general` (the OUTPUT bound, which only yields forced
phases). On tinyimagenet the output pass was ~97% of probe time and produced 2
forced phases out of 1,925 neurons. Retargeting `final` at the deepest freed
layer's consumer prunes everything downstream (`_set_used_nodes`), so the pass
runs a short path with spec 1. Measured on tinyimagenet full coverage:

| | control | interm-only |
|---|---|---|
| probe median | 30.19s | **2.38s** (12.7×) |
| implication edges | 82 | 82 (identical) |
| implied cuts | 12 | 12 (identical) |
| forced phases | 2 | 0 (channel dropped) |

⇒ **The shipped full-coverage arm has NO unit-derivation channel** — it is
carried entirely by bound tightening and cuts. This is the key divergence from
Marabou, whose arc was unit-dominated (+13/+26/+28). abcrown's root probe barely
refutes phases (alpha-CROWN root bounds are already tight, so a single pin rarely
closes one); hypothesis: CNN vs dense-network, untested.

### Population A/B, 8 configs, treatment vs BICCOS baseline (cut-matched)

`WIDE_ROOT=1 WIDE_WINDOW=1 PROBE_MAX_NEURONS=0 WIDE_INTERM_ONLY=1`, full
mirror+SAT stack. **cut-matched** = both arms carry `--enable_cut
--enable_bab_cut --biccos_cuts`, so the comparison isolates probing (vs the raw
config-only baseline, which measures the cut machinery: raw→BICCOS is +1.84×
time = the cuts, BICCOS→treatment only +1.12× = probing). net = solves gained;
dom = domains-visited ratio.

| config | net / time / dom vs BICCOS |
|---|---|
| cifar_cnn_a_adv | +1 / 0.90× / 0.63× |
| cifar_cnn_a_mix | +2 / 0.87× / 0.70× |
| cifar_cnn_b_adv | +2 / 0.93× / 0.54× |
| mnist_cnn_a_adv | −1 / 0.97× / 0.96× |
| tinyimagenet | +3 / 1.12× / 0.81× |
| cifar10-resnet | 0 / 1.00× / 0.76× |
| cifar100 | −3 / 1.08× / 0.69× |
| oval_base | 0 / 0.96× / 0.58× |

**Domains visited drop uniformly (0.54–0.81×): the facts prune the tree on
every config** — the clean mechanistic result, invisible against the raw
baseline. Small SDP-FO CNNs win outright (+1/+2 solves AND faster). The
negatives are the two largest nets (cifar100, cifar10-resnet-adjacent) where
per-node cut cost outweighs the pruning; the domain reduction is real there too
(0.69×/0.76×) but does not convert. Full raw-baseline table and the SDP-FO "_4"
variants are pending. All arms: 0 safe↔unsafe disagreements.

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

Alpha-knob notes: the oracle is beta-CROWN + GCP-CROWN with the alphas
held FIXED at the parent slice (`enable_alpha_crown: False`) — see
"Alpha re-optimization is dead" in the graveyard for the measurement
that removed it. `vivify_iterations` (0 = inherit
`solver:beta-crown:iteration`) therefore now trains only the betas and
the general (cut) betas. The probing feature's retained-alpha-node set
(`phase_probing_keep_alpha_nodes`) is structurally IRRELEVANT here: the
vivifier fixes ALL intermediate bounds, so intermediate-start-node
alphas are never consumed by its probes (and retention is released
before BaB starts anyway). "0 iterations" is the alpha grade — measured
dead above. Implementation trap: the iteration override must be applied
AFTER `set_crown_bound_opts('beta')`, which stamps the config iteration
count over `optimize_bound_args` (the first sweep silently ran every arm
at 10 iterations; caught because two "arms" were byte-identical —
incidentally a determinism check).

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

### Closing the pool–oracle loop: `vivify_use_cuts` (2026-07-16)

The beta-grade oracle has ALWAYS engaged GCP-CROWN cuts when a cut module
existed (`set_cut_params` flips per-activation `cut_used`, and
`set_beta_cuts` registers the general betas with the optimizer, so the
cut multipliers are re-optimized on the pinned domains by the
`vivify_iterations` loop — no extra machinery was needed for that). What
`auto` (the old, still-default behavior) misses: the module is the pool
as of the PREVIOUS BICCOS rebuild, so round-1 probes run entirely
cut-less (the module does not exist yet), pending phase-probing
implied-bound cuts are invisible until their official install, and this
round's freshly inferred clauses never reach the probe pool.

`vivify_use_cuts: pool` closes the loop: each vivify call builds a
probe-scoped cut module from `biccos_cuts` (previous rounds, with
earlier vivifications already merged in — a clause shortened in round k
strengthens the oracle testing every clause of round k+1) + cplex cuts +
pending implied-bound cuts + this round's fresh clauses, deduped, and
swaps it in for the probes only (cutter/module state is fully restored
so BICCOS's own rebuild logic sees the exact pre-vivify state).
`off` disables cut terms in the probes (the ablation arm).

Soundness: every pool entry is entailed on the counterexample-relevant
region of THIS run's spec — the same conditioning the probes use. A
fresh clause participating in its own j < n probes is standard
SAT-vivification semantics (every counterexample in the pinned region
satisfies the entailed clause); it cannot fake a shortening
propositionally because root-false literals are screened by `_parse`.
The j = n full-pin probe of a pool-member clause IS self-certifying in
principle (the full pin contradicts the clause itself), so in pool mode
the full-pin stat measures the optimizer's cut exploitation rather than
pure grade adequacy — measured, it stays far from 100% (see below), i.e.
20 iterations do not fully exploit even the self-cut.

Measured (single trajectories, canonical arm + the flag; verdict parity
in all arms — idx0 safe everywhere, idx7 is the timeout instance in
control too). Arms are not iso-pool (oracle GPU time competes with BaB
inside the timeout): compare rates.

cifar100 idx0 (2–3-lit clause pool):

| use_cuts | eligible | shortened | lits | full-pin | probes / GPU s | verdict |
|---|---|---|---|---|---|---|
| off | 687 | 0 (0%) | 0 | 105/687 (15.3%) | 1385 / 12.5 | safe 71.5s |
| auto | 687 | 74 (10.8%) | 74 | 483/687 (70.3%) | 1385 / 13.5 | safe 68.7s |
| pool | 697 | **119 (17.1%)** | 119 | 458/697 (65.7%) | 1409 / 15.7 | safe 68.8s |

cifar100 idx7 (long clauses, 2–11 lits):

| use_cuts | eligible | shortened | lits | full-pin | probes / GPU s | verdict |
|---|---|---|---|---|---|---|
| off | 853 | 132 (15.5%) | 156 | 187/853 (21.9%) | 3680 / 20.1 | unknown 103.3s |
| auto | 953 | 136 (14.3%) | 160 | 259/953 (27.2%) | 4382 / 30.1 | unknown 101.2s |
| pool | 950 | **210 (22.1%)** | **235** | 344/950 (36.2%) | 4419 / 32.9 | unknown 110.3s |

Notable: on idx0 the cut pool is the ENTIRE source of shortening — with
cuts off the oracle shortens nothing (0/687) and full-pin adequacy
collapses to 15%. On idx7 the oracle retains most of its power without
cuts (15.5% vs 14.3% shortened rate) but the pool mode's fresh-pool
rebuild still buys +54% clauses / +47% literals over auto. Pool
construction cost is negligible (pool sizes 30–242 cuts, 16–20 rebuild
rounds, GPU delta within jitter).

### BCP-extended pin sets: `vivify_bcp` (2026-07-16)

Before spending GPU on a (clause, prefix) candidate, the prefix's pin
set is unit-propagated through the SAT layer's clause DB
(`PhaseSATLayer.propagate_pins`, persistent facts + run clauses scoped
to THIS run's fingerprint; this round's fresh clauses are mirrored into
the DB only AFTER vivification, so a clause never propagates against
itself). Propagation is per-prefix with exactly the prefix pins as
assumptions, so every implied literal's antecedents lie within the
prefix — the incremental-assumption scheme that makes prefix semantics
sound. Two uses:

1. a conflict at prefix j proves the pinned region empty of
   counterexamples by propagation alone — the clause shortens to j
   literals at ZERO GPU cost, and all probes with j' >= j are skipped
   (already entailed);
2. otherwise the implied literals join the prefix's pin set as extra
   clamps + SparseBeta splits, making the GPU probe strictly stronger.

Measured (same protocol; `bcp` = canonical arm + `--phase_probing_vivify_bcp`,
`pool+bcp` = both new flags):

| instance / arm | eligible | shortened | lits | of which BCP-only (0 GPU) | bcp extra pins | probes skipped | full-pin | probes / GPU s | bcp CPU | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| idx0 bcp | 687 | 74 (10.8%) | 74 | 0 | 6106 | 3 | 512/684 (74.9%) | 1382 / 12.7 | 0.01s | safe 64.6s |
| idx0 pool+bcp | 693 | 115 (16.6%) | 116 | 0 | 6696 | 1 | **591/692 (85.4%)** | 1396 / 12.6 | 0.01s | safe 62.0s |
| idx7 bcp | 957 | 151 (15.8%) | 179 | **149 (-177 lits)** | 8139 | 431 | 106/703 (15.1%) | 4007 / 30.1 | 0.04s | unknown 105.6s |
| idx7 pool+bcp | 954 | **223 (23.4%)** | **252** | 148 (-176 lits) | 8121 | 428 | 196/702 (27.9%) | 3977 / 30.8 | 0.04s | unknown 105.9s |

Notable: on idx7 BCP alone finds essentially all of the auto arm's
shortenings for free (149 zero-GPU vs auto's 136 GPU-probed) — the run
clause DB accumulated in earlier rounds subsumes much of what the GPU
oracle re-proves — while the extra implied pins lift what the GPU adds
on top. On idx0 (2-lit pool, sparse clause interactions) BCP never
conflicts (0 zero-GPU shortenings) but its ~4.4 extra pins/probe raise
full-pin adequacy 70.3% → 74.9% (85.4% with pool). CAVEAT: in bcp arms
the full-pin stat is NOT comparable to non-bcp arms — clauses whose full
pin conflicts under BCP skip their j = n probe (full_tested drops, and
the skipped ones are the "easy" clauses).

Combined `pool` + `bcp` is the strongest configuration measured on both
instances, at unchanged verdicts and ~equal GPU. Suggested cluster arm:
canonical treated command + `--phase_probing_vivify_use_cuts pool
--phase_probing_vivify_bcp`.

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

### `max_neurons` sweep (full stack + valve, idx0, 2026-07-16)

More probing = more facts = less search, but probe cost is SUPERLINEAR
(5× per 2× neurons) with the current implementation:

| N | probe GPU | escalated | edges / implied cuts | run-3 visited | wall | verdict |
|---|---|---|---|---|---|---|
| 64 | 8.3s (13% of wall) | 54 | 19 / 0 | 2106 | 62.4s | safe |
| 128 | 42.6s (44%) | 118 | 34 / 8 | **1572 (−25%)** | 97.6s | safe |
| 256 | 101.8s | 246 | 67 / 16 | truncated | 142.4s | unknown |
| 512 | 235.3s | 616 | 143 / 28 | truncated | 276.5s | unknown |

The N=128 arm proves the information channel scales (search −25%,
implied cuts appear); the cost blowup is an artifact, not intrinsic:
(a) escalation count scales with N — beyond the top-64, nearly every
probe is a "near miss" at default escalate fracs; (b) probes are
chunked PER PINNED LAYER — top-512 neurons spread across all layers,
so batch-128 GPU calls run nearly empty. Prerequisites for raising N
on the cluster: cross-layer probe packing + escalation budget cap,
then budget-scaled N (roadmap item 1). Until then: N=64 for
~100s-budget families, N=128 only where timeouts are long.

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

- **Alpha re-optimization in the vivify oracle** (removed 2026-07-19).
  The probes trained alphas, betas and general (cut) betas through one
  parameter group, so `vivify_iterations` priced all three together and
  the alphas' contribution had never been isolated. Freezing them moves
  the full-pin adequacy DIAGNOSTIC (idx0 489/569 → 421/569, idx7 25.8%
  → 20.7%) and the DELIVERABLE by nothing: idx0 stays at 169 shortened
  / 171 literals, idx7's GPU-attributable ~76 shortenings are identical
  across four trajectories. Now `enable_alpha_crown: False` — alphas
  are fixed relaxation coefficients from the parent slice. Taking them
  out of the parameter group (not `lr_alpha=0`, which still pays for
  the backward pass) cut vivify GPU 41% on idx0, 16.4s → 9.7s. The
  oracle is marginally weaker as a side effect (`enable_alpha_crown`
  also skips `node.opt_start()`; idx0 settles at 166/169), which is
  sound — fewer optimized parameters can only loosen a bound. Escape
  hatch `PHASE_PROBING_VIVIFY_OPT_ALPHA=1`.
  **Do not resurrect off an adequacy argument**: full-pin adequacy is
  an oracle-strength diagnostic that has now twice moved freely while
  the deliverable stood still (see the protocol note below).
- **The GPU joint-pin descent, on cifar100** (kept, but measured to
  change nothing on this family — 2026-07-19). GPU probes on vs off
  (`PHASE_PROBING_VIVIFY_NO_GPU=1`, every zero-GPU duty still running),
  clean pairs in one job:

  | idx | shortened | domains ON | domains OFF | verdict |
  |---|---|---|---|---|
  | 0 | 169 | 1532 | 1532 | safe / safe |
  | 3 | 82 | 1332 | 1332 | safe / safe |
  | 4 | 0 | 1184 | 1184 | safe / safe |
  | 5 | 116 | 12578 | 12066 | unknown / unknown |
  | 7 | ~76 (GPU part) | 7450 | 9966 | unknown / unknown |

  Zero verdict changes; identical domain counts on all three
  deterministic instances; both timeout instances get equal or better
  throughput WITHOUT it, at 1–33s of GPU each inside ~100s budgets.
  The delivery channel is fine — probing + SAT move idx0 from the
  BICCOS-only control's 2748 domains to 1532 (−44%) — vivification
  specifically contributes nothing to that. Marabou converged on the
  same place from the other direction: their mirror superseded the
  hand-rolled LP-descent shortening machinery ("superseded, not
  refuted"), and their descent runs at an 88–99% hit rate where ours
  runs at 11–17%. NOT gated off by default yet — that flips the
  meaning of every historical A/B, and the measurement is one config
  family with mostly 2-literal clause pools. The doc's own long-clause
  regime (idx7-like) is where multi-literal descent should pay and is
  exactly where the timeout masks the outcome.
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
   verdict/stat comparison over the full benchmark is pending. Include a
   `--phase_probing_vivify_use_cuts pool --phase_probing_vivify_bcp` arm
   (strongest measured single-instance configuration).
2. Vivification oracle strength: 60–75% of full pins do NOT re-verify at
   the beta grade on most instances — the remaining gap to
   `biccos_verification` is per-parent-domain alphas (probes use a
   batch-1 slice of the current net alphas). Plumbing the source domain's
   alphas per clause through `constraint_strengthening` is the next rung.
   (The 2026-07-16 closed-loop/BCP work narrows this differently: pool
   cuts + implied pins lift idx0 full-pin adequacy to 85%.)
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

## The mirror oracle (ported 2026-07-16, `--phase_probing_mirror`)

The port plan below is now BUILT (duties 1–3; learner harvest deferred).
`PhaseSATLayer` grew three mirror duties, all gated by
`solver: phase_probing: mirror` (requires `sat_layer`); the flag also
supersedes `vivify_bcp` (it force-enables the extras loop but replaces
its conflict test):

- **Assumption-core vivification** (`vivify_clause_pins`): per eligible
  clause, ONE conflict-bounded `solve_limited(assumptions=pins)`
  (conf_budget 200) BEFORE any GPU probe; on UNSAT the failed-assumption
  core IS the shortened clause (arbitrary subset — not just a prefix,
  which BCP/GPU descent structurally cannot produce). Mirror-shortened
  clauses skip their GPU probes entirely. Self-refutation is impossible:
  fresh clauses are mirrored into the DB only after their own attempt
  (the `infered_cuts.py` ordering already guaranteed this).
- **Per-run UNSAT certificate**: empty core (PySAT `get_core() -> None`)
  = the run DB is boolean-unsat below the assumptions ⇒ THIS OR group
  has no counterexample ⇒ verified. Delivered by pruning every
  subsequently picked domain of the run (`process_picked_domains`
  returns keep=[]) and skipping all further vivification probes.
- **Failed-literal probing** (`failed_literal_pass`): after each clause
  batch lands in `add_blocking_cuts`, gated on DB growth >= 32 clauses
  (Marabou's gate) and a 1s cap: `propagate([lit])` both signs per known
  var; a conflict makes the negation a run-scoped forced phase (unit in
  the run DB). Sees everything BICCOS has learned — a fact source GPU
  probing never has.

Measured (canonical treated arm + `vivify_use_cuts pool`, mirror replacing
bcp; single trajectories, verdict parity everywhere):

| instance / arm | shortened | lits | zero-GPU shortened | GPU probes / s | oracle CPU | verdict |
|---|---|---|---|---|---|---|
| idx0 pool+bcp (baseline) | 115/693 | 116 | 0 | 1396 / 12.6 | 0.01s | safe 62.0s |
| idx0 pool+mirror | 115/696 | 116 | 0 (696 calls, all SAT) | 1405 / 16.1 | 0.005s | safe 66.9s |
| idx7 pool+bcp (baseline) | 223/954 | 252 | 148 (-176 lits) | 3977 / 30.8 | 0.04s | unknown 105.9s |
| idx7 pool+mirror | **314/953** | **484** | **239 (-408 lits)** | 3479 / 22.6 | **0.006s** | unknown 103.5s |

idx7: mirror cores dominate BCP exactly as in Marabou — +41% clauses,
+92% literals, 912 GPU probes skipped, GPU time -27%, at 6ms total CPU
(253 of 953 solves UNSAT, 0 budget-outs). idx0 confirms the converse:
a 2-literal pool with sparse clause interactions has NO boolean
structure — all 696 solves SAT — its shortenings come entirely from the
numeric cut pool. Failed-literal probing found 116 (idx0) / 12 (idx7)
run-scoped forced phases for ~5ms. No UNSAT certificate fired on either
instance (expect them on runs whose clause DB actually saturates).

### SAT-derived facts as GCP-CROWN cuts (`pop_new_cut_facts`, 2026-07-16)

Delivery upgrade (user insight: don't just prune picked domains —
feed the boolean facts into the RELAXATION so every subproblem
tightens): failed-literal units (1-literal blocking cuts, run-scoped =
exactly the BICCOS pool's own scoping since the cutter is rebuilt per
BaB bootstrap) and, once per run, the persistent probe implication
edges (box-sound 2-literal clause cuts) are exported in blocking-cut
wire format and appended to `tmp_cuts` right after the SAT mirror pass
in `BICCOS.update_cut` — they ride the normal pool → cut-module →
optimized-multiplier path, and in `pool` mode they also strengthen the
vivification oracle. 1-literal arelu cuts are a form BICCOS itself
already emits (post-strengthening), so the module path is proven.

Safety valve (`--phase_probing_fact_cuts_max`, default 50): every
installed cut is a general-beta constraint optimized per domain, so
fact cuts respect a number_cuts-style per-run budget — units first
(a forced phase is the strongest clause cut), edges fill the rest;
0 disables injection. UPSTREAM FINDING (2026-07-16): BICCOS's own
`number_cuts` cap is leaky — `biccos_cuts[:max_cuts_num+1]` truncation
only runs in the "Stop inferring" branch (`infered_cuts.py` ~line 264);
during active inference `merge_cuts` grows the pool unbounded and
`net.cutter.cuts = biccos_cuts + cplex + pending` installs ALL of it
(measured: 213 installed cuts on idx7 at default number_cuts=50). Not
changed here (it would alter the control arm); flag if cut-count
overhead shows up in cluster profiles.

### GPU dry-round gate (`--phase_probing_vivify_dry_rounds N`)

Takes GPU out of the loop when it stops paying (the "delegate to the
fast SAT solver" direction): after N consecutive vivify rounds in which
GPU probes shortened nothing beyond the zero-GPU passes (mirror cores /
BCP conflicts), GPU probes are gated OFF for the rest of that BaB run —
mirror/FLP/domain-filter duties keep running (they are microseconds).
Default 0 = off. Motivation measured on idx0: 16s GPU total with most
late rounds at 0 shortenings for ~1s GPU each.

### Full new stack measured (mirror + fact cuts + dry_rounds 3, 2026-07-16)

Single trajectories, canonical pool arm + the three new pieces; verdict
parity everywhere. Trajectories shift strongly (recovered GPU time goes
to BaB — pool sizes and round counts are not comparable across arms;
compare rates and GPU spend):

| instance | shortened | mirror zero-GPU | vivify GPU | flp units | fact cuts | verdict |
|---|---|---|---|---|---|---|
| idx0 | 164/554 (29.6%) | 1 | **9.0s** (was 16.1 mirror-only, 12.6 bcp) | 86 | 9 rounds injected | safe 67.5s |
| idx7 | 161/226 (71%!) | 88 (39% of eligible) | **3.1s** (was 22.6 mirror-only, 30.8 bcp) | 10 | 3 rounds | unknown 101.0s |

The dry gate fired on both instances and is the dominant GPU saver
(idx7 vivification GPU: 30.8s bcp-arm → 3.1s, a 10× reduction, while
the shortened RATE rose to 71% because the mirror keeps working after
the gate closes). idx0's wall time sits in the 62–77s jitter band of
all treated arms; the instance-level win from the freed GPU must be
read at cluster scale (or on budget-starved instances, where 13–27s of
returned GPU is the difference between a verdict and a timeout — see
roadmap item 1).

## Plan: the mirror-oracle port (from Marabou, 2026-07-16) — DONE same day

(Kept for the duty rationale and transferable laws; duties 1–3 are
implemented and measured above. Duty 5 — learner harvest — remains
deferred pending a PySAT Learner-interface workaround.)

Marabou consolidated its entire boolean side into ONE extra CaDiCaL (the
"mirror") holding only query-entailed clauses, with five duties. The SAT
layer here (`sat_layer.py`, PySAT/CaDiCaL) is already that instance — this
plan upgrades its duties to match, replacing bespoke logic and GPU calls
with microsecond SAT queries. The overhead argument cuts our way twice:
every duty moved into the solver is (a) code we no longer hand-prove and
(b) work moved OFF the GPU (the BCP pre-pass already shortens 149 idx7
clauses at literally zero GPU cost — this generalizes).

1. **UNSAT-proof duty (new, highest value).** Marabou's key discovery: when
   a DB of entailed clauses goes UNSAT below its assumptions, the query
   itself is refuted — an early-termination certificate. Here, clauses are
   entailed PER OR-GROUP, so the certificate is per-run: if the SAT layer's
   per-run DB (BICCOS + vivified + forced phases under the run's
   `cs`/`thresholds` fingerprint) is UNSAT, that OR group is VERIFIED —
   skip its remaining BaB entirely. Detection is free: PySAT
   `solve(assumptions=[])` (or an empty `propagate` conflict) after each
   clause batch lands. Soundness mirrors Marabou's: a satisfiable spec
   region keeps its entailed clause set consistent, so no false verdicts;
   contradictory entailed facts are exactly what a verified region
   produces. (Marabou measured proofs firing on the oracle's FIRST solve.)
2. **Failed-literal probing (boolean, not GPU).** For each undecided phase
   literal: `propagate([lit])`; conflict ⇒ forced phase for this run, at
   CPU propagation cost — a third source of forced phases after spec
   probing and vivification, and it sees everything BICCOS has learned,
   which GPU-side probing never does.
3. **Assumption-core vivification upgrade.** The current BCP pre-pass
   detects conflicts by propagation only; upgrade near-miss clauses to a
   conflict-bounded `solve(assumptions=neg(clause))` and take the
   UNSAT-core (`get_core`) as the shortened clause — full conflict
   analysis, still zero GPU. Marabou's ordering rule applies: a clause
   enters the SAT DB only AFTER its own vivification attempt
   (self-refutation guard); PySAT/CaDiCaL cannot delete clauses, so
   per-run conditionality must ride the run-fingerprint DB flush (already
   built) and assumptions, never permanent adds.
4. **Cadence advantage — exploit it.** Marabou's oracle is throttled by
   CDCL restart cadence (first luby restart lands at ~95% of the search on
   the calibration anchors). BaB has no such wall: every `update_cut` /
   BICCOS round is an amortized point, typically dozens per instance. Run
   duties 1–3 at every round; the same machinery Marabou could only invoke
   2–3 times per run fires continuously here. This is the structural
   reason to expect the port to OUTPERFORM the original.
5. **Learner harvest (optional, needs API check).** Marabou streams the
   oracle's own learned units/binaries back as entailed facts via
   `connect_learner`. PySAT does not expose CaDiCaL's Learner interface;
   approximate with periodic `propagate`-closure sweeps over undecided
   literals (equivalent information at slightly higher cost), or bind the
   C++ interface later if measurements justify it.

Transferable laws (measured in Marabou, assume they hold here):
- Clauses are portable; numeric propagation results are not — re-derive
  bounds, transfer clauses.
- Boolean guidance ≠ box knowledge even when logically equivalent —
  delivery representation changes search trajectories.
- The oracle doubles as a runtime soundness audit: any unsound clause in
  any channel surfaces as a premature boolean conflict long before it
  corrupts a verdict.

## Conditioned re-probing at depth (ported 2026-07-19, `--phase_probing_reprobe`)

The root probe runs pre-BaB, where the GCP-CROWN cut pool is EMPTY, so
it is plain beta-CROWN **by construction** — `_PhaseProber`'s beta rung
passes no `cutter=` and never flips `cut_used`. By the time BICCOS has
inferred clauses the pool holds 30–242 cuts, and on this benchmark the
pool is the entire source of oracle power (vivification idx0: 0/687
shortened with cuts off, 119/697 with the pool). So re-probing here is
NOT the Marabou original — theirs reruns the same DeepPoly/simplex
stack under a smaller box; ours runs a **strictly stronger oracle than
the root pass could ever have run**.

Mechanics: whenever the SAT layer's run-scoped forced-phase set grows
(`_flp_decided`, the level-0-fixed-set trigger; naive per Marabou —
their density gate measured as a wash and was removed the same day),
re-probe the still-unstable neurons with those forced phases pinned, at
beta grade with the pool cuts. The forced phases ARE the box tightening
and the beta oracle already expresses pins as clamps + SparseBeta
splits, so a re-probe is just the pin set {forced phases} +
{candidate}. A verified region ⇒ the pin is refuted ⇒ its negation is
entailed. Facts are RUN-scoped (conditioned on this OR group's spec AND
on the run's own forced phases): they enter `run_clauses` via
`add_run_unit`, never the persistent DB, and ride the existing fact-cut
channel into the pool.

**The cut pool is load-bearing, and that is measured, not assumed.** A
plain-CROWN premise diagnostic (`PHASE_PROBING_REPROBE_DIAG=1`) clamps
the forced phases into the root box and recomputes: the box shrink
ALONE moves 0 neurons on 3 of 5 idx0 passes, tightens by ~0.1% of box
width (max 0.47%, vs the root probe's own 0.183 max), and stabilizes
NOTHING. The same pins at beta grade with cuts refute 23 phases.
Marabou's "density grows as boxes shrink" premise is weak here; what
works is the oracle upgrade, not the box.

Measured (cifar100, single trajectories, default-off so control arms
are untouched):

| idx | units found | re-probe time | domains OFF | domains ON | verdict |
|---|---|---|---|---|---|
| 0 | 23 | 5.8s | 2106 | 1486 | safe / safe |
| 3 | 5 | 1.7s | 1332 | 1332 | safe / safe |
| 5 | **93** | 4.0s | 12072 | 12586 | unknown / unknown |
| 7 | 2 | 1.6s | 7926 | 7404 | unknown / unknown |

No verdict changed anywhere, and no consistent search effect. The
sharpest datum is **idx5**: it found by far the most facts (93) and
visited MORE domains. idx3 is deterministic and 5 units moved it by
exactly zero. **The idx0 −29% is NOT a treatment effect** — that
instance is bimodal between ~1500 and ~2110 domains from identical
code, and paired repeats put the OFF arm in BOTH modes (2106, 2106,
1532) while ON stayed low (1486, 1502, 1500). Suggestive at best —
3-of-3 low for ON vs 1-of-3 for OFF is not separation at n=3 on a
bimodal instance. Do not quote the −29%.

Lands default-off as machinery for a cluster sweep. Cost never bound:
1.6–5.8s of the 15s budget, so a larger budget is nearly free to try.

## Measurement protocol (read before trusting any number here)

Four separate times, an intermediate metric of this subsystem moved
independently of the outcome:

1. full-pin adequacy tripled (28% → 70%) for +4 shortened clauses;
2. 169 shortened clauses on idx0 changed domains visited by 0;
3. a leaked bound option lost idx0 its verdict (safe 73s → unknown
   114s at 400 domains) while `shortened` went UP, 169 → 177;
4. re-probing's 93 units on idx5 came with MORE domains visited.

So: **verdict and domains visited are the primary readout**; clause
counters are diagnostics. Concretely —

- run treated/control pairs in the SAME job (GPU free-memory couples
  into trajectories via `auto_enlarge_batch_size`);
- record the instance's noise floor before reading a delta. Measured:
  idx3 deterministic (1332 across three different code paths); idx7
  ±13% on shortened-clause count between byte-identical runs; **idx0
  bimodal between ~1500 and ~2110 domains** — it is NOT deterministic,
  despite earlier claims in this doc;
- on timeout instances domains-visited is THROUGHPUT, not progress —
  removing work raises it (idx7 vivify-off: 9966 vs 7450). Only the
  verdict is meaningful there;
- a knob that produces byte-identical arms is inert, not neutral. Two
  such false arms were caught this way (an `lr_alpha`-free
  `requires_grad` freeze, silently overridden by auto_LiRPA's
  `_set_alpha` at `optimized_bounds.py:125`; and the original
  `vivify_iterations` sweep).

## Roadmap: getting better on alpha-beta-CROWN (priority order, 2026-07-16)

The biggest available wins are RECOVERIES, not features — fix the cluster
regressions before building anything new.

1. **Budget-aware inprocessing gating** (fixes the largest measured loss).
   The viv+SAT cluster arm loses ~64 cifar100 instances (safe 117→63,
   other 51→115) to timeout starvation: 16–40s of flat inprocessing spend
   inside ~100s instance budgets. Scale every inprocessing stage to the
   remaining timeout: skip probing/vivification when
   remaining_budget < k × expected_cost, engage vivification only if BaB
   is still alive after N rounds, and let vivify_iterations degrade
   (20 → 10 → skip) as budget tightens. SAT-solver duties (BCP pre-pass,
   the planned mirror-oracle duties) are exempt — they are microseconds
   and should always run. Success metric: cifar100 viv+SAT arm returns to
   >= control's 117 safe while keeping its wins elsewhere.
2. **cifar_cnn_b_adv probing-arm autopsy** (~44 instances lost by probing
   alone, yet viv+SAT RESCUES it — 96/70 vs control 95/70). Isolate which
   probing channel hurts there (hull write-backs? implied cuts? alpha
   retention interplay?) with single-instance ablations before the next
   cluster round. The rescue-by-viv+SAT inversion suggests a trajectory
   effect, not a cost effect — compare per-instance wall times first to
   split cost-vs-trajectory.
3. **Mirror-oracle duties on the SAT layer** — DONE 2026-07-16 (see the
   mirror-oracle section): get_core vivification (+41% clauses / +92%
   literals over BCP on idx7, 6ms CPU), failed-literal probing (86–116
   units/instance), per-OR-group UNSAT certificate (built, not yet
   observed firing), SAT-fact cuts into the pool, and the GPU dry-round
   gate (idx7 vivify GPU 30.8s → 3.1s). Remaining from the plan:
   learner harvest (duty 5, needs PySAT Learner access) and cluster
   exposure of the new arm.
4. ~~**Per-parent-domain alphas for the vivify oracle**~~ — DROPPED
   2026-07-19. It was motivated entirely by the full-pin adequacy gap
   ("60–75% of full pins do not re-verify"), and adequacy is now
   measured twice to be decoupled from the deliverable: removing alpha
   optimization altogether dropped adequacy 14% and changed shortened
   clauses by zero. Plumbing FRESHER alphas cannot be worth it when
   removing them entirely costs nothing. Cuts, not alphas, are the
   lever on this oracle (`use_cuts off` → 0/687 on idx0).
5. **Cluster sweep of re-probing** (`--phase_probing_reprobe`, built
   2026-07-19, default-off). Single trajectories on one family cannot
   price a trajectory-mediated effect; the cluster can. Include a
   `--phase_probing_reprobe_budget 30` arm — no instance came close to
   the 15s cap, so a bigger budget is nearly free. Watch for idx5's
   signature (most facts, MORE domains) at scale: if it holds, the
   facts are actively misleading the branching heuristic rather than
   being merely neutral, which is a different and more interesting
   failure than vivification's.
6. **Decide the GPU joint-pin descent** (see the graveyard entry).
   Measured to change no verdict and no domain count on cifar100 at
   1–33s/instance. Gating it off returns exactly the budget re-probing
   wants. Needs either a long-clause family showing it pays, or a
   cluster arm confirming the null, before the default flips.
7. ~~**LP/MILP rung for the probes**~~ — PAUSED INDEFINITELY
   2026-07-19 (user decision). The machinery exists and is cheap to
   wire (`lp_mip_solver/bounds_core.py:99` `build_the_model_lp`, the
   unused `lp_solver` worker at `:41`, `update_model_bounds` at `:293`,
   `copy_model` at `utils.py:259`, the `BestBdStop`/`objbound` trick at
   `refine_core.py:489`, and the `NestablePool` already cached on
   `m.pool`). It is not worth building: the gurobi rung ALREADY ran
   exact MILP at 30s/probe and closed nothing, and exact MILP dominates
   any LP relaxation — a cheaper approximation finds the same nothing
   faster. Root probing is deficit-limited (a single pin cannot close
   the ~1.0 output deficit), not oracle-limited.
8. **Long-clause benchmarks for the cluster** (idx7-like: clauses 4–6+
   literals). The 2-literal pools of most cifar100 instances cap
   vivification at forced-phase conversions; oval21/22, sri_resnet, and
   BaB-hard families with deep trees are where multi-literal descent and
   the SAT layer's cross-domain transfer have room. Also re-run one family
   with the FULL treated stack so `viv lines`/`sat lines` are finally
   nonzero at scale — those subsystems still have zero cluster exposure.
9. **SAT-layer phases into branching proposals** (untried): implied phases
   currently only clamp bounds; feeding them into split selection touches
   the branching heuristic — measure carefully, trajectory-sensitive.
