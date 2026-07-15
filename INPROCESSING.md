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

## Graveyard / known limits

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

1. Re-measure hulls + implied cuts on BaB-hard instances (not MTS-closable
   ones) with `oracle: alpha`.
2. Per-probe alpha re-optimization rung, if hull gains justify the compute.
3. Forced-phase hunting on small-deficit instances (where `mip_confirm`
   would finally have a behavioral test case).
