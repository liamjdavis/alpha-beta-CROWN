#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
#########################################################################
"""Phase probing with hull-based bound refinement and an escalating probe
oracle ladder.

A preprocessing step analogous to MILP presolve probing, run after the
incomplete verifier's initial bounds exist:

1. Identify unstable ReLU neurons (pre-activation lb < 0 < ub).
2. For each unstable neuron i, run TWO probes: pin phase "active"
   (clamp the pre-activation lower bound to 0) and pin phase "inactive"
   (clamp the pre-activation upper bound to 0). A probe is one bound
   computation with the clamped intermediate bounds held fixed for the
   pinned layer and all layers before it, while everything downstream
   (and the output) is recomputed.
3. From each probe collect:
   (a) whether the probe's output lower bound already satisfies the
       specification -- then that phase region contains no counterexample and
       the OPPOSITE phase may be forced (SPEC-CONDITIONAL, see below);
   (b) the recomputed downstream intermediate bounds;
   (c) phase implication edges: "pin i active forces j active/inactive"
       whenever the recomputed bounds make a downstream ReLU stable
       (box-sound facts, used for BICCOS cut vivification);
   (d) implied-bound cut candidates from the per-phase bounds of downstream
       neurons (see _PhaseProber.emit_implied_cuts()).
4. HULL REFINEMENT (always sound, box-UNCONDITIONAL): every real input
   satisfies one of neuron i's two phases, so for any downstream neuron v the
   root bounds can be tightened to the elementwise hull
       [min(lb_active(v), lb_inactive(v)), max(ub_active(v), ub_inactive(v))].
   We take the tightest hull across all probed neurons i and intersect with
   the original bounds.
5. The refined bounds replace the incomplete verifier's intermediate bounds
   in-place, so they flow into BaB (via build_with_refined_bounds) as its
   reference/intermediate bounds. A single final output bound recomputation
   with the refined bounds may verify the property immediately.

PROBE ORACLE LADDER
Probes start at rung "crown" (plain CROWN backward). Probes that are
near-misses -- their remaining output-margin deficit is within
escalate_margin_frac of the full-box deficit at the same rung, or their pair
tightened some downstream neuron by escalate_hull_frac of its width -- are
escalated up the ladder (capped by solver:phase_probing:oracle):
  * "alpha": recompute the probe with the optimized alphas from the initial
    alpha-CROWN pass reused (reuse_alpha, the BaB update_bounds shortcut
    pattern), including the downstream intermediate-bound recomputation;
    still one batched compute_bounds per chunk.
  * "beta": treat the pin as a one-split BaB domain -- attach a SparseBeta
    built from a single-entry split history and run beta-CROWN
    (CROWN-optimized) with all intermediate bounds fixed. This strengthens
    the forced-phase test (output bound only; no downstream recomputation).
  * "gurobi": solve the pinned region exactly with a MILP built by
    build_solver_module. Skipped gracefully whenever Gurobi or its license
    is unavailable; activates automatically once a valid license appears.

SOUNDNESS NOTES
- Hull-refined bounds (step 4) are valid for EVERY input in the original
  input box, independently of the specification. Bounds from different rungs
  may be mixed freely in the hull: each probe's bounds are valid on its own
  phase region. The same holds for implication edges and implied-bound cuts.
- Forced phases (step 3a) are only SPEC-CONDITIONAL: "phase A contains no
  counterexample of THIS specification" allows clamping the root bounds to
  phase B only while verifying THIS specification (which is what the
  downstream BaB does). Such clamped bounds must never be reused for a
  different specification. They are applied only when
  solver:phase_probing:apply_forced_phases is true.

Probes are GPU-batched: each batch element shares the input box but carries a
different one-neuron clamp in its interm_bounds tensors -- the exact
mechanism BaB uses for splitting. compute_bounds runs once per chunk.
"""

import json
import os
import time

import torch

import arguments
from auto_LiRPA.utils import (
    stop_criterion_all,
    stop_criterion_batch_any,
    stop_criterion_general,
)
from state.beta import BetaFullData
from utils import expand_batch

_RUNGS = ['crown', 'alpha', 'beta', 'gurobi']

# Number of times the wide-root clamp hook actually wrote a clamp. Guards
# the interm_bounds-fixed pin layers (see _make_clamp_hook).
_CLAMPS_APPLIED = 0


def _flatten_solver_vars(v, out):
    """Flatten (possibly nested lists of) gurobi vars into a flat list."""
    if isinstance(v, (list, tuple)):
        for item in v:
            _flatten_solver_vars(item, out)
    else:
        out.append(v)
    return out


def _is_oom(e):
    """CUDA OOM raised inside TorchScript kernels surfaces as a plain
    RuntimeError, so match on the message as well."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and 'out of memory' in str(e).lower())


def _chunk_too_big(e):
    """Conditions a smaller probe chunk can fix: plain CUDA OOM, plus the
    TorchScript fuser's 32-bit element-count assert
    (`inputs[0].numel() <= std::numeric_limits<uint32_t>::max()`), which a
    big chunk over a wide layer trips well before memory runs out.

    _is_oom alone does NOT match the fuser assert, so the retry loop could
    not back off from it and the run died outright -- this killed the
    cifar100 full-coverage job at instance 109/200. Chunking only changes
    how probes are grouped into batches, never which probes run or what they
    prove, so backing off here is scientifically inert."""
    return _is_oom(e) or (
        isinstance(e, RuntimeError) and 'numeric_limits<uint32_t>' in str(e))


def _clamp_index(clamps, device):
    """Split [(row, nidx, sign), ...] into the four index tensors the clamp
    hook needs: (rows_active, idx_active, rows_inactive, idx_inactive).
    Returns None when there is nothing to clamp at this layer."""
    if not clamps:
        return None
    out = []
    for want in (True, False):
        sel = [(r, n) for (r, n, s) in clamps if (s > 0) is want]
        out.append(torch.tensor([r for r, _ in sel], dtype=torch.long,
                                device=device))
        out.append(torch.tensor([n for _, n in sel], dtype=torch.long,
                                device=device))
    return tuple(out)


def _make_clamp_hook(node):
    """Build the per-batch-row bound surgery for one node, called by
    BoundedModule.compute_intermediate_bounds right after this node's
    bounds are finalized and before node.interval is set.

    Three jobs, in this order:
      1. root intersection -- the layer is FREE in this pass, so plain CROWN
         may recompute it looser than the root's alpha-optimized bounds.
         Those root bounds hold on the whole input box, hence on every
         pinned sub-region, so intersecting is sound and keeps the wide rung
         at least as strong as the per-layer rung (which fed the root values
         in through interm_bounds);
      2. window write-back -- rows whose pin lies further upstream than the
         depth window get the ROOT bounds restored outright, so their probe
         stops paying for recomputation past the window;
      3. clamp -- rows pinned AT this layer get lb=0 (active) or ub=0
         (inactive) written in, which every downstream layer then sees.
    A row is never both (2) and (3): its own pin sits at distance 0, inside
    any window >= 0.
    """
    def hook(self=node):
        lo, up = self.lower, self.upper
        if lo is None or up is None:
            return
        B = lo.shape[0]
        if B != self._probe_batch:
            return  # not the probe batch (e.g. a batch-1 reference pass)
        olb, oub = self._probe_orig
        lo = torch.maximum(lo, olb)
        up = torch.minimum(up, oub)
        restore = self._probe_restore
        if restore is not None:
            view = restore.view(B, *([1] * (lo.dim() - 1)))
            lo = torch.where(view, olb.expand_as(lo), lo)
            up = torch.where(view, oub.expand_as(up), up)
        clamps = self._probe_clamps
        if clamps is not None:
            if lo is self.lower:
                lo, up = lo.clone(), up.clone()
            flat_lo, flat_up = lo.view(B, -1), up.view(B, -1)
            # clamp with max/min, not assignment: the pin layer is FREE
            # here, so by the time the clamp lands its bounds have been
            # recomputed and intersected with the root, and the neuron may
            # no longer straddle zero. Assigning 0 would then CROSS the
            # bounds (lb > ub) and feed a garbage relaxation downstream.
            # Intersecting is the same restriction and degenerates to the
            # assignment exactly when the neuron is still unstable, which is
            # the only case the per-layer _pin_bounds ever sees.
            # The far side is pulled along so the interval can never CROSS.
            # If recomputation already decided the neuron the other way the
            # pinned region is EMPTY; collapsing it to a degenerate point is
            # a superset of empty, hence sound, and any output bound derived
            # from it is vacuously valid.
            # Counter, not decoration: pinned layers are now passed in
            # interm_bounds and rely on clamp_interim_bounds firing on
            # BoundedModule's already-current path. If that path ever stops
            # calling the hook, every probe silently becomes a no-op and the
            # run still "succeeds" with zero facts. _CLAMPS_APPLIED must be
            # non-zero on any real probing pass.
            global _CLAMPS_APPLIED
            _CLAMPS_APPLIED += 1
            r_act, i_act, r_inact, i_inact = clamps
            if r_act.numel():
                v = flat_lo[r_act, i_act].clamp(min=0.)
                flat_lo[r_act, i_act] = v
                flat_up[r_act, i_act] = flat_up[r_act, i_act].clamp(min=v)
            if r_inact.numel():
                v = flat_up[r_inact, i_inact].clamp(max=0.)
                flat_up[r_inact, i_inact] = v
                flat_lo[r_inact, i_inact] = flat_lo[r_inact, i_inact].clamp(
                    max=v)
        self.lower, self.upper = lo, up
        # Keep node.linear in sync -- BoundedModule does the same after its
        # own reference-bound tightening (see compute_intermediate_bounds).
        if getattr(self, 'linear', None) is not None:
            self.linear.lower, self.linear.upper = lo, up
    return hook


class _SpecChecker:
    """Score/verify a batch of output lower bounds against the whole
    (remaining) specification.

    Handles the OR-of-ANDs structure used by the incomplete verifier when the
    disjuncts are optimized jointly (interm bound batch size 1):
      * an OR group is verified if ANY of its clauses has lb > rhs, or if it
        was already verified globally by the initial bounds;
      * the specification is verified if ALL OR groups are verified.

    score(lb) = min over not-yet-verified OR groups of the best clause margin
    (lb - rhs); > 0 means the region is verified. `sel` restricts the clause
    set to a subset of the full clause space (used when probing with the
    possibly prune_after_crown-pruned model.c).
    """

    def __init__(self, spec_handler, rhs, or_spec_size, device):
        num_clause = rhs.shape[1]
        sc = spec_handler.stop_criterion
        # unverified_or_mask is indexed over the ORIGINAL OR groups
        # (it was set before pruning in SpecHandler.post_process).
        unverified = spec_handler.unverified_or_mask.to(device)
        if sc is stop_criterion_batch_any:
            # Single OR with one/multiple AND clauses (num_or == 1):
            # verified iff any clause passes.
            self.or_ids = torch.zeros(num_clause, dtype=torch.long, device=device)
            self.verified_or = torch.zeros(1, dtype=torch.bool, device=device)
            self.num_groups = 1
        elif sc is stop_criterion_all:
            # Multiple ORs with a single AND each, optimized jointly:
            # clause j <-> OR j.
            self.or_ids = torch.arange(num_clause, device=device)
            self.verified_or = ~unverified
            self.num_groups = num_clause
        elif sc is stop_criterion_general:
            # Multiple ORs with multiple ANDs, optimized jointly.
            or_spec_size = or_spec_size.to(device)
            self.num_groups = or_spec_size.shape[0]
            self.or_ids = torch.repeat_interleave(
                torch.arange(self.num_groups, device=device), or_spec_size)
            assert self.or_ids.shape[0] == num_clause
            self.verified_or = ~unverified
        else:
            raise ValueError(f'Unknown stop criterion: {sc}')
        assert self.verified_or.shape[0] == self.num_groups
        self.rhs = rhs.to(device)

    def score(self, lb, sel=None):
        """lb: [batch, k] output lower bounds valid on some region (k = full
        clause count, or len(sel) when sel is given). Returns [batch] float:
        min over unverified OR groups of the best clause margin."""
        if sel is None:
            rhs, oid = self.rhs, self.or_ids
        else:
            rhs, oid = self.rhs[:, sel], self.or_ids[sel]
        margins = (lb - rhs).float()
        per_or = torch.full((lb.shape[0], self.num_groups), float('-inf'),
                            device=lb.device)
        per_or.scatter_reduce_(
            1, oid.unsqueeze(0).expand_as(margins), margins, reduce='amax')
        per_or[:, self.verified_or] = float('inf')
        return per_or.min(dim=1).values

    def region_verified(self, lb, sel=None):
        """True iff the region provably contains no counterexample."""
        return self.score(lb, sel) > 0


class _PhaseProber:
    """Holds all state of one probing session (see module docstring)."""

    def __init__(self, model, x, c, rhs, or_spec_size, spec_handler, ret):
        self.cfg = arguments.Config['solver']['phase_probing']
        self.model = model
        self.net = model.net
        self.device = model.device
        self.final_name = model.final_name
        self.spec_handler = spec_handler
        self.ret = ret
        self.x = x
        self.c = c
        self.rhs = rhs
        self.c_dev = c.to(self.device)
        self.checker = _SpecChecker(spec_handler, rhs, or_spec_size, self.device)

        lb_dict, ub_dict = ret['lower_bounds'], ret['upper_bounds']
        self.interm_names = [
            k for k in lb_dict
            if k != self.final_name and isinstance(lb_dict[k], torch.Tensor)
        ]
        self.node_order = {node.name: i
                           for i, node in enumerate(self.net.nodes())}
        # ReLU pre-activation layers with bounds, topological order.
        self.relu_preacts = []
        self.relu_idx_of_preact = {}  # pre-act name -> index into net.relus
        for ridx, r in enumerate(getattr(self.net, 'relus', [])):
            pre = r.inputs[0]
            if pre.name in self.interm_names \
                    and pre.name not in self.relu_idx_of_preact:
                self.relu_preacts.append(pre.name)
                self.relu_idx_of_preact[pre.name] = ridx
        self.relu_preacts.sort(key=lambda name: self.node_order[name])

        self.orig_lb = {k: lb_dict[k].detach().to(self.device)
                        for k in self.interm_names}
        self.orig_ub = {k: ub_dict[k].detach().to(self.device)
                        for k in self.interm_names}
        self.ref_lb = {k: self.orig_lb[k].clone() for k in self.interm_names}
        self.ref_ub = {k: self.orig_ub[k].clone() for k in self.interm_names}
        self.unstable_mask = {
            k: ((self.orig_lb[k].reshape(-1) < 0)
                & (self.orig_ub[k].reshape(-1) > 0))
            for k in self.relu_preacts
        }

        # probe bookkeeping: (layer, nidx, sign) -> {'score','verified','rung'}
        # sign: +1 = pin active (lb -> 0), -1 = pin inactive (ub -> 0),
        # matching the BaB split history convention.
        self.probe_state = {}
        self.pair_gain = {}  # (layer, nidx) -> max relative hull tightening
        self.both_verified = None
        # box-sound implication edges (layer, nidx, sign) -> set of
        # (layer_j, nidx_j, sign_j)
        self.implications = {}
        self._num_edges = 0
        # implied-bound cut candidates: key (pin_layer, pin_idx, v_layer,
        # v_idx, kind) -> (gap_frac, bound_inactive, bound_active)
        self.cut_cands = {}
        self.collect_cuts = (
            arguments.Config['bab']['cut']['enabled']
            and self.cfg['implied_cuts_max'] > 0)

        # clause subset matching model.c (set up lazily for rung >= alpha)
        self.sel = None
        self.c_sel = None
        # Alpha retention (see LiRPANet.alpha_drop_unused and the gate in
        # incomplete_verifier_func.py): when set, the net still holds the
        # intermediate-start-node alphas from build(), so the alpha rung
        # can recompute downstream bounds under a pin (alpha-quality
        # hulls). trim_retained_alphas() narrows the set to the layers
        # actually recomputed; probe_and_refine releases everything.
        self.alpha_interm_retained = bool(
            getattr(model, 'phase_probing_keep_alpha_nodes', None))
        self.wide_root = bool(self.cfg.get('wide_root', False))
        self.wide_root_window = int(self.cfg.get('wide_root_window', -1))
        self.wide_root_sparse = bool(self.cfg.get('wide_root_sparse', True))
        self.wide_root_interm_only = bool(
            self.cfg.get('wide_root_interm_only', False))
        # Window units: position within the topologically ordered INTERMEDIATE
        # layers, not raw node_order. node_order enumerates every graph node
        # (conv/add/relu/...), so a window of 2-3 nodes often fails to reach
        # the next ReLU pre-activation at all -- measured: W=0 and W=1 were
        # bit-identical (zero facts), as were W=2 and W=3. In layer units the
        # dial is monotone and means what it says.
        self.layer_pos = {
            name: i for i, name in enumerate(
                sorted(self.interm_names, key=lambda n: self.node_order[n]))
        }
        self._gurobi = None  # cached (grb, model, flat out vars) or False
        self._margin_gains = []  # per-escalated-probe margin improvements

        self.stats = {
            'probed_neurons': 0, 'forced_phases': 0,
            'forced_dropped_by_mip': 0, 'tightened_neurons': 0,
            'avg_tightening': 0.0, 'max_tightening': 0.0,
            'escalated': {r: 0 for r in _RUNGS[1:]},
            'implication_edges': 0, 'implied_cuts': 0, 'time': 0.0,
            # wall clock per rung, so a slow pass can be attributed instead
            # of guessed at (the crown/alpha split is the whole question at
            # full coverage, where ~97% of probes escalate).
            'rung_time': {},
        }

    # ------------------------------------------------------------------
    # Probe execution
    # ------------------------------------------------------------------

    def _probe_x_c(self, B, sel):
        new_x = expand_batch(self.x, B, device=self.device)
        C = (self.c_dev if sel is None else self.c_sel).expand(B, -1, -1)
        return new_x, C

    def _pin_bounds(self, layer_name, probes, base_lb=None, base_ub=None):
        """Bounds of the pinned layer with per-element one-neuron clamps.
        probes: list of (nidx, sign)."""
        B = len(probes)
        src_lb = self.orig_lb[layer_name] if base_lb is None else base_lb
        src_ub = self.orig_ub[layer_name] if base_ub is None else base_ub
        rep = [B] + [1] * (src_lb.dim() - 1)
        pin_lb, pin_ub = src_lb.repeat(*rep), src_ub.repeat(*rep)
        flat_lb, flat_ub = pin_lb.view(B, -1), pin_ub.view(B, -1)
        for j, (nidx, sign) in enumerate(probes):
            if sign > 0:
                flat_lb[j, nidx] = 0.  # pin ACTIVE (z >= 0)
            else:
                flat_ub[j, nidx] = 0.  # pin INACTIVE (z <= 0)
        return pin_lb, pin_ub

    def _run_probe_chunk(self, layer_name, pairs, rung):
        """One batched crown/alpha probe chunk over neuron PAIRS
        (active at 2j, inactive at 2j+1). Returns [B, k] output lower
        bounds; the crown rung folds hulls / edges / cut candidates as a
        side effect."""
        B = 2 * len(pairs)
        pin_order = self.node_order[layer_name]
        probes = []
        for nidx in pairs:
            probes.append((nidx, +1))
            probes.append((nidx, -1))

        # rung 'crown': fix layers up to the pin, RECOMPUTE everything
        # downstream (hull refinement source). rung 'alpha': the same
        # downstream recomputation, with the build()-time alphas reused for
        # the recomputed intermediate start nodes -- available because those
        # alphas were retained through build()'s alpha_drop_unused() (see
        # LiRPANet.alpha_drop_unused; gated on phase probing). Without
        # retention the alpha rung falls back to fixing ALL layers (a
        # one-split BaB domain view) and strengthens the OUTPUT margin only.
        recompute_downstream = rung == 'crown' or (
            rung == 'alpha' and self.alpha_interm_retained)
        probe_ib = {}
        for k in self.interm_names:
            if recompute_downstream and self.node_order[k] > pin_order:
                continue  # downstream: recomputed under the clamp
            if k == layer_name:
                continue
            probe_ib[k] = [
                self.orig_lb[k].expand(B, *self.orig_lb[k].shape[1:]),
                self.orig_ub[k].expand(B, *self.orig_ub[k].shape[1:]),
            ]
        probe_ib[layer_name] = list(self._pin_bounds(layer_name, probes))

        sel = self.sel if rung == 'alpha' else None
        new_x, C = self._probe_x_c(B, sel)
        with torch.no_grad():
            # rung 'crown': plain CROWN. rung 'alpha': reuse the optimized
            # final-start-node alphas left on the net by model.build() (the
            # BaB update_bounds shortcut pattern); the stored alphas have
            # batch dim 1 and broadcast over the probe batch.
            lb_out = self.net.compute_bounds(
                x=(new_x,), C=C, method='backward',
                reuse_alpha=(rung == 'alpha'),
                interm_bounds=probe_ib, bound_upper=False)[0]
            if recompute_downstream:
                self._fold_downstream(layer_name, pairs, B, pin_order)
        return lb_out

    def _fold_downstream(self, layer_name, pairs, B, pin_order):
        """Fold hull refinement of downstream layers into ref_lb/ref_ub;
        collect per-pair hull gains, implication edges and implied-bound cut
        candidates. All facts here are box-sound (each phase bound is valid
        on its phase region; the hull covers the whole box)."""
        npairs = len(pairs)
        gain = torch.zeros(npairs, device=self.device)
        for k in self.interm_names:
            if self.node_order[k] <= pin_order:
                continue
            node = self.net[k]
            dl, du = node.lower, node.upper
            if dl is None or du is None or dl.shape[0] != B \
                    or dl.shape[1:] != self.orig_lb[k].shape[1:]:
                continue  # not recomputed in this pass
            dl = dl.detach().view(npairs, 2, *dl.shape[1:])
            du = du.detach().view(npairs, 2, *du.shape[1:])
            hull_l = dl.min(dim=1).values
            hull_u = du.max(dim=1).values
            width = (self.orig_ub[k] - self.orig_lb[k]).clamp(min=1e-12)
            red = ((hull_l - self.orig_lb[k]).clamp(min=0)
                   + (self.orig_ub[k] - hull_u).clamp(min=0)) / width
            gain = torch.maximum(gain, red.view(npairs, -1).max(dim=1).values)
            self.ref_lb[k] = torch.maximum(
                self.ref_lb[k], hull_l.max(dim=0, keepdim=True).values)
            self.ref_ub[k] = torch.minimum(
                self.ref_ub[k], hull_u.min(dim=0, keepdim=True).values)
            if k in self.unstable_mask:
                self._collect_edges_and_cuts(
                    [(layer_name, n) for n in pairs], k, dl, du, width)
        for p, nidx in enumerate(pairs):
            key = (layer_name, nidx)
            self.pair_gain[key] = max(self.pair_gain.get(key, 0.),
                                      float(gain[p]))

    def _collect_edges_and_cuts(self, pair_keys, k, dl, du, width):
        """dl/du: [npairs, 2, ...] downstream bounds of layer k under the
        probes of pair_keys (a list of (pin_layer, pin_idx), dim 1 of
        dl/du: 0=active, 1=inactive). Restricted to the ORIGINALLY unstable
        neurons of layer k. pair_keys may mix pin layers (wide root rung)."""
        npairs = len(pair_keys)
        um = self.unstable_mask[k]
        if not bool(um.any()):
            return
        fl = dl.reshape(npairs, 2, -1)[:, :, um]
        fu = du.reshape(npairs, 2, -1)[:, :, um]
        uidx = um.nonzero().reshape(-1)

        # Implication edges: pin (i, phase) makes neuron j of layer k stable.
        if self._num_edges < 1_000_000:
            for side, sign in ((0, +1), (1, -1)):
                for cond, tgt_sign in (((fl[:, side, :] >= 0), +1),
                                       ((fu[:, side, :] <= 0), -1)):
                    for p, j in cond.nonzero().tolist():
                        src = (*pair_keys[p], sign)
                        self.implications.setdefault(src, set()).add(
                            (k, int(uidx[j]), tgt_sign))
                        self._num_edges += 1

        # Implied-bound cut candidates (see emit_implied_cuts for semantics).
        if not self.collect_cuts:
            return
        gap_frac = self.cfg['implied_cuts_gap_frac']
        w = width.reshape(-1)[um]
        for kind, arr in (('ub', fu), ('lb', fl)):
            b_act, b_inact = arr[:, 0, :], arr[:, 1, :]
            gapf = (b_act - b_inact).abs() / w
            for p, j in (gapf >= gap_frac).nonzero().tolist():
                key = (*pair_keys[p], k, int(uidx[j]), kind)
                rec = (float(gapf[p, j]), float(b_inact[p, j]),
                       float(b_act[p, j]))
                old = self.cut_cands.get(key)
                if old is None or rec[0] > old[0]:
                    self.cut_cands[key] = rec

    # ------------------------------------------------------------------
    # Alpha retention management (see LiRPANet.alpha_drop_unused)
    # ------------------------------------------------------------------

    def trim_retained_alphas(self, candidates):
        """Narrow the intermediate-start-node alphas retained through
        build() (sentinel 'all') to the start nodes the alpha rung can
        actually use: intermediate nodes strictly downstream of the
        earliest probed layer -- the only nodes ever recomputed under a
        pin. Frees the rest BEFORE the memory-heavy batched probes run.
        candidates: list of (layer_name, nidx, score)."""
        if not self.alpha_interm_retained:
            return
        min_pin = min(self.node_order[name] for name, _, _ in candidates)
        keep = {k for k in self.interm_names
                if self.node_order[k] > min_pin}
        self.model.phase_probing_keep_alpha_nodes = keep
        self.model.alpha_drop_unused()
        print(f'Phase probing: retaining intermediate-start-node alphas of '
              f'{len(keep)} layers for alpha-rung downstream recomputation.')

    # ------------------------------------------------------------------
    # Rung: crown / alpha (batched pairs, downstream recomputation)
    # ------------------------------------------------------------------

    def _time_rung(self, rung, wide=False):
        """Context manager accumulating wall clock into stats['rung_time'].
        Distinguishes the wide and per-layer paths so a run that probes wide
        at crown and escalates per-layer at alpha is attributable."""
        key = f'{rung}{"-wide" if wide else ""}'
        stats = self.stats

        class _T:
            def __enter__(_s):
                _s.t0 = time.time()
                return _s

            def __exit__(_s, *exc):
                stats['rung_time'][key] = (
                    stats['rung_time'].get(key, 0.) + time.time() - _s.t0)
                return False
        return _T()

    def run_pair_rung(self, rung, pairs_by_layer):
        """Run all probes of `rung` ('crown' or 'alpha') over
        pairs_by_layer: {layer_name: [nidx, ...]}. Updates probe_state."""
        with self._time_rung(rung):
            self._run_pair_rung(rung, pairs_by_layer)

    def _run_pair_rung(self, rung, pairs_by_layer):
        sel = self.sel if rung == 'alpha' else None
        pairs_per_chunk = max(1, max(2, self.cfg['batch_size']) // 2)
        for layer_name in sorted(pairs_by_layer,
                                 key=lambda n: self.node_order[n]):
            neurons = pairs_by_layer[layer_name]
            pos, cur = 0, pairs_per_chunk
            torch.cuda.empty_cache()
            while pos < len(neurons):
                chunk = neurons[pos:pos + cur]
                try:
                    lb_out = self._run_probe_chunk(layer_name, chunk, rung)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if not _is_oom(e):
                        raise
                    torch.cuda.empty_cache()
                    if cur == 1:
                        raise
                    cur = max(1, cur // 2)
                    print(f'Phase probing: CUDA OOM, reducing probe chunk to '
                          f'{2 * cur} probes.')
                    continue
                self._record_scores([(layer_name, n) for n in chunk],
                                    self.checker.score(lb_out, sel), rung)
                pos += len(chunk)
                if self.both_verified is not None:
                    return

    def _record_scores(self, pair_keys, scores, rung):
        """Fold one chunk's output-margin scores into probe_state.
        pair_keys: list of (layer, nidx), one per PAIR; scores: [2*npairs]
        laid out active at 2j, inactive at 2j+1."""
        for j, (layer_name, nidx) in enumerate(pair_keys):
            for off, sign in ((0, +1), (1, -1)):
                s = float(scores[2 * j + off])
                key = (layer_name, nidx, sign)
                st = self.probe_state.setdefault(
                    key, {'score': float('-inf'), 'verified': False,
                          'rung': rung})
                if rung != 'crown' and st['score'] > float('-inf'):
                    # track how much the stronger oracle moved the probe's
                    # output margin (reporting only)
                    self._margin_gains.append(max(0., s - st['score']))
                st['score'] = max(st['score'], s)
                st['verified'] |= s > 0
                st['rung'] = rung
            if self.probe_state[(layer_name, nidx, +1)]['verified'] \
                    and self.probe_state[(layer_name, nidx, -1)]['verified']:
                self.both_verified = (layer_name, nidx)
        if rung == 'crown':
            self.stats['probed_neurons'] += len(pair_keys)

    # ------------------------------------------------------------------
    # Rung: wide crown (cross-layer batch, per-row clamp hook)
    # ------------------------------------------------------------------
    #
    # The per-layer rung above is forced to issue one compute_bounds call
    # per pinned layer because interm_bounds is a PER-CALL dict: a layer is
    # either fixed for the whole batch or free for the whole batch, and a
    # probe pinned at layer L needs L fixed-with-clamp while everything
    # below L is free. Two probes at different depths cannot agree on that.
    #
    # The wide rung sidesteps it by not expressing the clamp through
    # interm_bounds at all. Every layer from the earliest pin onward is left
    # FREE, and the clamp is written into node.lower/node.upper for the
    # owning rows by a hook on clamp_interim_bounds, which BoundedModule
    # calls at the end of compute_intermediate_bounds (bound_general.py, in
    # both the freshly-computed and the cached-bounds paths) just before
    # node.interval is set. Because intermediate bounds are computed in
    # topological order, a row's clamp is installed before anything
    # downstream of it is bounded, so each row still gets full downstream
    # recomputation under its own pin -- the strong oracle -- while rows
    # pinned at different depths share a single backward pass.
    #
    # Soundness is the same restriction the per-layer rung makes, applied at
    # a later point in the same computation: setting lb=0 (resp. ub=0) for
    # one row narrows that row's region to the pinned phase, and rows never
    # interact because every downstream op is batch-elementwise.
    #
    # The same hook implements the depth window: for a row whose pin is more
    # than `wide_root_window` positions upstream of the current node, the
    # ROOT bounds are written back for that row, capping per-probe cost at
    # the window size instead of the whole remaining network. Writing back a
    # bound that is valid on the entire input box is always sound; it only
    # gives up tightness.

    def _install_clamp_hooks(self, rows_at, row_pin_order, B, window, min_pin):
        """Patch clamp_interim_bounds on every layer that needs per-row
        surgery and register it in layers_with_constraint so BoundedModule
        actually calls it. Returns an undo callable."""
        net = self.net
        prev_constraint = list(net.layers_with_constraint)
        patched = []
        for k in self.interm_names:
            ko = self.layer_pos[k]
            if ko < min_pin:
                continue  # fixed via interm_bounds; never recomputed
            clamps = rows_at.get(k)
            restore = None
            if window >= 0:
                m = (row_pin_order + window) < ko
                if bool(m.any()):
                    restore = m
            # Every freed layer is hooked, not just the pinned ones: the
            # root-bound intersection in the hook is what keeps a freed
            # layer from being recomputed LOOSER than the root value that
            # the per-layer rung would have fed in through interm_bounds.
            node = net[k]
            # Pre-split into (rows, indices) index tensors per phase so the
            # hook does two vectorized index_puts instead of one GPU sync
            # per clamped neuron (thousands, at full coverage).
            node._probe_clamps = _clamp_index(clamps, self.device)
            node._probe_restore = restore
            node._probe_orig = (self.orig_lb[k], self.orig_ub[k])
            node._probe_batch = B
            node.clamp_interim_bounds = _make_clamp_hook(node)
            patched.append(node)
            if k not in net.layers_with_constraint:
                net.layers_with_constraint.append(k)

        def undo():
            net.layers_with_constraint = prev_constraint
            for n in patched:
                # the patch is an INSTANCE attribute shadowing the class
                # method; deleting it restores the original no-op.
                n.__dict__.pop('clamp_interim_bounds', None)
                for a in ('_probe_clamps', '_probe_restore', '_probe_orig',
                          '_probe_batch'):
                    n.__dict__.pop(a, None)
        return undo

    def _run_wide_chunk(self, pairs, rung):
        """One cross-layer probe chunk. pairs: list of (layer_name, nidx);
        row 2j pins pair j ACTIVE, row 2j+1 pins it INACTIVE. Returns
        [B, k] output lower bounds and folds the downstream hull."""
        npairs = len(pairs)
        B = 2 * npairs
        window = self.wide_root_window
        # Depths are LAYER positions (index into the topologically sorted
        # intermediate layers), so a window of 1 means exactly one
        # intermediate layer past the pin. See layer_pos in __init__.
        pin_order = torch.tensor([self.layer_pos[ln] for ln, _ in pairs],
                                 device=self.device)
        min_pin = int(pin_order.min())
        max_pin = int(pin_order.max())
        row_pin_order = pin_order.repeat_interleave(2)

        # Only the band [min_pin, max_pin + window] is freed and recomputed
        # under the hooks; everything outside it is fixed at the root values
        # through interm_bounds:
        #   * strictly upstream of the EARLIEST pin -- no clamp in this chunk
        #     can affect it;
        #   * beyond the LATEST pin plus the window -- outside every row's
        #     window, so the hook would only overwrite it with the root
        #     values anyway. Fixing it here means it is never computed at
        #     all, which is where the window's cost saving actually comes
        #     from. Sorting the chunk by depth keeps this band narrow.
        far = (max_pin + window) if window >= 0 else None
        # The PINNED layers are fixed at their root values too, not freed.
        # Recomputing a pinned layer is pure waste: the hook clamps it and
        # intersects it back against the root immediately afterwards, so
        # CROWN can only ever return something looser than what we overwrite
        # it with. It is also the dominant cost -- measured on tinyimagenet,
        # a pins=[0,0] chunk freed 2 layers totalling 1390 unstable neurons,
        # most of them the pin layer's own. Fixing it here still delivers the
        # clamp: BoundedModule calls clamp_interim_bounds on the
        # already-current path too (bound_general.py, the
        # is_lower_bound_current early return), which is where a layer passed
        # in interm_bounds lands.
        pinned_names = {ln for ln, _ in pairs}
        probe_ib = {}
        for k in self.interm_names:
            ko = self.layer_pos[k]
            if k in pinned_names or ko < min_pin \
                    or (far is not None and ko > far):
                probe_ib[k] = [
                    self.orig_lb[k].expand(B, *self.orig_lb[k].shape[1:]),
                    self.orig_ub[k].expand(B, *self.orig_ub[k].shape[1:]),
                ]

        rows_at = {}
        for j, (ln, nidx) in enumerate(pairs):
            rows_at.setdefault(ln, []).append((2 * j, nidx, +1))
            rows_at[ln].append((2 * j + 1, nidx, -1))

        # Cost anatomy (PHASE_PROBING_CHUNK_DIAG=1). The hypothesis under test
        # is that wide-root cost is QUADRATIC in the unstable count: a freed
        # layer's bounds are computed by a backward pass whose spec dimension
        # is that layer's OWN unstable count, so per chunk the work is
        # B x unstable_freed, and with total/P chunks the total is
        # 2 x total_unstable x unstable_freed. If that holds, no amount of
        # chunk/window/sparsity tuning fixes tinyimagenet -- only removing the
        # per-freed-layer backward pass does.
        diag = os.environ.get('PHASE_PROBING_CHUNK_DIAG') == '1'
        if diag:
            freed = [k for k in self.interm_names if k not in probe_ib]
            n_unstable_freed = sum(
                int(self.unstable_mask[k].sum()) for k in freed
                if k in self.unstable_mask)
            torch.cuda.synchronize()
            t_chunk = time.time()

        undo = self._install_clamp_hooks(rows_at, row_pin_order, B, window,
                                         min_pin)
        try:
            sel = self.sel if rung == 'alpha' else None
            new_x, C = self._probe_x_c(B, sel)
            # Sparse intermediate bounds: without aux_reference_bounds
            # auto_LiRPA falls back to IBP to decide which neurons of a freed
            # layer are unstable (get_ref_intermediate_bounds), and IBP is far
            # looser than the root's alpha-CROWN bounds -- so it computes a
            # dense superset. Handing it the ROOT bounds restricts the
            # recomputation to neurons that are actually unstable there, which
            # is the same assumption BaB makes; a neuron stable at the root
            # stays stable under a clamp, since the hook intersects every
            # freed layer back against the root anyway.
            # Measured on cifar100 idx0 at full coverage: 153.2s -> 23.4s, but
            # NOT a pure speedup -- neurons stable at the root stop being
            # recomputed, so their refinement is lost (edges 502 -> 130) and
            # the looser probe bounds collapse the escalation set (2816 -> 4).
            # Toggle so cost and fact loss can be separated.
            aux_ref = {k: [self.orig_lb[k].expand(B, *self.orig_lb[k].shape[1:]),
                           self.orig_ub[k].expand(B, *self.orig_ub[k].shape[1:])]
                       for k in self.interm_names
                       if k not in probe_ib} if self.wide_root_sparse else None
            with torch.no_grad():
                if self.wide_root_interm_only:
                    # INTERMEDIATE-ONLY. compute_bounds is
                    #     check_prior_bounds(final, C)   <- computes the freed
                    #                                       layer's bounds
                    #     backward_general(final, C)     <- full-network output
                    #                                       pass
                    # and only the first produces what window=1 harvests (hull
                    # refinement, implication edges, implied cuts). Measured on
                    # tinyimagenet: the output pass costs ~0.82s per B=128
                    # chunk -- essentially the entire 28.7s -- and buys 2
                    # forced phases out of 1925 probed neurons.
                    # Retargeting `final` at the deepest freed layer's consumer
                    # makes _set_used_nodes prune everything downstream, so the
                    # backward pass runs over a short path with a spec of 1
                    # instead of the whole network with the full spec.
                    # No output bound => no probe scores, no forced phases, no
                    # escalation; the hull/edge/cut channel is unaffected.
                    lb_out = None
                    deepest = max((k for k in self.interm_names
                                   if k not in probe_ib),
                                  key=lambda k: self.layer_pos[k], default=None)
                    if deepest is not None:
                        tgt = self.net[deepest].output_name[0]
                        shape = self.net[tgt].output_shape[1:]
                        dummy_c = torch.zeros(B, 1, *shape, device=self.device)
                        self.net.compute_bounds(
                            x=(new_x,), C=dummy_c, method='backward',
                            final_node_name=tgt,
                            interm_bounds=probe_ib, bound_upper=False,
                            aux_reference_bounds=aux_ref)
                        self._fold_wide(pairs, pin_order, B, window)
                else:
                    lb_out = self.net.compute_bounds(
                        x=(new_x,), C=C, method='backward',
                        reuse_alpha=(rung == 'alpha'),
                        interm_bounds=probe_ib, bound_upper=False,
                        aux_reference_bounds=aux_ref)[0]
                    self._fold_wide(pairs, pin_order, B, window)
        finally:
            undo()
        if diag:
            torch.cuda.synchronize()
            dt = time.time() - t_chunk
            work = B * max(1, n_unstable_freed)
            print(f'CHUNKDIAG B={B} pins=[{min_pin},{max_pin}] '
                  f'freed={len(freed)} unstable_freed={n_unstable_freed} '
                  f'work={work} t={dt:.4f}s us_per_work={1e6 * dt / work:.4f} '
                  f'clamps={_CLAMPS_APPLIED}')
        return lb_out

    def _fold_wide(self, pairs, pin_order, B, window):
        """Hull fold for a cross-layer chunk. A pair only contributes at
        layer k when k is genuinely downstream of THAT pair's pin and inside
        its window -- otherwise the row carries no clamp at k (or was
        written back to root) and there is nothing to learn from it."""
        npairs = len(pairs)
        gain = torch.zeros(npairs, device=self.device)
        for k in self.interm_names:
            ko = self.layer_pos[k]
            act = pin_order < ko
            if window >= 0:
                act = act & ((pin_order + window) >= ko)
            if not bool(act.any()):
                continue
            node = self.net[k]
            dl, du = node.lower, node.upper
            if dl is None or du is None or dl.shape[0] != B \
                    or dl.shape[1:] != self.orig_lb[k].shape[1:]:
                continue  # not recomputed in this pass
            idx = act.nonzero().reshape(-1)
            dl = dl.detach().view(npairs, 2, *dl.shape[1:])[idx]
            du = du.detach().view(npairs, 2, *du.shape[1:])[idx]
            hull_l = dl.min(dim=1).values
            hull_u = du.max(dim=1).values
            width = (self.orig_ub[k] - self.orig_lb[k]).clamp(min=1e-12)
            red = ((hull_l - self.orig_lb[k]).clamp(min=0)
                   + (self.orig_ub[k] - hull_u).clamp(min=0)) / width
            gain[idx] = torch.maximum(
                gain[idx], red.view(idx.numel(), -1).max(dim=1).values)
            self.ref_lb[k] = torch.maximum(
                self.ref_lb[k], hull_l.max(dim=0, keepdim=True).values)
            self.ref_ub[k] = torch.minimum(
                self.ref_ub[k], hull_u.min(dim=0, keepdim=True).values)
            if k in self.unstable_mask:
                self._collect_edges_and_cuts(
                    [pairs[i] for i in idx.tolist()], k, dl, du, width)
        for p, key in enumerate(pairs):
            self.pair_gain[key] = max(self.pair_gain.get(key, 0.),
                                      float(gain[p]))

    def run_wide_rung(self, rung, candidates):
        """Cross-layer replacement for run_pair_rung. candidates: list of
        (layer, nidx, score)."""
        with self._time_rung(rung, wide=True):
            self._run_wide_rung(rung, candidates)

    def _run_wide_rung(self, rung, candidates):
        pairs_all, seen = [], set()
        for (name, nidx, _) in candidates:
            if (name, nidx) in seen:
                continue
            seen.add((name, nidx))
            pairs_all.append((name, nidx))
        # Topological order, so a chunk spans a contiguous depth range and
        # the freed-layer set (everything from the chunk's earliest pin on)
        # stays as small as possible.
        pairs_all.sort(key=lambda t: (self.node_order[t[0]], t[1]))
        # Adaptive probe budget. Full coverage is affordable in absolute terms
        # but NOT relative to every config's per-instance timeout: measured on
        # tinyimagenet (100s budget), the probe ran to a 30.2s median / 77.0s
        # max and turned 16 safe instances into unknown for zero gains
        # (-16 solved, 1.479x time), while cheap-probe configs gained. Capping
        # probe wall clock at a fraction of the instance timeout degrades
        # COVERAGE gracefully instead of eating the search's budget.
        # Truncating is legitimate here precisely because chunks are
        # depth-sorted and every chunk is self-contained: stopping early
        # yields a smaller but still valid probe set. (On the per-layer path
        # the same truncation would bias coverage toward the earliest layers.)
        budget = None
        frac = float(self.cfg.get('time_budget_frac', 0.) or 0.)
        if frac > 0:
            timeout = float(arguments.Config['bab']['timeout'])
            budget = frac * timeout
        t_start = time.time()
        per_chunk = max(1, max(2, self.cfg['batch_size']) // 2)
        pos, cur = 0, per_chunk
        sel = self.sel if rung == 'alpha' else None
        n_layers = len({ln for ln, _ in pairs_all})
        print(f'Phase probing: wide root rung over {len(pairs_all)} neurons '
              f'across {n_layers} layers '
              f'(window={self.wide_root_window}, chunk={2 * per_chunk}).')
        torch.cuda.empty_cache()
        # Cap how many layer positions one chunk may span. The freed band is
        # [min_pin, max_pin + window], so a chunk straddling distant layers
        # frees everything between them: measured on tinyimagenet, one
        # pins=[0,9] chunk freed 10 layers / 1925 unstable neurons, costing
        # what ~10 well-formed chunks cost. Allowing a span of 1 still lets
        # thin adjacent layers share a batch (the point of the wide rung)
        # without ever opening a wide band.
        max_span = 1
        while pos < len(pairs_all):
            if budget is not None and time.time() - t_start > budget:
                # Always report what was dropped: a silently truncated pass
                # reads as "full coverage found nothing" in the summary.
                self.stats['coverage_truncated'] = len(pairs_all) - pos
                print(f'Phase probing: probe budget {budget:.1f}s exhausted, '
                      f'stopping at {pos}/{len(pairs_all)} neurons '
                      f'({100. * pos / len(pairs_all):.0f}% coverage).')
                break
            chunk = pairs_all[pos:pos + cur]
            # Truncate the chunk at the first pair that would widen the band
            # past max_span (pairs_all is depth-sorted, so this is a prefix).
            base = self.layer_pos[chunk[0][0]]
            for m, (ln, _) in enumerate(chunk):
                if self.layer_pos[ln] - base > max_span:
                    chunk = chunk[:m]
                    break
            try:
                lb_out = self._run_wide_chunk(chunk, rung)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _chunk_too_big(e):
                    raise
                torch.cuda.empty_cache()
                if cur == 1:
                    raise
                cur = max(1, cur // 2)
                print(f'Phase probing: chunk too large ({type(e).__name__}), '
                      f'reducing wide probe chunk to {2 * cur} probes.')
                continue
            if lb_out is not None:
                self._record_scores(chunk, self.checker.score(lb_out, sel), rung)
            elif rung == 'crown':
                # interm-only skips _record_scores (no output bound to score),
                # which is where probed_neurons is normally counted. Count it
                # here or the summary reports probed=0 on a pass that probed
                # everything.
                self.stats['probed_neurons'] += len(chunk)
            pos += len(chunk)
            if self.both_verified is not None:
                return

    # ------------------------------------------------------------------
    # Rung: beta (one-split domains, output bound only)
    # ------------------------------------------------------------------

    def _saved_final_alphas(self):
        saved = {}
        for m in self.net.get_enabled_opt_act():
            a = m.alpha.get(self.final_name, None) \
                if hasattr(m, 'alpha') else None
            if isinstance(a, torch.Tensor):
                saved[m.name] = a
        return saved

    def _run_beta_chunk(self, layer_name, probes, saved_alphas):
        """One batched beta-CROWN probe chunk. probes: [(nidx, sign), ...].
        The pin is expressed BOTH as a clamped intermediate bound and as a
        one-entry split history whose SparseBeta enforces the split
        constraint via its Lagrangian (exactly a one-split BaB domain)."""
        B = len(probes)
        # Split history format (see LiRPANet.empty_history):
        # {layer: (loc, sign, bias, score, depth)}; sign +1 = active.
        empty = {layer.name: ([], [], [], [], [])
                 for layer in self.net.split_nodes}
        history = []
        for nidx, sign in probes:
            h = dict(empty)
            h[layer_name] = ([nidx], [sign], [0.], [0.], [1])
            history.append(h)
        d = {'history': history, 'betas': [None] * B}
        beta_data, _ = BetaFullData.from_domain_dict(
            d, bias=True, device=self.device)
        beta_data.attach_to_net(self.model)

        # Expand the final-start-node alphas from the initial pass to the
        # probe batch (intermediate bounds are fixed, so only these are used).
        for m in self.net.get_enabled_opt_act():
            if m.name in saved_alphas:
                a = saved_alphas[m.name].detach()
                rep = [1, 1, B] + [1] * (a.dim() - 3)
                m.alpha[self.final_name] = a.repeat(*rep).requires_grad_(True)

        # All intermediate bounds fixed (current refined values), pin clamped.
        probe_ib = {}
        for k in self.interm_names:
            if k == layer_name:
                continue
            probe_ib[k] = [
                self.ref_lb[k].expand(B, *self.ref_lb[k].shape[1:]),
                self.ref_ub[k].expand(B, *self.ref_ub[k].shape[1:]),
            ]
        probe_ib[layer_name] = list(self._pin_bounds(
            layer_name, probes,
            base_lb=self.ref_lb[layer_name], base_ub=self.ref_ub[layer_name]))

        new_x, C = self._probe_x_c(B, self.sel)

        def never_stop(x):
            # Run all optimization iterations; per-element early stopping
            # would need the OR-group logic which the optimizer cannot use.
            return torch.zeros(x.shape[0], 1, dtype=torch.bool,
                               device=x.device)

        self.net.set_bound_opts({
            'optimize_bound_args': {
                'enable_beta_crown': True,
                'fix_interm_bounds': True,
                'stop_criterion_func': never_stop,
                'multi_spec_keep_func': None,
                'iteration':
                    arguments.Config['solver']['beta-crown']['iteration'],
            },
            'enable_opt_interm_bounds': False,
        })
        self.model.set_crown_bound_opts('beta')
        with torch.enable_grad():
            # No decision_thresh: we run all iterations (never_stop) and do
            # not want the optimizer's spec pruning, which would need a
            # multi_spec_keep_func matching our OR-group semantics.
            lb_out = self.net.compute_bounds(
                x=(new_x,), C=C, method='CROWN-optimized',
                interm_bounds=probe_ib, bound_upper=False)[0]
        return lb_out.detach()

    def run_beta_rung(self, probe_keys):
        saved_alphas = self._saved_final_alphas()
        if not saved_alphas:
            print('Phase probing: no stored alphas, skipping beta rung.')
            return
        # Stash every non-final-start-node alpha (in particular the
        # intermediate-start-node alphas retained for the alpha rung) for
        # the duration of the beta rung: all intermediate bounds are fixed
        # here so they are never used, but the optimizer's best-alpha
        # bookkeeping (_update_optimizable_activations) indexes ALL alpha
        # entries by the probe batch dim, and the retained entries have
        # batch dim 1 (build state) -> device-side assert.
        acts = {m.name: m for m in self.net.get_enabled_opt_act()}
        stashed = {}
        for name, m in acts.items():
            for spec in [s for s in m.alpha if s != self.final_name]:
                stashed.setdefault(name, {})[spec] = m.alpha.pop(spec)
        by_layer = {}
        for (layer, nidx, sign) in probe_keys:
            by_layer.setdefault(layer, []).append((nidx, sign))
        chunk_size = max(2, self.cfg['batch_size'])
        try:
            for layer_name in sorted(by_layer,
                                     key=lambda n: self.node_order[n]):
                probes = by_layer[layer_name]
                pos, cur = 0, chunk_size
                torch.cuda.empty_cache()
                while pos < len(probes):
                    chunk = probes[pos:pos + cur]
                    try:
                        lb_out = self._run_beta_chunk(
                            layer_name, chunk, saved_alphas)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if not _is_oom(e):
                            raise
                        torch.cuda.empty_cache()
                        if cur == 1:
                            raise
                        cur = max(1, cur // 2)
                        print('Phase probing: CUDA OOM in beta rung, '
                              f'reducing chunk to {cur} probes.')
                        continue
                    scores = self.checker.score(lb_out, self.sel)
                    for j, (nidx, sign) in enumerate(chunk):
                        st = self.probe_state[(layer_name, nidx, sign)]
                        st['score'] = max(st['score'], float(scores[j]))
                        st['verified'] |= float(scores[j]) > 0
                        st['rung'] = 'beta'
                        other = self.probe_state.get((layer_name, nidx, -sign))
                        if st['verified'] and other and other['verified']:
                            self.both_verified = (layer_name, nidx)
                    pos += len(chunk)
                    if self.both_verified is not None:
                        return
        finally:
            # Restore the batch-1 alphas (final AND stashed non-final) and
            # turn beta off again so the final recompute (and any later
            # consumer) sees the build state.
            for m in self.net.get_enabled_opt_act():
                if m.name in saved_alphas:
                    m.alpha[self.final_name] = saved_alphas[m.name]
            for name, entries in stashed.items():
                acts[name].alpha.update(entries)
            self.net.set_bound_opts({
                'optimize_bound_args': {'enable_beta_crown': False}})

    # ------------------------------------------------------------------
    # Rung: gurobi (exact MILP per probe)
    # ------------------------------------------------------------------

    def _get_gurobi(self):
        """Build (once) the Gurobi MILP over the ORIGINAL bounds. Returns
        (grb, model, out_vars) or None if unavailable (graceful skip; this
        activates automatically once a valid license is present)."""
        if self._gurobi is not None:
            return self._gurobi or None
        try:
            import gurobipy as grb
            t_build = time.time()
            x1 = expand_batch(self.x, 1, device=self.device)
            interm_bounds = {k: (self.orig_lb[k], self.orig_ub[k])
                             for k in self.interm_names}
            if getattr(self.net, 'solver_model', None) is not None:
                self.net._reset_solver_model()
                self.net._reset_solver_vars(self.net.final_node())
            out_vars = self.net.build_solver_module(
                x=(x1,), C=self.c, interm_bounds=interm_bounds,
                model_type='mip', solver_pkg='gurobi')
            m = self.net.solver_model
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 30.)
            # Leave CPU headroom for other work on the box; the MILP is
            # CPU-side while the rest of probing is on the GPU.
            m.setParam('Threads', min(8, os.cpu_count() or 8))
            m.update()
            self._gurobi = (grb, m, _flatten_solver_vars(out_vars, []))
            print(f'Phase probing: built gurobi MILP '
                  f'({m.NumVars} vars, {m.NumConstrs} constrs, '
                  f'{m.NumBinVars} binaries) in '
                  f'{time.time() - t_build:.1f}s.')
        except Exception as e:  # noqa: BLE001 - MILP is best-effort by design
            print(f'Phase probing: Gurobi MILP unavailable ({e}), skipping '
                  'gurobi rung / MIP confirmation.')
            self._gurobi = False
            return None
        return self._gurobi

    def _gurobi_region_verified(self, layer_name, nidx, region_sign):
        """Exactly check that the region pinned to region_sign (+1: z >= 0,
        -1: z <= 0) contains no counterexample. Returns True/False/None
        (None = gurobi unavailable / error)."""
        g = self._get_gurobi()
        if g is None:
            return None
        grb, m, out_vars = g
        num_clause = self.c.shape[1]
        rhs_flat = self.rhs.reshape(-1).tolist()
        c_cpu = self.c[0].detach().cpu()

        def clause_obj(j):
            if len(out_vars) == num_clause:
                # C was merged into the last linear layer: vars are margins.
                return out_vars[j]
            # C was not merged (non-linear last layer): build c_j y manually.
            expr = grb.LinExpr()
            for o, var in enumerate(out_vars):
                coeff = float(c_cpu[j, o])
                if coeff != 0.:
                    expr += coeff * var
            return expr

        or_ids = self.checker.or_ids.tolist()
        groups = [g_ for g_ in range(self.checker.num_groups)
                  if not bool(self.checker.verified_or[g_])]
        clauses_by_group = {g_: [] for g_ in groups}
        for j, g_ in enumerate(or_ids):
            if g_ in clauses_by_group:
                clauses_by_group[g_].append(j)

        try:
            pre_vars = _flatten_solver_vars(
                self.net[layer_name].solver_vars, [])
            z = pre_vars[nidx]
            pin = m.addConstr(z >= 0 if region_sign > 0 else z <= 0,
                              name='phase_probe_pin')
            m.update()
            try:
                region_empty, all_ok = False, True
                for g_ in groups:
                    group_ok = False
                    for j in clauses_by_group[g_]:
                        m.setObjective(clause_obj(j), grb.GRB.MINIMIZE)
                        m.optimize()
                        if m.status == grb.GRB.INFEASIBLE:
                            region_empty = True  # empty region: trivially ok
                            break
                        # ObjBound is a valid lower bound on the minimum even
                        # if the time limit was hit.
                        try:
                            obj_bound = m.ObjBound
                        except (AttributeError, grb.GurobiError):
                            obj_bound = float('-inf')
                        if obj_bound > rhs_flat[j]:
                            group_ok = True
                            break
                    if region_empty:
                        break
                    if not group_ok:
                        all_ok = False
                        break
                return region_empty or all_ok
            finally:
                m.remove(pin)
                m.update()
        except Exception as e:  # noqa: BLE001 - MILP is best-effort by design
            print(f'Phase probing: gurobi solve error for neuron {nidx} in '
                  f'{layer_name} ({e}).')
            return None

    def run_gurobi_rung(self, probe_keys):
        # Measurement cap: PHASE_PROBING_GUROBI_MAX=N limits how many
        # escalated probes are solved exactly (0 / unset = no cap). Useful
        # when the per-probe MILP cost is high.
        cap = int(os.environ.get('PHASE_PROBING_GUROBI_MAX', '0'))
        if cap > 0 and len(probe_keys) > cap:
            print(f'Phase probing: capping gurobi rung to first {cap} of '
                  f'{len(probe_keys)} escalated probes '
                  '(PHASE_PROBING_GUROBI_MAX).')
            probe_keys = probe_keys[:cap]
        t0, solved, num_verified = time.time(), 0, 0
        try:
            for (layer, nidx, sign) in probe_keys:
                t1 = time.time()
                res = self._gurobi_region_verified(layer, nidx, sign)
                print(f'Phase probing: gurobi probe ({layer}, {nidx}, '
                      f'{sign:+d}) -> {res} in {time.time() - t1:.1f}s')
                if res is None:
                    return  # gurobi unavailable; skip the rest gracefully
                solved += 1
                num_verified += bool(res)
                st = self.probe_state[(layer, nidx, sign)]
                st['rung'] = 'gurobi'
                if res:
                    st['verified'] = True
                    other = self.probe_state.get((layer, nidx, -sign))
                    if other and other['verified']:
                        self.both_verified = (layer, nidx)
                        return
        finally:
            el = time.time() - t0
            print(f'Phase probing: gurobi rung solved {solved} probes '
                  f'({num_verified} region-verified) in {el:.1f}s'
                  + (f' (avg {el / solved:.1f}s/probe).' if solved else '.'))

    # ------------------------------------------------------------------
    # Ladder orchestration
    # ------------------------------------------------------------------

    def _setup_sel(self):
        """Match the rows of model.c (possibly pruned by prune_after_crown
        during build) back to clause indices of the full c. Needed because
        the alphas stored on the net correspond to model.c's clause set."""
        if self.sel is not None:
            return True
        mc = getattr(self.model, 'c', None)
        if mc is None or mc.dim() != 3 or mc.shape[0] != 1:
            return False
        mc = mc.to(self.c_dev)
        eq = (mc[0].unsqueeze(1) == self.c_dev[0].unsqueeze(0)).all(dim=-1)
        if not bool(eq.any(dim=1).all()):
            return False
        self.sel = eq.float().argmax(dim=1)
        self.c_sel = mc
        return True

    def _full_box_score(self, rung):
        """Full-box (no pin) reference score at the given rung."""
        sel = self.sel if rung == 'alpha' else None
        new_x, C = self._probe_x_c(1, sel)
        interm = {k: [self.orig_lb[k], self.orig_ub[k]]
                  for k in self.interm_names}
        with torch.no_grad():
            lb = self.net.compute_bounds(
                x=(new_x,), C=C, method='backward',
                reuse_alpha=(rung == 'alpha'),
                interm_bounds=interm, bound_upper=False)[0]
        return float(self.checker.score(lb, sel)[0])

    def _escalation_set(self, ref_score):
        """Probes that are near-misses at the current rung: remaining
        deficit within escalate_margin_frac of the full-box deficit, or a
        pair hull gain above escalate_hull_frac. (All candidate scores are
        <= 0 here, otherwise the probe is verified.)"""
        frac = self.cfg['escalate_margin_frac']
        hull_frac = self.cfg['escalate_hull_frac']
        thr = frac * ref_score if ref_score < 0 else float('-inf')
        keys = []
        for key, st in self.probe_state.items():
            if st['verified']:
                continue
            near = ref_score < 0 and st['score'] >= thr
            hull = self.pair_gain.get(key[:2], 0.) >= hull_frac
            if near or hull:
                keys.append(key)
        return keys

    def run_ladder(self, candidates):
        """candidates: list of (layer, nidx, score). Runs the base rung and
        escalates near-misses up to cfg['oracle']."""
        max_rung = _RUNGS.index(self.cfg['oracle'])

        if self.wide_root:
            # Same strong crown oracle, one cross-layer batch instead of one
            # call per pinned layer. Escalation below stays on the per-layer
            # path: the escalation set is small, so the batching win there is
            # negligible and the alpha rung's retained-alpha bookkeeping is
            # tied to run_pair_rung.
            self.run_wide_rung('crown', candidates)
        else:
            pairs_by_layer = {}
            for (name, idx, _) in candidates:
                pairs_by_layer.setdefault(name, []).append(idx)
            self.run_pair_rung('crown', pairs_by_layer)
        if self.both_verified is not None or max_rung < 1:
            return
        ref_crown = self._full_box_score('crown')
        esc = self._escalation_set(ref_crown)

        # ---- rung alpha ----
        ref_alpha = None
        if esc:
            alpha_ok = self._setup_sel() and bool(self._saved_final_alphas())
            if not alpha_ok:
                print('Phase probing: alphas/clause mapping unavailable, '
                      'skipping alpha rung.')
            else:
                try:
                    ref_alpha = self._full_box_score('alpha')
                    # Escalate whole PAIRS so the hull fold stays
                    # well-defined within a chunk.
                    alpha_pairs, seen = {}, set()
                    for (layer, nidx, _) in esc:
                        if (layer, nidx) not in seen:
                            seen.add((layer, nidx))
                            alpha_pairs.setdefault(layer, []).append(nidx)
                    self.stats['escalated']['alpha'] = 2 * len(seen)
                    print(f'Phase probing: escalating {2 * len(seen)} probes '
                          f'({len(seen)} pairs) to rung alpha (full-box '
                          f'score crown={ref_crown:.4f}, '
                          f'alpha={ref_alpha:.4f}).')
                    if self.wide_root:
                        # At full coverage ~97% of probes clear the
                        # escalation threshold, so leaving this on the
                        # per-layer path would run the superlinear rung over
                        # nearly everything -- exactly what the wide crown
                        # rung exists to avoid. Same candidate set, wide
                        # batching.
                        self.run_wide_rung('alpha', [
                            (ln, nidx, 0.) for ln, idxs in alpha_pairs.items()
                            for nidx in idxs])
                    else:
                        self.run_pair_rung('alpha', alpha_pairs)
                except (torch.cuda.OutOfMemoryError, RuntimeError,
                        KeyError) as e:
                    # KeyError: an intermediate-start-node alpha needed for
                    # the downstream recomputation is missing (not retained
                    # through build(), e.g. a model/config corner case).
                    if _is_oom(e):
                        raise
                    print(f'Phase probing: alpha rung failed ({e!r}), '
                          'falling back to crown results.')
                    ref_alpha = None
        if self.both_verified is not None or max_rung < 2:
            return

        # ---- rung beta ----
        if self._setup_sel():
            ref = ref_alpha if ref_alpha is not None else ref_crown
            esc = self._escalation_set(ref)
            if esc:
                self.stats['escalated']['beta'] = len(esc)
                print(f'Phase probing: escalating {len(esc)} probes to rung '
                      'beta.')
                try:
                    self.run_beta_rung(esc)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if _is_oom(e):
                        raise
                    print(f'Phase probing: beta rung failed ({e!r}), falling '
                          'back to previous results.')
        else:
            print('Phase probing: clause mapping unavailable, skipping beta '
                  'rung.')
        if self.both_verified is not None or max_rung < 3:
            return

        # ---- rung gurobi ----
        ref = ref_alpha if ref_alpha is not None else ref_crown
        esc = self._escalation_set(ref)
        if esc:
            self.stats['escalated']['gurobi'] = len(esc)
            print(f'Phase probing: escalating {len(esc)} probes to rung '
                  'gurobi.')
            self.run_gurobi_rung(esc)

    # ------------------------------------------------------------------
    # Products: forced phases, cuts, implications
    # ------------------------------------------------------------------

    def forced_phases(self):
        """(layer, nidx) -> phase to FORCE ('active'/'inactive'); the
        OPPOSITE phase region was verified by a probe. SPEC-CONDITIONAL."""
        forced = {}
        for (layer, nidx, sign), st in self.probe_state.items():
            if st['verified']:
                forced[(layer, nidx)] = 'inactive' if sign > 0 else 'active'
        return forced

    def emit_implied_cuts(self):
        """Build GCP-CROWN cut dicts from the collected candidates.

        For a pinned unstable neuron i with activation indicator z_i
        (z=1 active) and a downstream pre-activation v with per-phase bounds
        (A = under the active pin, I = under the inactive pin):
            ub cut:  v <= ub_I + (ub_A - ub_I) * z_i
                     -> 1*v - (ub_A - ub_I)*z_i <= ub_I        (c = -1)
            lb cut:  v >= lb_I + (lb_A - lb_I) * z_i
                     -> 1*v - (lb_A - lb_I)*z_i >= lb_I        (c = +1)
        Box-sound by construction: at z_i=1 the bound is the active-phase
        bound, at z_i=0 the inactive-phase bound, and every input realizes
        one of the phases. Decisions are [relu_layer_index, flat_idx]
        following the cuts/cutter.py wire format.
        """
        max_cuts = self.cfg['implied_cuts_max']
        ranked = sorted(self.cut_cands.items(), key=lambda kv: -kv[1][0])
        cuts = []
        for (pin_layer, pin_idx, v_layer, v_idx, kind), \
                (gapf, b_inact, b_act) in ranked[:max_cuts]:
            cuts.append({
                'x_decision': [], 'x_coeffs': [],
                'relu_decision': [], 'relu_coeffs': [],
                'arelu_decision': [
                    [self.relu_idx_of_preact[pin_layer], pin_idx]],
                'arelu_coeffs': [-(b_act - b_inact)],
                'pre_decision': [
                    [self.relu_idx_of_preact[v_layer], v_idx]],
                'pre_coeffs': [1.0],
                'bias': b_inact,
                'c': -1 if kind == 'ub' else 1,
            })
        return cuts

    def implication_edges_int(self):
        """Implication graph keyed by relu-layer INDEX (the arelu_decision
        convention), for BICCOS cut vivification."""
        out = {}
        for (layer, nidx, sign), targets in self.implications.items():
            li = self.relu_idx_of_preact.get(layer)
            if li is None:
                continue
            tset = out.setdefault((li, nidx, sign), set())
            for (tl, tn, ts) in targets:
                ti = self.relu_idx_of_preact.get(tl)
                if ti is not None:
                    tset.add((ti, tn, ts))
        return out


def vivify_biccos_cuts(cuts, implications_int, forced_int):
    """First-order vivification of BICCOS blocking clauses using the probe
    phase-implication graph.

    A BICCOS blocking clause (indicator-only cut, c = -1) has the form
        sum_i s_i z_i <= (#positive s_i) - 1,
    equivalent to the disjunction OR_i (literal_i) where literal_i is
    "z_i differs from the blocked assignment": literal_i = (z_i = 0) for
    s_i = +1 and (z_i = 1) for s_i = -1.

    A literal L_a is removable when
      * some other literal L_b in the clause satisfies (NOT L_b) => (NOT L_a)
        via a probe implication edge, or
      * (NOT L_a) is a probe-forced fact.
    Soundness: the clause is entailed (on the counterexample-relevant
    region); take any point falsifying all remaining literals -- then L_a
    must hold (clause), but NOT L_b holds, so NOT L_a holds (edge/fact), a
    contradiction. Hence the shortened clause is entailed. Probe edges are
    box-sound; forced facts are spec-conditional -- the same conditionality
    BICCOS cuts themselves have (they block verified subtrees of THIS
    property).

    Polarity bookkeeping: the literal for coefficient s_a is "phase(-s_a)",
    so NOT(L_a) is "phase(s_a)". The removal conditions therefore read:
    edge (b, s_b) => (a, s_a) present, or forced phase of a equals s_a.

    `cuts` is modified in place. Returns (num_literals_removed,
    num_cuts_shortened).
    """
    removed, shortened = 0, 0
    if not implications_int and not forced_int:
        return removed, shortened
    for cut in cuts:
        if cut.get('x_decision') or cut.get('relu_decision') \
                or cut.get('pre_decision') or cut.get('c') != -1:
            continue
        dec = [tuple(d) for d in cut['arelu_decision']]
        signs = [1 if s > 0 else -1 for s in cut['arelu_coeffs']]
        npos = sum(1 for s in signs if s > 0)
        if abs(float(cut['bias']) - (npos - 1)) > 1e-6:
            continue  # not a pure blocking clause
        lits = list(zip(dec, signs))
        changed, any_removed = True, False
        while changed and len(lits) > 1:
            changed = False
            for ai, ((al, an), a_s) in enumerate(lits):
                # NOT(L_a) is "neuron a in phase a_s".
                removable = forced_int.get((al, an)) == a_s
                if not removable:
                    for bi, ((bl, bn), b_s) in enumerate(lits):
                        if bi == ai:
                            continue
                        if (al, an, a_s) in implications_int.get(
                                (bl, bn, b_s), ()):
                            removable = True
                            break
                if removable:
                    del lits[ai]
                    removed += 1
                    any_removed = changed = True
                    break
        if any_removed:
            shortened += 1
            cut['arelu_decision'] = [list(d) for d, _ in lits]
            cut['arelu_coeffs'] = [float(s) for _, s in lits]
            cut['bias'] = float(sum(1 for _, s in lits if s > 0) - 1)
    return removed, shortened


class ClauseVivifier:
    """Joint-pin descent vivification of BICCOS blocking clauses.

    A BICCOS blocking clause is a disjunction OR_i L_i over ReLU phase
    literals (see vivify_biccos_cuts for the encoding). For a PREFIX of j
    literals, pin the negations of all j literals simultaneously -- multi-
    neuron clamps in one domain with every intermediate bound fixed, i.e. a
    j-split BaB domain view -- and recompute the output lower bound. If the
    pinned region is verified (contains no counterexample of THIS
    specification), then L_1 OR ... OR L_j is entailed on the
    counterexample-relevant region and REPLACES the clause: a strictly
    stronger cut with the same conditionality BICCOS cuts already have.

    Negative result ported from the Marabou side of this project: testing
    literals via intersections of stored SINGLE-pin probe bounds is dead
    (0 removals in ~49k checks there); the pins must be propagated JOINTLY,
    which is what the batched compute_bounds below does.

    CONDITIONING (important): BaB runs once per unverified OR group with a
    single-clause specification (d['cs']); BICCOS clauses learned in that
    run are entailed w.r.t. THAT clause only. The probes therefore use the
    picked domains' cs/thresholds -- testing against the full multi-group
    spec would demand entailment the clauses never had (measured: 0
    shortenings at both crown and beta grade before this fix).

    All (clause, prefix) candidates of one cut-inference round are batched
    into as few GPU calls as possible. Literals are ordered
    most-tightening-first (probe hull gain, then root instability score) so
    short prefixes carry the strongest pins. Probes run at cfg
    ['vivify_grade']: 'crown' is a plain backward pass; 'alpha' reuses the
    net's CURRENT final-start-node alphas (sliced to batch 1, broadcast
    over the probe batch -- they are spec-consistent with this BaB run);
    'beta' additionally enforces the pins with SparseBeta Lagrangians and
    optimizes over the probe batch WITH the GCP-CROWN cut pool, the exact
    grade BICCOS re-verified the clauses at. Around every chunk the net's
    beta/cut state is swapped out and restored -- stale batch-mismatched
    state would otherwise leak into the probe (see backward_bound.py's
    enable_beta_crown gate and BoundRelu.cut_used).

    Constructed at probe time (_probe_and_refine) because it needs the
    refined root bounds and the probe hull-gain ordering data; stashed on
    the LiRPANet and consumed by BICCOS.update_cut. Everything heavy is
    kept on the CPU between calls.
    """

    def __init__(self, prober, model):
        cfg = arguments.Config['solver']['phase_probing']
        self.model = model
        self.net = model.net
        self.device = model.device
        self.final_name = model.final_name
        self.x = prober.x
        # Needed by _harvest_reprobe_edges (unstable masks, layer order,
        # preact->relu index map) -- the same structures the root pass's
        # _collect_edges_and_cuts reads.
        self.prober = prober
        self.reprobe_wide = cfg.get('reprobe_wide', False)
        self.grade = cfg['vivify_grade']
        self.max_lits = cfg['vivify_max_lits']
        self.budget = cfg['vivify_budget']
        # Optimizer iterations per beta-grade probe chunk -- the dominant
        # strength-vs-cost knob of the oracle (final alphas / SparseBetas /
        # cut betas are re-optimized per probe from the parent-slice warm
        # start). 0 = inherit solver:beta-crown:iteration.
        self.iterations = (cfg['vivify_iterations']
                           if cfg['vivify_iterations'] > 0 else
                           arguments.Config['solver']['beta-crown']
                           ['iteration'])
        # Cut usage of the beta-grade oracle: 'auto' = the net's current
        # cut module (the pool as of the previous BICCOS rebuild; round 1
        # runs cut-less), 'off' = no cut terms (ablation), 'pool' = a
        # probe-scoped module rebuilt per vivify call from the CURRENT
        # pool + this round's fresh clauses (closed loop; see
        # _install_probe_cut_pool).
        self.use_cuts_mode = cfg['vivify_use_cuts']
        # BCP pre-pass over the SAT layer's clause DB (see vivify()).
        self.bcp = cfg['vivify_bcp']
        # Mirror oracle (supersedes the BCP conflict test with a
        # conflict-bounded assumption-core solve; implies the extra-pin
        # propagation of the BCP pass).
        self.mirror = cfg['mirror']
        if self.mirror:
            self.bcp = True
        # GPU dry-round gate: after this many consecutive rounds with no
        # GPU-side shortening, stop spending GPU on this run (the CPU
        # passes keep running). 0 = off.
        self.dry_rounds = cfg['vivify_dry_rounds']
        # Conditioned re-probing at depth (see reprobe()).
        self.reprobe_enabled = cfg['reprobe']
        self.reprobe_budget = cfg['reprobe_budget']
        self.reprobe_max_neurons = cfg['reprobe_max_neurons']
        self._gpu_dry = 0
        self._gpu_run_key = None
        self.batch_size = max(2, cfg['batch_size'])
        self.interm_names = list(prober.interm_names)
        self.preact_of_relu_idx = {
            v: k for k, v in prober.relu_idx_of_preact.items()}
        # Re-probe premise diagnostic (measurement only, see
        # _reprobe_premise_diagnostic).
        self.relu_idx_of_preact = dict(prober.relu_idx_of_preact)
        self.node_order = dict(prober.node_order)
        self.unstable_mask = {k: v.clone()
                              for k, v in prober.unstable_mask.items()}
        # Refined root bounds (the bounds BaB starts from; forced-phase
        # clamps included -- spec-conditional like the clauses themselves).
        self.base_lb = {k: prober.ref_lb[k].detach().cpu()
                        for k in self.interm_names}
        self.base_ub = {k: prober.ref_ub[k].detach().cpu()
                        for k in self.interm_names}
        # Literal ordering data: probe hull gain (primary), instability
        # score of the refined root bounds (secondary).
        self.pair_gain = {}
        for (layer, nidx), g in prober.pair_gain.items():
            ridx = prober.relu_idx_of_preact.get(layer)
            if ridx is not None:
                self.pair_gain[(ridx, nidx)] = g
        self.split_node_names = [n.name for n in self.net.split_nodes]
        # Per-vivify-call spec (this BaB run's clause) -- see vivify().
        self.cur_c, self.cur_rhs = None, None
        self.stats = {
            'pool': 0, 'eligible': 0, 'shortened': 0, 'removed_lits': 0,
            'removed_root': 0, 'probes': 0, 'full_verified': 0,
            'full_tested': 0, 'gpu_time': 0.0, 'skipped_multi_c': 0,
            'len_hist': {},  # eligible clause length -> count
            # Probe-scoped cut pool ('pool' mode): last pool size + rounds.
            'cut_pool_size': 0, 'cut_pool_rounds': 0,
            # BCP pre-pass: clauses shortened by propagation conflict alone
            # (zero GPU), literals those conflicts removed, extra implied
            # pins added to GPU probes, GPU probes skipped, CPU seconds.
            'bcp_shortened': 0, 'bcp_removed_lits': 0, 'bcp_pins': 0,
            'bcp_skipped_probes': 0, 'bcp_time': 0.0,
            # Mirror oracle: clauses shortened by assumption-core solves
            # (zero GPU), literals removed that way, GPU probes skipped.
            'mirror_shortened': 0, 'mirror_removed_lits': 0,
            'mirror_skipped_probes': 0,
        }

    # -- clause parsing -------------------------------------------------

    def _parse(self, cut):
        """Blocking clause -> ordered literal list [(ridx, nidx, sign)]
        with sign = arelu coefficient = the phase pinned by the literal's
        NEGATION (+1 active / -1 inactive), or None if not eligible.
        Root-stable literals are screened out CPU-side first:
        a literal false at the root is removable unconditionally; a literal
        true at the root makes the whole clause tautological (skipped)."""
        if cut.get('x_decision') or cut.get('relu_decision') \
                or cut.get('pre_decision') or cut.get('c') != -1:
            return None, 0
        signs = [1 if s > 0 else -1 for s in cut['arelu_coeffs']]
        npos = sum(1 for s in signs if s > 0)
        if abs(float(cut['bias']) - (npos - 1)) > 1e-6:
            return None, 0
        lits, removed_root = [], 0
        seen = set()
        for (ridx, nidx), s in zip(cut['arelu_decision'], signs):
            name = self.preact_of_relu_idx.get(ridx)
            if name is None or (ridx, nidx) in seen:
                return None, 0  # unknown layer / duplicated neuron: skip
            seen.add((ridx, nidx))
            lb = float(self.base_lb[name].view(-1)[nidx])
            ub = float(self.base_ub[name].view(-1)[nidx])
            if (s > 0 and ub <= 0) or (s < 0 and lb >= 0):
                # Literal TRUE at the root: clause is a tautology.
                return None, 0
            if (s > 0 and lb >= 0) or (s < 0 and ub <= 0):
                removed_root += 1  # literal FALSE at the root: drop it
                continue
            width = max(ub - lb, 1e-12)
            instab = abs(lb * ub) / width
            lits.append((self.pair_gain.get((ridx, nidx), 0.), instab,
                         ridx, nidx, s))
        lits.sort(key=lambda t: (-t[0], -t[1]))
        return [(r, n, s) for _, _, r, n, s in lits], removed_root

    @staticmethod
    def _rewrite(cut, lits):
        """Replace the clause by the given literal list (same wire format,
        decisions sorted the way BICCOS emits them)."""
        merged = sorted(([r, n], float(s)) for r, n, s in lits)
        cut['arelu_decision'] = [d for d, _ in merged]
        cut['arelu_coeffs'] = [c for _, c in merged]
        cut['bias'] = float(sum(1 for _, c in merged if c > 0) - 1)

    # -- GPU probe execution --------------------------------------------

    def _pinned_interm_bounds(self, pins_per_probe):
        """Batched interm_bounds with every layer fixed at the refined root
        value and the pinned neurons clamped per batch element."""
        B = len(pins_per_probe)
        pinned_layers = set()
        for pins in pins_per_probe:
            for (ridx, _, _) in pins:
                pinned_layers.add(self.preact_of_relu_idx[ridx])
        probe_ib = {}
        for k in self.interm_names:
            lb = self.base_lb[k].to(self.device)
            ub = self.base_ub[k].to(self.device)
            if k in pinned_layers:
                rep = [B] + [1] * (lb.dim() - 1)
                probe_ib[k] = [lb.repeat(*rep), ub.repeat(*rep)]
            else:
                probe_ib[k] = [lb.expand(B, *lb.shape[1:]),
                               ub.expand(B, *ub.shape[1:])]
        for j, pins in enumerate(pins_per_probe):
            for (ridx, nidx, sign) in pins:
                name = self.preact_of_relu_idx[ridx]
                if sign > 0:
                    probe_ib[name][0].view(B, -1)[j, nidx] = 0.  # pin ACTIVE
                else:
                    probe_ib[name][1].view(B, -1)[j, nidx] = 0.  # pin INACTIVE
        return probe_ib

    def _reprobe_premise_diagnostic(self, sat_layer):
        """Measurement-only (PHASE_PROBING_REPROBE_DIAG=1): price the
        re-probe premise BEFORE building re-probing.

        Marabou's conditioned re-probing rests on "density grows as boxes
        shrink": once the level-0 fixed set grows, re-deriving facts under
        the tightened box finds things the root pass could not. The
        abcrown equivalent of that fixed set is the SAT layer's run-scoped
        forced phases (failed-literal units + mirror-derived units). This
        clamps them into the refined root box, recomputes intermediate
        bounds ONCE (batch 1, crown, all-fixed-upstream -- milliseconds),
        and reports what actually moved:

          fixed   -- run-scoped forced phases available to clamp
          moved   -- unstable neurons whose pre-activation bounds changed
          newstab -- unstable neurons the clamps alone made STABLE (these
                     are free: no probe needed, and they leave the probe
                     target set)
          tighten -- mean/max relative box tightening over moved neurons

        A pass where nothing moves means re-probing would re-derive the
        root pass's own results at full cost -- exactly the zero-fact
        passes that cost Marabou instance 2_1 its verdict. This tells us
        the firing rate and the yield ceiling before any GPU is spent.
        """
        st = self.stats
        decided = getattr(sat_layer, '_flp_decided', None)
        if not decided:
            return
        n_fixed = len(decided)
        if n_fixed == st.get('reprobe_last_fixed', 0):
            return  # trigger: growth of the level-0 fixed set
        st['reprobe_last_fixed'] = n_fixed
        st['reprobe_passes'] = st.get('reprobe_passes', 0) + 1
        name_of_var = {v: k for k, v in sat_layer._var_of.items()}
        pins = []
        for lit in decided:
            entry = name_of_var.get(abs(lit))
            if entry is None:
                continue
            name, nidx = entry
            ridx = self.relu_idx_of_preact.get(name)
            if ridx is not None:
                pins.append((ridx, nidx, 1 if lit > 0 else -1))
        if not pins:
            return
        t0 = time.time()
        undo = self._swap_net_state()
        try:
            probe_ib = self._pinned_interm_bounds([pins])
            # Keep the layers that CARRY the clamps (the pinned ones) plus
            # everything upstream of the earliest pin, and FREE the rest so
            # their bounds are recomputed under the clamps. Passing a layer
            # in interm_bounds fixes it, so dropping the pinned layers here
            # would silently delete the clamps and measure the root box
            # against itself.
            pinned_names = {self.preact_of_relu_idx[r] for (r, _, _) in pins}
            first = min(self.node_order[n] for n in pinned_names)
            probe_ib = {k: v for k, v in probe_ib.items()
                        if k in pinned_names or self.node_order[k] < first}
            n_free = len(self.interm_names) - len(probe_ib)
            # Two recomputations, IDENTICAL except for the clamps: the
            # baseline must use the same method and the same freed layers,
            # or the comparison measures the method rather than the pins.
            # (Comparing against base_lb/base_ub -- alpha-CROWN optimized
            # and hull-refined -- floors every delta to zero: plain CROWN
            # is looser than that baseline everywhere.)
            def _recompute(ib):
                with torch.no_grad():
                    self.net.compute_bounds(
                        x=(expand_batch(self.x, 1, device=self.device),),
                        C=self.cur_c, method='backward',
                        interm_bounds=ib, bound_upper=False)
                return {k: (self.net[k].lower.detach().clone(),
                            self.net[k].upper.detach().clone())
                        for k in self.interm_names
                        if k not in ib and k in self.unstable_mask
                        and self.net[k].lower is not None
                        and self.net[k].upper is not None}

            unclamped = {k: [self.base_lb[k].to(self.device),
                             self.base_ub[k].to(self.device)]
                         for k in probe_ib}
            ref = _recompute(unclamped)
            cur = _recompute(probe_ib)
            moved = newstab = 0
            rels = []
            for k, (nl_t, nu_t) in cur.items():
                if k not in ref:
                    continue
                um = self.unstable_mask[k]
                if not bool(um.any()):
                    continue
                nl = nl_t.reshape(-1)[um]
                nu = nu_t.reshape(-1)[um]
                ol = ref[k][0].reshape(-1)[um]
                ou = ref[k][1].reshape(-1)[um]
                width = (ou - ol).clamp(min=1e-12)
                delta = ((nl - ol).clamp(min=0) + (ou - nu).clamp(min=0))
                hit = delta > 1e-6
                moved += int(hit.sum())
                newstab += int(((nl >= 0) | (nu <= 0)).sum())
                if bool(hit.any()):
                    rels.append((delta / width)[hit])
            rel = torch.cat(rels) if rels else torch.zeros(1)
            st['reprobe_moved'] = st.get('reprobe_moved', 0) + moved
            st['reprobe_newstab'] = st.get('reprobe_newstab', 0) + newstab
            st['reprobe_diag_time'] = (st.get('reprobe_diag_time', 0.)
                                       + time.time() - t0)
            print(f'Phase probing re-probe premise: pass '
                  f"{st['reprobe_passes']} -- fixed={n_fixed} "
                  f'({len(pins)} clampable, {n_free} layers recomputed), '
                  f'moved={moved}, '
                  f'newly_stable={newstab}, tighten avg='
                  f'{float(rel.mean()):.6f} max={float(rel.max()):.6f}, '
                  f'{time.time() - t0:.3f}s (cumulative: '
                  f"passes={st['reprobe_passes']}, moved={st['reprobe_moved']}, "
                  f"newly_stable={st['reprobe_newstab']}, "
                  f"time={st['reprobe_diag_time']:.2f}s).")
        except Exception as e:
            print(f'Phase probing re-probe premise: diagnostic failed ({e!r}).')
        finally:
            undo()

    def _harvest_reprobe_edges(self, chunk_keys, verified, sat_layer, run_key):
        """Harvest binary implications from a re-probe chunk.

        Marabou harvests binaries in BOTH its root probe and its re-probe, and
        on its calibration instances the binaries are the bulk of the payload
        (3_4 root: 3 units vs 141 binaries; re-probe: 10 vs 14). abcrown had
        the whole implication-graph machinery -- the root probe collects edges
        (_collect_edges_and_cuts) and the SAT layer stores them as (-src OR
        tgt) clauses that the mirror propagates over -- but re-probe only ever
        called add_run_unit, so it contributed nothing to the graph. Isolated
        units just clamp one neuron each; edges are what the boolean layer can
        actually propagate over.

        A pin that was NOT refuted (verified == False) leaves the net's
        downstream nodes holding the bounds computed under it, exactly as
        _fold_downstream reads them. Any originally-unstable neuron that is
        stable under the pin gives the edge (pin => that phase). Same detector
        and same encoding as the root pass; RUN-scoped because the pin set also
        carries this run's forced phases.

        Returns the number of new edges installed.
        """
        added = 0
        # Inverse of prober.relu_idx_of_preact (pre-act name -> relu index),
        # built once: pair_gain is keyed by (pre-act name, neuron).
        if not hasattr(self, '_preact_of_ridx'):
            self._preact_of_ridx = {
                ridx: name
                for name, ridx in self.prober.relu_idx_of_preact.items()}
        for k in self.prober.interm_names:
            if k not in self.prober.unstable_mask:
                continue
            um = self.prober.unstable_mask[k]
            if not bool(um.any()):
                continue
            node = self.net[k]
            dl, du = node.lower, node.upper
            if dl is None or du is None or dl.shape[0] != len(chunk_keys):
                continue  # not recomputed for this chunk
            tgt_ridx = self.prober.relu_idx_of_preact.get(k)
            if tgt_ridx is None:
                continue
            fl = dl.detach().reshape(len(chunk_keys), -1)[:, um]
            fu = du.detach().reshape(len(chunk_keys), -1)[:, um]
            uidx = um.nonzero().reshape(-1)
            for cond, tgt_sign in ((fl >= 0, +1), (fu <= 0, -1)):
                for pidx, j in cond.nonzero().tolist():
                    if verified[pidx]:
                        continue  # refuted pin -> it is a unit, not an edge
                    sridx, snidx, ssign = chunk_keys[pidx]
                    if sridx == tgt_ridx and snidx == int(uidx[j]):
                        continue  # a pin implying itself carries no news
                    # Strength = how much the SOURCE pin actually moved the
                    # box at root (pair_gain, the same measured quantity that
                    # ranks re-probe candidates). The cut budget is tens while
                    # a pass harvests thousands, so this decides which edges
                    # reach the cutter at all.
                    src_name = self._preact_of_ridx.get(sridx)
                    score = self.prober.pair_gain.get((src_name, snidx), 0.0) \
                        if src_name is not None else 0.0
                    if sat_layer.add_run_edge(sridx, snidx, ssign, tgt_ridx,
                                              int(uidx[j]), tgt_sign, run_key,
                                              score=score):
                        added += 1
        return added

    def _clamp_forced_phase(self, ridx, nidx, sign):
        """Clamp a proven phase into the STATIC intermediate-bound templates
        so it reaches BaB's branching mask.

        sign > 0 means the neuron is forced ACTIVE (pre-activation >= 0), so
        its lower bound clamps to 0; sign < 0 forces INACTIVE (<= 0), so the
        upper bound clamps to 0. Either way the neuron stops being unstable
        and compute_unstable_mask drops it from the split candidates.

        Only ever TIGHTENS (torch.clamp toward 0 from the feasible side), and
        only for a fact this run already proved entailed, so it cannot admit
        behaviour the query does not have. Returns 1 if a bound moved.

        Requires the static-template path (bab interm_transfer off, the
        setting our configs use). With per-domain transfer the templates are
        not the source of truth, so this no-ops rather than clamp something
        that will be overwritten.
        """
        # domain_interm_factory lives on the LiRPANet WRAPPER (self.model,
        # beta_CROWN_solver.py), not on the BoundedModule (self.net = model.net)
        # -- reading it off self.net silently returned None and the clamp never
        # fired. Fall back to net for any caller that passes the wrapper as net.
        net = self.net
        factory = getattr(self.model, 'domain_interm_factory', None) \
            or getattr(net, 'domain_interm_factory', None)
        if factory is None:
            return 0
        static_lb = getattr(factory, 'static_lb', None)
        static_ub = getattr(factory, 'static_ub', None)
        if not static_lb or not static_ub:
            return 0
        relus = getattr(net, 'relus', [])
        if ridx >= len(relus):
            return 0
        name = relus[ridx].inputs[0].name
        if name not in static_lb or name not in static_ub:
            return 0
        try:
            lo = static_lb[name].view(static_lb[name].shape[0], -1)
            hi = static_ub[name].view(static_ub[name].shape[0], -1)
            if nidx >= lo.shape[1]:
                return 0
            moved = 0
            if sign > 0:
                cur = lo[:, nidx]
                if bool((cur < 0).any()):
                    lo[:, nidx] = torch.clamp(cur, min=0.)
                    moved = 1
            else:
                cur = hi[:, nidx]
                if bool((cur > 0).any()):
                    hi[:, nidx] = torch.clamp(cur, max=0.)
                    moved = 1
            # Never leave an empty box: if the clamp crossed the other bound
            # the region is infeasible, which BaB will detect on its own; keep
            # the tensors consistent rather than inverted.
            if moved and bool((lo[:, nidx] > hi[:, nidx]).any()):
                lo[:, nidx] = torch.minimum(lo[:, nidx], hi[:, nidx])
            return moved
        except Exception:
            return 0

    def reprobe(self, sat_layer, run_key, pool_cuts=None):
        """Conditioned re-probing at depth (ported from Marabou).

        The root probe pass runs pre-BaB, where the GCP-CROWN cut pool is
        EMPTY -- it is plain beta-CROWN by construction. By the time BaB
        has inferred clauses the pool holds tens to hundreds of cuts, and
        on the calibration instance the pool is the entire source of
        oracle power (idx0 vivification: 0/687 shortened with cuts off,
        119/697 with the pool). So re-probing here is not a repeat of the
        root pass under a smaller box -- it is a STRICTLY STRONGER oracle
        than the root pass could ever have run, which is the structural
        difference from the Marabou original (their re-probe reruns the
        same stack).

        The run's forced phases are the box tightening, and the beta-grade
        oracle already expresses pins as clamps + SparseBeta splits, so
        the re-probe of candidate neuron j is simply the pin set
        {forced phases} + {j pinned}. If that region verifies, the pin is
        refuted and its NEGATION is a forced phase -- run-scoped, exactly
        like the clauses it was derived alongside.

        Facts go into the run DB (flushed on run change), never the
        persistent one: they are conditioned on this OR group's spec and
        on run-scoped forced phases.
        """
        st = self.stats
        decided = getattr(sat_layer, '_flp_decided', None)
        if not decided or run_key is None:
            return
        if len(decided) == st.get('reprobe_last_fixed', 0):
            return  # trigger: growth of the level-0 fixed set
        st['reprobe_last_fixed'] = len(decided)
        budget = self.reprobe_budget
        if st.get('reprobe_time', 0.) >= budget:
            return
        max_n = self.reprobe_max_neurons
        name_of_var = {v: k for k, v in sat_layer._var_of.items()}

        base_pins, decided_keys = [], set()
        for lit in decided:
            entry = name_of_var.get(abs(lit))
            if entry is None:
                continue
            name, nidx = entry
            ridx = self.relu_idx_of_preact.get(name)
            if ridx is not None:
                base_pins.append((ridx, nidx, 1 if lit > 0 else -1))
                decided_keys.add((ridx, nidx))
        if not base_pins:
            return

        # Candidates: still-unstable neurons that are not already decided,
        # ranked by the root pass's hull gain (the neurons whose pins moved
        # the box most are the ones most likely to refute under a tighter
        # box + cuts). Cheap proxy; selection is budget allocation, never
        # soundness -- a skipped probe loses a fact, it cannot produce a
        # wrong one.
        # WIDE mode (--phase_probing_reprobe_wide): draw candidates from EVERY
        # still-unstable neuron, not just the ones the root probe happened to
        # examine. pair_gain only ever holds the root pass's top-N (64 by
        # default) out of ~1,445 unstable on cifar100 -- a 4.4% slice -- so it,
        # not the neuron cap, is what really bounds coverage. Marabou's
        # skeleton is exhaustive by construction (100% of unfixed ReLUs) and
        # that is the one structural difference with this port we never closed;
        # every delivery channel we measured came back at exactly 1.000x, which
        # is equally consistent with "facts are worthless" and with "4% of the
        # neurons cannot move a search over the other 96%".
        #
        # Affordable because this path fixes ALL intermediate bounds
        # (_pinned_interm_bounds) and pins per BATCH ROW, so probes from
        # different layers share one backward pass. The root pass cannot do
        # that: it OMITS downstream layers to recompute them under the clamp,
        # and which layers are omitted depends on the pin's depth, so it must
        # group by layer and pays a full pass per group (65 -> 145 ms/neuron
        # measured from n=64 to n=256). The trade is that we lose downstream
        # recomputation: no hull refinement, no downstream-derived implication
        # edges, and a weaker refutation test. Hull tightening measures 0.002
        # avg and the edges measured inert, so the trade is cheap.
        cands = []
        if self.reprobe_wide:
            for name in self.prober.relu_preacts:
                um = self.prober.unstable_mask.get(name)
                ridx = self.prober.relu_idx_of_preact.get(name)
                if um is None or ridx is None or not bool(um.any()):
                    continue
                for nidx in um.nonzero().reshape(-1).tolist():
                    if (ridx, nidx) in decided_keys:
                        continue
                    # Rank by root hull gain when known, else 0: unexamined
                    # neurons sort last but are still probed when budget allows.
                    cands.append((self.pair_gain.get((ridx, nidx), 0.0),
                                  ridx, int(nidx)))
        else:
            for (ridx, nidx), gain in self.pair_gain.items():
                if (ridx, nidx) not in decided_keys:
                    cands.append((gain, ridx, nidx))
        if not cands:
            return
        cands.sort(reverse=True)
        cands = cands[:max_n]

        t0 = time.time()
        st['reprobe_passes'] = st.get('reprobe_passes', 0) + 1
        pins_per_probe, keys = [], []
        for (_, ridx, nidx) in cands:
            for sign in (1, -1):
                pins_per_probe.append(base_pins + [(ridx, nidx, sign)])
                keys.append((ridx, nidx, sign))

        undo = self._swap_net_state()
        pool_undo = None
        new_units = 0
        clamped = 0
        edges = 0
        try:
            if (self.use_cuts_mode == 'pool' and self.grade == 'beta'
                    and arguments.Config['bab']['cut']['bab_cut']):
                pool_undo = self._install_probe_cut_pool([], pool_cuts)
            for i in range(0, len(pins_per_probe), self.batch_size):
                if time.time() - t0 > budget - st.get('reprobe_time', 0.):
                    st['reprobe_truncated'] = st.get('reprobe_truncated', 0) + 1
                    break
                # Free the allocator between chunks, as the root probe's own
                # loops do (run_pair_rung / the beta rung). Without this the
                # pass leaves the cache inflated and the BaB that follows OOMs
                # in ordinary bounding -- measured on cifar100, where only the
                # re-probe arm died (1.69 GiB alloc failing with 1.61 GiB free
                # of 79 GiB) while baseline/canonical/mirror/n32/n256 all
                # completed 200/200 on the same config.
                torch.cuda.empty_cache()
                chunk = pins_per_probe[i:i + self.batch_size]
                verified = self._run_chunk(chunk)
                st['reprobe_probes'] = st.get('reprobe_probes', 0) + len(chunk)
                # Harvest binaries from the SAME bounds the refutation test
                # just computed -- free, and the payload Marabou gets most of
                # its value from (see _harvest_reprobe_edges).
                vlist = verified.tolist()
                edges += self._harvest_reprobe_edges(
                    keys[i:i + len(chunk)], vlist, sat_layer, run_key)
                for j, ok in enumerate(verified.tolist()):
                    if not ok:
                        continue
                    # The pinned region verified => no counterexample of
                    # this run lies in it => the pin is refuted and its
                    # negation is entailed for this run.
                    ridx, nidx, sign = keys[i + j]
                    if sat_layer.add_run_unit(ridx, nidx, -sign, run_key):
                        new_units += 1
                        # Deliver the fact where BaB can actually use it.
                        # add_run_unit only reaches the SAT DB + fact-cut
                        # channel, which prune at pick time; the branching
                        # mask is recomputed every iteration from the DOMAIN
                        # bounds (bab.py: compute_unstable_mask over
                        # d["lower_bounds"]/["upper_bounds"]), so a neuron
                        # whose phase we have proven keeps being offered as a
                        # split candidate. Measured: 15,120 units bought +4.5
                        # pruned domains/instance and a median domain ratio of
                        # exactly 1.000 -- the facts never shrank the tree.
                        # Clamping the static templates fixes that: with
                        # interm_transfer off they are the source every
                        # domain's bounds are cloned from, so the neuron goes
                        # stable and drops out of the mask for good. Same
                        # mechanism that makes the ROOT probe's forced phases
                        # pay (it writes ref_lb/ref_ub -> ret -> BaB's
                        # starting bounds).
                        clamped += self._clamp_forced_phase(ridx, nidx, -sign)
        except Exception as e:
            print(f'Phase probing re-probe: pass failed ({e!r}).')
        finally:
            if pool_undo is not None:
                pool_undo()
            undo()
            # Never hand BaB an inflated allocator after a pass.
            torch.cuda.empty_cache()
        st['reprobe_units'] = st.get('reprobe_units', 0) + new_units
        st['reprobe_clamped'] = st.get('reprobe_clamped', 0) + clamped
        st['reprobe_edges'] = st.get('reprobe_edges', 0) + edges
        st['reprobe_time'] = st.get('reprobe_time', 0.) + time.time() - t0
        print(f"Phase probing re-probe: pass {st['reprobe_passes']} -- "
              f'{len(base_pins)} forced phases pinned, {len(cands)} neurons '
              f'({len(pins_per_probe)} probes) at {self.grade} grade'
              f"{' + pool cuts' if pool_undo is not None else ''}, "
              f'{new_units} new forced phases ({clamped} clamped into the '
              f'branching mask), {edges} new implication edges, '
              f'{time.time() - t0:.2f}s '
              f"(cumulative: passes={st['reprobe_passes']}, "
              f"probes={st.get('reprobe_probes', 0)}, "
              f"units={st['reprobe_units']}, "
              f"time={st['reprobe_time']:.2f}s / {budget:.0f}s budget).")

    def _verified(self, lb_out):
        """lb_out: [B, spec] output lower bounds on the pinned regions.
        Verified against THIS BaB run's clause set: any clause margin > rhs
        (stop_criterion_batch_any, BaB's own criterion for these runs)."""
        return (lb_out > self.cur_rhs.to(lb_out)).any(dim=1)

    def _run_chunk(self, pins_per_probe):
        """One batched probe over pins_per_probe: list (len B) of pin lists
        [(ridx, nidx, sign)]. Returns verified[B]."""
        if self.grade == 'beta':
            return self._run_chunk_beta(pins_per_probe)
        B = len(pins_per_probe)
        probe_ib = self._pinned_interm_bounds(pins_per_probe)
        new_x = expand_batch(self.x, B, device=self.device)
        C = self.cur_c.expand(B, -1, -1)
        lb_out = self.net.compute_bounds(
            x=(new_x,), C=C, method='backward',
            reuse_alpha=(self.grade == 'alpha'),
            interm_bounds=probe_ib, bound_upper=False)[0]
        return self._verified(lb_out)

    def _run_chunk_beta(self, pins_per_probe):
        """Beta-CROWN probe chunk: the pin set is expressed BOTH as clamped
        intermediate bounds and as a multi-entry split history whose
        SparseBeta enforces the split constraints via the Lagrangian --
        exactly a multi-split BaB domain, the same grade BICCOS verified
        the clause's source domain at (generalizes _PhaseProber's beta rung
        from one pin to several)."""
        B = len(pins_per_probe)
        probe_ib = self._pinned_interm_bounds(pins_per_probe)
        empty = {name: ([], [], [], [], []) for name in self.split_node_names}
        history = []
        for pins in pins_per_probe:
            h = dict(empty)
            by_layer = {}
            for (ridx, nidx, sign) in pins:
                by_layer.setdefault(
                    self.preact_of_relu_idx[ridx], []).append((nidx, sign))
            for name, entries in by_layer.items():
                k = len(entries)
                h[name] = ([n for n, _ in entries], [s for _, s in entries],
                           [0.] * k, [0.] * k, [1] * k)
            history.append(h)
        d = {'history': history, 'betas': [None] * B}
        beta_data, _ = BetaFullData.from_domain_dict(
            d, bias=True, device=self.device)
        beta_data.attach_to_net(self.model)

        # Repeat the batch-1 slice of the net's current final alphas
        # (stashed by _swap_net_state) to the probe batch -- from the SLICE,
        # not from m.alpha, which the previous chunk overwrote with its own
        # batch size. All intermediate bounds are fixed, so only these are
        # optimized.
        # requires_grad stays off: the probes do not optimize the alphas
        # (see enable_alpha_crown below), they only read them as the
        # relaxation the parent domain was bounded under.
        for m in self.net.get_enabled_opt_act():
            a = self._alpha_slice.get(m.name)
            if a is not None:
                rep = [1, 1, B] + [1] * (a.dim() - 3)
                m.alpha[self.final_name] = a.repeat(*rep)

        new_x = expand_batch(self.x, B, device=self.device)
        C = self.cur_c.expand(B, -1, -1)

        def never_stop(x):
            # Run all iterations; per-element early stopping would need the
            # OR-group logic which the optimizer cannot use.
            return torch.zeros(x.shape[0], 1, dtype=torch.bool,
                               device=x.device)

        # set_crown_bound_opts FIRST: it stamps lr/iteration from the
        # config, so the probe-specific opts (in particular the
        # vivify_iterations override) must be applied AFTER it.
        self.model.set_crown_bound_opts('beta')
        # The probes optimize the betas and the general (cut) betas ONLY:
        # the alphas stay fixed at the parent slice, serving as the
        # relaxation coefficients the probe is conditioned on. Measured
        # 2026-07-19 on cifar100 idx0 (deterministic) and idx7: alpha
        # re-optimization moves the full-pin adequacy diagnostic (idx0
        # 489/569 -> 421/569, idx7 25.8% -> 20.7% when frozen) and changes
        # the DELIVERABLE by nothing at all -- idx0 169 shortened / 171
        # literals and idx7's GPU-attributable ~76 shortenings are
        # identical trained or frozen. auto_LiRPA supports this directly
        # via enable_alpha_crown (guards at optimized_bounds.py:362,913);
        # with alphas out of the parameter group their gradients are no
        # longer computed, which is where the saving actually comes from
        # (an lr_alpha=0 freeze still pays for the backward pass).
        opt_args = {
            'enable_alpha_crown': False,
            'enable_beta_crown': True,
            'fix_interm_bounds': True,
            'stop_criterion_func': never_stop,
            'multi_spec_keep_func': None,
            'iteration': self.iterations,
        }
        # Escape hatch for re-measurement only.
        if os.environ.get('PHASE_PROBING_VIVIFY_OPT_ALPHA', '') == '1':
            opt_args['enable_alpha_crown'] = True
        self.net.set_bound_opts({
            'optimize_bound_args': opt_args,
            'enable_opt_interm_bounds': False,
        })
        # Verify WITH the GCP-CROWN cut pool, exactly like BICCOS's own
        # clause re-verification (biccos_verification): pool cuts are
        # entailed on the counterexample-relevant region of THIS spec, so
        # they may strengthen the probe. Fresh zero-init general betas per
        # chunk; BaB re-installs its own via set_cut_params next round
        # (the same overwrite biccos_verification already does). In 'pool'
        # mode the module seen here is the probe-scoped one installed by
        # _install_probe_cut_pool; 'off' disables cut terms entirely.
        use_cuts = (self.use_cuts_mode != 'off'
                    and getattr(self.net, 'cut_module', None) is not None
                    and arguments.Config['bab']['cut']['bab_cut'])
        if use_cuts:
            self.net.cut_used = True
            self.model.set_cut_params(B, B, None)
        with torch.enable_grad():
            lb_out = self.net.compute_bounds(
                x=(new_x,), C=C, method='CROWN-optimized',
                interm_bounds=probe_ib, bound_upper=False,
                cutter=self.model.cutter if use_cuts else None)[0]
        if use_cuts:
            # _swap_net_state's undo restores the pre-vivify cut flags.
            self.net.cut_used = False
            for m in self.net.splittable_activations:
                m.cut_used = False
        return self._verified(lb_out.detach())

    def _probe_all(self, cand_pins):
        """Run all candidates through _run_chunk with OOM backoff.
        Returns list of bool (verified), aligned with cand_pins."""
        results = []
        pos, cur = 0, self.batch_size
        while pos < len(cand_pins):
            chunk = cand_pins[pos:pos + cur]
            try:
                verified = self._run_chunk(chunk)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                if cur == 1:
                    raise
                cur = max(1, cur // 2)
                print('Phase probing joint vivification: CUDA OOM, reducing '
                      f'chunk to {cur} probes.')
                continue
            results.extend(bool(v) for v in verified)
            pos += len(chunk)
        return results

    def _swap_net_state(self):
        """Slice the net's current final-start-node alphas to batch 1
        (a valid warm start for any pinned subregion, spec-consistent with
        this BaB run's clause) and disable beta/cut terms. Returns an undo
        closure restoring the exact previous state."""
        saved_alpha, saved_nonfinal = {}, {}
        self._alpha_slice = {}
        acts = {m.name: m for m in self.net.get_enabled_opt_act()}
        if self.grade in ('alpha', 'beta'):
            for name, m in acts.items():
                a = m.alpha.get(self.final_name) \
                    if hasattr(m, 'alpha') else None
                if isinstance(a, torch.Tensor) and a.dim() >= 3 \
                        and a.shape[2] >= 1:
                    saved_alpha[name] = a
                    self._alpha_slice[name] = a[:, :, :1].detach().clone()
                    m.alpha[self.final_name] = self._alpha_slice[name]
        if self.grade == 'beta':
            # Stash every non-final-start-node alpha for the duration of
            # the probes: the optimizer's best-alpha bookkeeping indexes ALL
            # alpha entries by the probe batch dim, and any retained entries
            # with a different batch dim trigger a device-side assert (see
            # _PhaseProber.run_beta_rung).
            for name, m in acts.items():
                for spec in [s for s in m.alpha
                             if s != self.final_name]:
                    saved_nonfinal.setdefault(name, {})[spec] = \
                        m.alpha.pop(spec)
        # BOTH gates must be saved: the probes disable alpha optimization
        # (see the enable_alpha_crown note in _run_chunk) and leaving that
        # on the net starves every subsequent BaB bound computation --
        # measured 2026-07-19, cifar100 idx0 safe 73s -> unknown 114s at
        # 400 domains when only enable_beta_crown was restored.
        prev_opt = {
            k: self.net.bound_opts['optimize_bound_args'][k]
            for k in ('enable_alpha_crown', 'enable_beta_crown')}
        self.net.set_bound_opts(
            {'optimize_bound_args': {'enable_beta_crown': False}})
        prev_cut_net = getattr(self.net, 'cut_used', False)
        prev_cut_act = {m.name: m.cut_used
                        for m in self.net.splittable_activations}
        self.net.cut_used = False
        for m in self.net.splittable_activations:
            m.cut_used = False

        def undo():
            for name, a in saved_alpha.items():
                acts[name].alpha[self.final_name] = a
            for name, entries in saved_nonfinal.items():
                acts[name].alpha.update(entries)
            self.net.set_bound_opts({'optimize_bound_args': dict(prev_opt)})
            self.net.cut_used = prev_cut_net
            for m in self.net.splittable_activations:
                if m.name in prev_cut_act:
                    m.cut_used = prev_cut_act[m.name]
        return undo

    def _install_probe_cut_pool(self, fresh_cuts, pool_cuts):
        """'pool' mode: build a probe-scoped GCP-CROWN cut module over the
        FULL current pool -- BICCOS clauses of previous rounds (already in
        their vivified/merged form), cplex cuts, phase probing's pending
        implied-bound cuts (box-sound, installable before their official
        install) and this round's freshly inferred clauses. All of these
        are entailed on the counterexample-relevant region of THIS run's
        spec (the same conditioning the probes and the clauses share), so
        the oracle may use them: clause A strengthens the probe testing
        clause B, closing the loop between pool and oracle.

        Including the fresh clauses means a clause participates in its own
        probes; for prefixes j < n that is standard SAT-vivification
        semantics and sound (every counterexample in the pinned region
        satisfies the entailed clause), and it cannot fake a shortening
        propositionally because root-false literals were screened by
        _parse. The j = n diagnostic however becomes self-certifying: the
        full pin set contradicts the clause itself, so full-pin
        'verified' then measures the optimizer's cut exploitation, not
        oracle grade adequacy.

        The swap is fully undone afterwards: BICCOS's own rebuild logic at
        the end of update_cut must see the exact pre-vivify cutter/module
        state (including rounds where it decides NOT to rebuild).
        Returns an undo closure, or None when there is nothing to install.
        """
        cutter = getattr(self.model, 'cutter', None)
        if cutter is None:
            return None
        pending = list(getattr(
            self.model, 'phase_probing_pending_cuts', None) or [])
        seen, pool = set(), []
        for cut in list(pool_cuts or []) + pending + list(fresh_cuts or []):
            key = json.dumps(cut, sort_keys=True)
            if key not in seen:
                seen.add(key)
                pool.append(cut)
        if not pool:
            return None
        saved_cuts = cutter.cuts
        saved_cutter_module = cutter.cut_module
        saved_net_module = getattr(self.net, 'cut_module', None)
        saved_act_modules = {
            m.name: getattr(m, 'cut_module', None) for m in self.net.relus}
        cutter.cuts = pool
        # construct_cut_module also resets per-relu transient attrs
        # (masked_beta, *_beta_used, *_coeffs) -- all per-bound-call state
        # that the next beta attach / official rebuild re-establishes --
        # and flips cut_used flags, which _swap_net_state's undo restores.
        try:
            probe_module = cutter.construct_cut_module()
        except Exception:
            cutter.cuts = saved_cuts
            cutter.cut_module = saved_cutter_module
            raise
        self.net.cut_module = probe_module
        for m in self.net.relus:
            m.cut_module = probe_module
        self.stats['cut_pool_size'] = len(pool)
        self.stats['cut_pool_rounds'] += 1

        def undo():
            cutter.cuts = saved_cuts
            cutter.cut_module = saved_cutter_module
            self.net.cut_module = saved_net_module
            for m in self.net.relus:
                m.cut_module = saved_act_modules.get(m.name)
        return undo

    # -- entry point -----------------------------------------------------

    def vivify(self, cuts, d, pool_cuts=None):
        """Joint-pin descent over freshly inferred blocking clauses.
        `cuts` is modified in place; `d` is the picked-domain batch dict
        whose cs/thresholds define THIS BaB run's clause (the conditioning
        of the cuts). Prints per-call and cumulative stats."""
        st = self.stats
        st['pool'] += len(cuts)
        if self.budget <= 0:
            return
        cs, rhs = d.get('cs'), d.get('thresholds')
        if not isinstance(cs, torch.Tensor) or cs.numel() == 0 \
                or not isinstance(rhs, torch.Tensor):
            return
        if not bool((cs == cs[:1]).all()) or not bool((rhs == rhs[:1]).all()):
            # Heterogeneous specs in one batch (jointly optimized OR
            # groups): per-clause conditioning is ambiguous, skip.
            st['skipped_multi_c'] += 1
            return
        self.cur_c = cs[:1].detach().to(self.device)
        self.cur_rhs = rhs[:1].detach()
        # GPU dry-round gate state is per BaB run (same scoping as the
        # clause conditioning).
        gate_key = (cs[0].detach().cpu().numpy().tobytes(),
                    rhs[0].detach().cpu().numpy().tobytes())
        if gate_key != self._gpu_run_key:
            self._gpu_run_key = gate_key
            self._gpu_dry = 0
        gpu_off = 0 < self.dry_rounds <= self._gpu_dry
        _sl = getattr(self.model, 'phase_probing_sat_layer', None)
        # PHASE_PROBING_REPROBE_DIAG=1: price the re-probe premise only
        # (plain CROWN, no behavior change -- a LOWER bound on movement,
        # since the real oracle is beta grade + pool cuts).
        if os.environ.get('PHASE_PROBING_REPROBE_DIAG', '') == '1' \
                and _sl is not None:
            self._reprobe_premise_diagnostic(_sl)
        # Conditioned re-probing at depth (see reprobe()).
        if self.reprobe_enabled and _sl is not None:
            self.reprobe(_sl, _sl.run_key_of(cs, rhs), pool_cuts)
        # Measurement knob PHASE_PROBING_VIVIFY_NO_GPU=1: gate the GPU
        # descent off from the first round (the dry-round gate taken to its
        # limit) while every zero-GPU duty -- mirror cores, BCP,
        # failed-literal probing, domain filtering -- keeps running. Prices
        # the GPU vivification stage as a whole rather than one term inside
        # its optimizer.
        if os.environ.get('PHASE_PROBING_VIVIFY_NO_GPU', '') == '1':
            gpu_off = True
        # BCP pre-pass source: the SAT layer's clause DB, scoped to THIS
        # run (this round's fresh clauses are mirrored only AFTER
        # vivification, so a clause never propagates against itself).
        bcp_layer, bcp_run_key = None, None
        if self.bcp:
            bcp_layer = getattr(self.model, 'phase_probing_sat_layer', None)
            if bcp_layer is None:
                if not getattr(self, '_bcp_warned', False):
                    self._bcp_warned = True
                    print('Phase probing joint vivification: vivify_bcp '
                          'requested but the SAT layer is not armed '
                          '(enable --phase_probing_sat_layer); BCP '
                          'pre-pass disabled.')
            else:
                bcp_run_key = bcp_layer.run_key_of(cs, rhs)
        mirror = (self.mirror and bcp_layer is not None
                  and bcp_run_key is not None)
        if mirror and bcp_run_key == bcp_layer.run_key \
                and bcp_layer.run_unsat:
            # This OR group already carries an UNSAT certificate: it is
            # verified, its domains will be pruned at the next pick_out --
            # no clause of this run is worth a probe.
            return
        # Parse + screen CPU-side; collect (cut, ordered lits, prefix js).
        jobs, cand_pins, cand_key = [], [], []
        bcp_min = {}  # job_id -> prefix length proven by BCP conflict alone
        for cut in cuts:
            lits, removed_root = self._parse(cut)
            if lits is None:
                continue
            if removed_root and lits:
                self._rewrite(cut, lits)
                st['removed_lits'] += removed_root
                st['removed_root'] += removed_root
                st['shortened'] += 1
            n = len(lits)
            if n < 2 or n > self.max_lits:
                continue
            st['eligible'] += 1
            st['len_hist'][n] = st['len_hist'].get(n, 0) + 1
            # Mirror oracle pre-pass: one conflict-bounded assumption-core
            # solve per clause. An UNSAT core is the shortened clause --
            # full conflict analysis over everything the SAT layer holds,
            # zero GPU -- strictly stronger than the propagation-only
            # conflict test of the BCP loop below (which is kept purely
            # for its implied extra pins).
            if mirror:
                status, kept = bcp_layer.vivify_clause_pins(
                    lits, bcp_run_key)
                if status == 'unsat_run':
                    # Per-run UNSAT certificate: the OR group is verified;
                    # every queued probe of this run is dead weight.
                    st['mirror_skipped_probes'] += n + len(cand_pins)
                    print('Phase probing joint vivification: mirror UNSAT '
                          'certificate for this run -- skipping all '
                          'remaining vivification probes.')
                    return
                if status == 'core' and len(kept) < n:
                    short = [lits[i] for i in kept]
                    self._rewrite(cut, short)
                    st['shortened'] += 1
                    st['removed_lits'] += n - len(kept)
                    st['mirror_shortened'] += 1
                    st['mirror_removed_lits'] += n - len(kept)
                    st['mirror_skipped_probes'] += n
                    continue
            job_id = len(jobs)
            jobs.append((cut, lits))
            # BCP pre-pass: propagate each prefix's pin set through the
            # clause DB, incrementally (assumption ordering = prefix
            # ordering, so every implied literal's antecedents lie within
            # the prefix). A conflict at prefix j proves the pinned region
            # empty of counterexamples by propagation alone -- the clause
            # shortens to j literals with ZERO GPU cost, and probes for
            # j' >= j are pointless (already entailed). Without a
            # conflict, the implied literals join the prefix's pin set,
            # making the GPU probe strictly stronger.
            bcp_stop, extras = None, {}
            if bcp_layer is not None and bcp_run_key is not None:
                tb = time.time()
                for j in range(1, n + 1):
                    ok, implied = bcp_layer.propagate_pins(
                        lits[:j], bcp_run_key)
                    if not ok:
                        bcp_stop = j
                        break
                    if implied:
                        extras[j] = implied
                st['bcp_time'] += time.time() - tb
                if bcp_stop is not None:
                    st['bcp_skipped_probes'] += n - bcp_stop + 1
                    if bcp_stop < n:
                        bcp_min[job_id] = bcp_stop
                        st['bcp_shortened'] += 1
                        st['bcp_removed_lits'] += n - bcp_stop
            # Prefixes j = 1..n-1 shorten; j = n is the grade-adequacy
            # diagnostic (a clause whose FULL pin set does not verify at
            # this grade can never shorten here). BCP-conflicted prefixes
            # (j >= bcp_stop) are already entailed and are not probed.
            j_max = (bcp_stop - 1) if bcp_stop is not None else n
            if gpu_off:
                st['gpu_gated_probes'] = st.get('gpu_gated_probes', 0) + j_max
                j_max = 0
            for j in range(1, j_max + 1):
                if len(cand_pins) >= self.budget:
                    break
                # Pin the NEGATIONS of the prefix literals: the negation
                # of the literal for coefficient s is "phase(s)".
                extra = extras.get(j, [])
                st['bcp_pins'] += len(extra)
                cand_pins.append(list(lits[:j]) + list(extra))
                cand_key.append((job_id, j))
        if not cand_pins and not bcp_min:
            return
        t0 = time.time()
        results = []
        if cand_pins:
            undo = self._swap_net_state()
            pool_undo = None
            try:
                if (self.use_cuts_mode == 'pool' and self.grade == 'beta'
                        and arguments.Config['bab']['cut']['bab_cut']):
                    pool_undo = self._install_probe_cut_pool(cuts, pool_cuts)
                with torch.no_grad():
                    results = self._probe_all(cand_pins)
            finally:
                if pool_undo is not None:
                    pool_undo()
                undo()
        self.budget -= len(cand_pins)
        st['probes'] += len(cand_pins)
        st['gpu_time'] += time.time() - t0
        # Fold results: minimal verified prefix per clause (BCP conflicts
        # seed the fold; GPU results can only improve on them).
        min_j = dict(bcp_min)
        for (job_id, j), ok in zip(cand_key, results):
            n = len(jobs[job_id][1])
            if j == n:
                st['full_tested'] += 1
                st['full_verified'] += bool(ok)
            if ok and j < n:
                min_j[job_id] = min(min_j.get(job_id, n), j)
        for job_id, j in min_j.items():
            cut, lits = jobs[job_id]
            st['removed_lits'] += len(lits) - j
            st['shortened'] += 1
            self._rewrite(cut, lits[:j])
        # Dry-round accounting: a round counts as dry when the GPU probes
        # shortened nothing the zero-GPU passes had not already found.
        if self.dry_rounds > 0 and not gpu_off and cand_pins:
            gpu_new = sum(1 for jid, j in min_j.items()
                          if j < bcp_min.get(jid, len(jobs[jid][1])))
            self._gpu_dry = 0 if gpu_new else self._gpu_dry + 1
            if self._gpu_dry == self.dry_rounds:
                print('Phase probing joint vivification: '
                      f'{self.dry_rounds} consecutive GPU-dry rounds -- '
                      'gating GPU probes off for this run (CPU passes '
                      'continue).')
        bcp_part = ''
        if bcp_layer is not None:
            bcp_part = (f"bcp_shortened={st['bcp_shortened']} "
                        f"(-{st['bcp_removed_lits']} lits, 0 GPU), "
                        f"bcp_pins={st['bcp_pins']}, "
                        f"bcp_skipped_probes={st['bcp_skipped_probes']}, "
                        f"bcp_time={st['bcp_time']:.2f}s, ")
        if mirror:
            ms = bcp_layer.stats
            bcp_part += (
                f"mirror_shortened={st['mirror_shortened']} "
                f"(-{st['mirror_removed_lits']} lits, 0 GPU), "
                f"mirror_skipped_probes={st['mirror_skipped_probes']}, "
                f"mirror_calls={ms['mirror_calls']} "
                f"(unsat={ms['mirror_unsat']}, sat={ms['mirror_sat']}, "
                f"unknown={ms['mirror_unknown']}, "
                f"time={ms['mirror_time']:.3f}s), "
                f"flp_units={ms['flp_units']}, ")
        if st.get('gpu_gated_probes'):
            bcp_part += (f"gpu_gated_probes={st['gpu_gated_probes']} "
                         f"(dry={self._gpu_dry}), ")
        pool_part = ''
        if st['cut_pool_rounds']:
            pool_part = (f"probe_cut_pool={st['cut_pool_size']} cuts "
                         f"({st['cut_pool_rounds']} rounds), ")
        print('Phase probing joint vivification: '
              f'{len(min_j)} of {len(jobs)} clauses shortened this round '
              f'({len(cand_pins)} probes, {time.time() - t0:.2f}s); '
              f"cumulative: pool={st['pool']}, eligible={st['eligible']}, "
              f"shortened={st['shortened']}, "
              f"literals_removed={st['removed_lits']} "
              f"(root-screened {st['removed_root']}), "
              f"full-pin verified {st['full_verified']}/{st['full_tested']}, "
              f"probes={st['probes']}, gpu_time={st['gpu_time']:.2f}s, "
              + bcp_part + pool_part
              + f'budget_left={self.budget}, '
              f"len_hist={dict(sorted(st['len_hist'].items()))}")


def probe_and_refine(model, x, c, rhs, or_spec_size, spec_handler, ret):
    """Wrapper around _probe_and_refine (see its docstring) that ALWAYS
    releases the alphas retained through build() for the alpha rung
    (LiRPANet.alpha_drop_unused), on every exit path, so BaB sees exactly
    the usual post-build alpha state and memory footprint."""
    try:
        return _probe_and_refine(model, x, c, rhs, or_spec_size,
                                 spec_handler, ret)
    finally:
        if getattr(model, 'phase_probing_keep_alpha_nodes', None):
            model.phase_probing_keep_alpha_nodes = None
            model.alpha_drop_unused()


def _probe_and_refine(model, x, c, rhs, or_spec_size, spec_handler, ret):
    """Run phase probing and refine ret's intermediate bounds in place.

    Args:
        model: LiRPANet built by the incomplete verifier (initial bounds done).
        x, c, rhs, or_spec_size: the (joint) specification exactly as fed to
            model.build() -- NOT the pruned versions.
        spec_handler: SpecHandler of the incomplete verifier (provides the
            stop criterion and the original unverified-OR mask).
        ret: incomplete verifier result dict; ret['lower_bounds'] /
            ret['upper_bounds'] are refined IN PLACE.

    Returns:
        (verified_early, stats): verified_early is True when probing alone
        proves the property (spec closes); stats is a dict of summary values.

    Side products stashed on `model` (consumed later in the cut pipeline,
    see cuts/infered_cuts.py):
        model.phase_probing_pending_cuts     -- implied-bound GCP-CROWN cuts
        model.phase_probing_implications_int -- phase implication edges
        model.phase_probing_forced_int       -- forced phases (relu-idx keys)
    """
    cfg = arguments.Config['solver']['phase_probing']
    start_time = time.time()

    def _summary(stats, msg=None):
        stats['time'] = time.time() - start_time
        if msg:
            print(f'Phase probing: skipped ({msg}).')
        esc = stats.get('escalated', {})
        print(
            'Phase probing summary: '
            f"probed={stats['probed_neurons']} neurons, "
            f"oracle={cfg['oracle']} "
            f"(escalated: alpha={esc.get('alpha', 0)}, "
            f"beta={esc.get('beta', 0)}, gurobi={esc.get('gurobi', 0)}), "
            f"forced_phases={stats['forced_phases']} "
            f"(dropped_by_mip={stats['forced_dropped_by_mip']}), "
            f"tightened={stats['tightened_neurons']} neurons, "
            f"avg_tightening={stats['avg_tightening']:.6f}, "
            f"max_tightening={stats['max_tightening']:.6f}, "
            f"implication_edges={stats['implication_edges']}, "
            f"implied_cuts={stats['implied_cuts']}, "
            + (f"coverage_truncated={stats['coverage_truncated']} neurons, "
               if stats.get('coverage_truncated') else '')
            + f"time={stats['time']:.2f}s"
            + (f" (rung: " + ', '.join(
                f'{r}={t:.2f}s' for r, t in sorted(stats['rung_time'].items())
              ) + ')' if stats.get('rung_time') else '')
        )

    empty_stats = {
        'probed_neurons': 0, 'forced_phases': 0, 'forced_dropped_by_mip': 0,
        'tightened_neurons': 0, 'avg_tightening': 0.0, 'max_tightening': 0.0,
        'escalated': {}, 'implication_edges': 0, 'implied_cuts': 0,
        'time': 0.0,
    }

    # ------------------------------------------------------------------
    # Feasibility guards (v1 limitations).
    # ------------------------------------------------------------------
    lb_dict, ub_dict = ret.get('lower_bounds'), ret.get('upper_bounds')
    if not lb_dict or not ub_dict:
        _summary(empty_stats, 'no intermediate bounds available')
        return False, empty_stats
    if arguments.Config['solving']['solving_mode']:
        _summary(empty_stats, 'solving mode is not supported')
        return False, empty_stats
    if arguments.Config['bab']['branching']['input_split']['enable']:
        _summary(empty_stats, 'input split is not supported')
        return False, empty_stats
    if getattr(model.net, 'cut_used', False) \
            or getattr(model.net, 'cut_module', None) is not None:
        # Cuts merely being ENABLED is fine: the cut module is only built
        # during BaB, after this hook, so probing runs BEFORE cut
        # initialization. We only bail out if cut terms are already active
        # in bound computations, which would make probe results depend on
        # cut betas.
        _summary(empty_stats, 'cut module already active')
        return False, empty_stats

    final_name = model.final_name
    interm_names = [
        k for k in lb_dict
        if k != final_name and isinstance(lb_dict[k], torch.Tensor)
    ]
    if any(lb_dict[k].shape[0] != 1 for k in interm_names):
        # Disjuncts optimized separately: intermediate bounds are per-OR.
        # Probing per (OR, neuron) pair is possible but not implemented.
        _summary(empty_stats, 'intermediate bounds are not shared '
                              '(batch size > 1)')
        return False, empty_stats

    prober = _PhaseProber(model, x, c, rhs, or_spec_size, spec_handler, ret)
    stats = prober.stats
    if not prober.relu_preacts:
        _summary(stats, 'no ReLU pre-activation layers with bounds')
        return False, stats

    # ------------------------------------------------------------------
    # Candidate selection.
    # ------------------------------------------------------------------
    # Ranking mode. The interval score |lb*ub|/(ub-lb) is a purely geometric
    # proxy: it says a neuron is undecided, not that deciding it matters to the
    # SPEC. BaBSR weights the same interval term by the neuron's output
    # sensitivity |lA| (the coefficient CROWN's backward pass already produced),
    # which is what actually predicts whether a pin can refute anything. Same
    # cost -- lA is read from `ret`, no extra bound computation.
    # `--phase_probing_select interval` restores the old ranking.
    select = cfg.get('select', 'babsr')
    lA = ret.get('lA') or {}
    relu_name_of_preact = {}
    if select == 'babsr' and lA:
        relus = getattr(prober.net, 'relus', [])
        for pre_name, ridx in prober.relu_idx_of_preact.items():
            if ridx < len(relus):
                relu_name_of_preact[pre_name] = relus[ridx].name

    candidates = []
    babsr_used = 0
    for name in prober.relu_preacts:
        l = prober.orig_lb[name].reshape(-1)
        u = prober.orig_ub[name].reshape(-1)
        unstable = prober.unstable_mask[name].nonzero().reshape(-1)
        if unstable.numel() == 0:
            continue
        # Interval term, shared by both modes (BaBSR's "intercept" factor).
        score = (l[unstable] * u[unstable]).abs() \
            / (u[unstable] - l[unstable]).clamp(min=1e-12)
        if select == 'babsr':
            # Weight by output sensitivity: mean |lA| over the spec rows,
            # matching heuristics/babsr.py:babsr_score_intercept_only. Any
            # layer whose lA is missing or mis-shaped silently keeps the
            # interval score, so this can only reorder, never crash.
            a = lA.get(relu_name_of_preact.get(name))
            if a is not None:
                try:
                    w = a.reshape(-1, a.shape[-1]) if a.dim() > 2 else a
                    w = w.abs().mean(dim=0).reshape(-1).to(score.device)
                    if w.numel() == l.numel():
                        score = score * w[unstable].clamp(min=1e-12)
                        babsr_used += 1
                except Exception:
                    pass
        for idx, s in zip(unstable.tolist(), score.tolist()):
            candidates.append((name, idx, s))
    stats['select_mode'] = select
    stats['select_babsr_layers'] = babsr_used
    if not candidates:
        _summary(stats, 'no unstable ReLU neurons')
        return False, stats
    max_neurons = cfg['max_neurons']
    if max_neurons > 0 and len(candidates) > max_neurons:
        candidates.sort(key=lambda t: -t[2])
        candidates = candidates[:max_neurons]

    # ------------------------------------------------------------------
    # Probe oracle ladder.
    # ------------------------------------------------------------------
    # Free retained alphas the alpha rung can never use (start nodes not
    # downstream of any probed layer) before the memory-heavy probes.
    prober.trim_retained_alphas(candidates)
    prober.run_ladder(candidates)

    if prober.both_verified is not None:
        layer_name, nidx = prober.both_verified
        print(f'Phase probing: both phases of neuron {nidx} in layer '
              f'{layer_name} are verified; property is verified.')
        _summary(stats)
        return True, stats

    forced = prober.forced_phases()
    stats['forced_phases'] = len(forced)
    stats['implication_edges'] = prober._num_edges
    if prober._margin_gains:
        gains = prober._margin_gains
        print(f'Phase probing: escalation improved probe output margins by '
              f'avg {sum(gains) / len(gains):.6f} / max {max(gains):.6f} '
              f'over {len(gains)} escalated probe evaluations.')

    # ------------------------------------------------------------------
    # Optional exact MIP confirmation of forced phases (secondary feature).
    # ------------------------------------------------------------------
    apply_forced = cfg['apply_forced_phases']
    if forced and apply_forced and cfg['mip_confirm']:
        confirmed = {}
        for (layer, nidx), phase in forced.items():
            # The probe-verified region is the OPPOSITE of the forced phase.
            region_sign = +1 if phase == 'inactive' else -1
            res = prober._gurobi_region_verified(layer, nidx, region_sign)
            if res is None:
                confirmed = forced  # gurobi unavailable: graceful skip
                break
            if res:
                confirmed[(layer, nidx)] = phase
        dropped = len(forced) - len(confirmed)
        if dropped:
            print(f'Phase probing: {dropped} forced phases NOT confirmed by '
                  'MIP, dropping them.')
        stats['forced_dropped_by_mip'] = dropped
        forced = confirmed

    # ------------------------------------------------------------------
    # Apply forced phases (SPEC-CONDITIONAL clamps -- sound only for this
    # verification; see module docstring).
    # ------------------------------------------------------------------
    ref_lb, ref_ub = prober.ref_lb, prober.ref_ub
    orig_lb, orig_ub = prober.orig_lb, prober.orig_ub
    verified_early = False
    if forced and apply_forced:
        for (layer_name, nidx), phase in forced.items():
            fl = ref_lb[layer_name].view(-1)
            fu = ref_ub[layer_name].view(-1)
            if phase == 'active':
                fl[nidx] = torch.clamp(fl[nidx], min=0.)
            else:
                fu[nidx] = torch.clamp(fu[nidx], max=0.)
            if fl[nidx] > fu[nidx]:
                # The forced phase contradicts the (hull-refined) bounds:
                # then all real inputs live in the probe-verified region
                # => property verified.
                print(f'Phase probing: forced phase of neuron {nidx} in '
                      f'layer {layer_name} contradicts refined bounds; '
                      'property is verified.')
                verified_early = True
                fl[nidx], fu[nidx] = 0., 0.  # keep bounds consistent

    # ------------------------------------------------------------------
    # Soundness self-check: never loosen, and keep lb <= ub.
    # ------------------------------------------------------------------
    tol = 1e-6
    total_tightening, tightened = 0., 0
    for k in interm_names:
        assert (ref_lb[k] >= orig_lb[k] - tol).all(), \
            f'Phase probing loosened a lower bound of {k}'
        assert (ref_ub[k] <= orig_ub[k] + tol).all(), \
            f'Phase probing loosened an upper bound of {k}'
        assert (ref_lb[k] <= ref_ub[k] + tol).all(), \
            f'Phase probing produced lb > ub for {k}'
        delta = (ref_lb[k] - orig_lb[k]) + (orig_ub[k] - ref_ub[k])
        t_mask = delta > 0
        tightened += int(t_mask.sum())
        if t_mask.any():
            total_tightening += float(delta[t_mask].sum())
            stats['max_tightening'] = max(stats['max_tightening'],
                                          float(delta.max()))
    stats['tightened_neurons'] = tightened
    stats['avg_tightening'] = total_tightening / max(tightened, 1)

    # ------------------------------------------------------------------
    # Final output bound recomputation with the refined bounds (once).
    # Reuses the initial alphas when available (needs the model.c clause
    # mapping); otherwise plain CROWN.
    # ------------------------------------------------------------------
    final_lb = None
    if not verified_early:
        use_alpha = prober._setup_sel() and bool(prober._saved_final_alphas())
        sel = prober.sel if use_alpha else None
        probe_ib = {k: [ref_lb[k], ref_ub[k]] for k in interm_names}
        new_x, C = prober._probe_x_c(1, sel)
        with torch.no_grad():
            lb = model.net.compute_bounds(
                x=(new_x,), C=C, method='backward', reuse_alpha=use_alpha,
                interm_bounds=probe_ib, bound_upper=False)[0]
        if bool(prober.checker.region_verified(lb, sel)[0]):
            print('Phase probing: specification verified after hull-based '
                  'bound refinement.')
            verified_early = True
        if sel is None:
            final_lb = lb

    if verified_early:
        _summary(stats)
        return True, stats

    # ------------------------------------------------------------------
    # Write the refined bounds back into ret (in place) so they reach BaB
    # as its reference/intermediate bounds via build_with_refined_bounds().
    # ------------------------------------------------------------------
    for k in interm_names:
        old = lb_dict[k]
        lb_dict[k] = ref_lb[k].to(device=old.device, dtype=old.dtype)
        old = ub_dict[k]
        ub_dict[k] = ref_ub[k].to(device=old.device, dtype=old.dtype)

    # For a single-OR spec the probe output shape matches the stored final
    # bounds; fold in the recomputed output lower bound (max of two valid
    # lower bounds is valid). Other spec types have post-processed shapes.
    if (final_lb is not None and spec_handler.spec_type.name == 'SINGLE_OR'
            and ret.get('global_lb') is not None
            and ret['global_lb'].shape == final_lb.shape):
        improved = torch.maximum(
            ret['global_lb'], final_lb.to(ret['global_lb']))
        ret['global_lb'] = improved
        if isinstance(lb_dict.get(final_name), torch.Tensor) \
                and lb_dict[final_name].shape == improved.shape:
            lb_dict[final_name] = improved.to(lb_dict[final_name])

    # ------------------------------------------------------------------
    # Stash side products for the cut pipeline (installed later; see
    # cuts/infered_cuts.py). Implied cuts and implication edges are
    # box-sound; forced facts are spec-conditional (acceptable for BICCOS,
    # whose own cuts are spec-conditional too).
    # ------------------------------------------------------------------
    if prober.collect_cuts:
        model.phase_probing_pending_cuts = prober.emit_implied_cuts()
        stats['implied_cuts'] = len(model.phase_probing_pending_cuts)
        print(f'Phase probing: emitted {stats["implied_cuts"]} implied-bound '
              'cuts (pending until the cut module is built).')
    if cfg['vivify_joint'] and arguments.Config['bab']['cut']['enabled']:
        # Joint-pin descent vivifier for BICCOS clauses (see ClauseVivifier);
        # consumed by BICCOS.update_cut on every cut-inference round.
        model.phase_probing_vivifier = ClauseVivifier(prober, model)
        print('Phase probing: joint-pin clause vivifier armed '
              f'(grade {model.phase_probing_vivifier.grade}, '
              f"budget {cfg['vivify_budget']} probes).")
    if cfg['sat_layer']:
        # CPU clause DB over phase literals (see sat_layer.py); filters
        # picked BaB domains by unit propagation, fed by BICCOS.update_cut.
        try:
            from sat_layer import PhaseSATLayer
            model.phase_probing_sat_layer = PhaseSATLayer(
                model, prober.implication_edges_int(),
                {(prober.relu_idx_of_preact[layer], nidx):
                    (+1 if phase == 'active' else -1)
                 for (layer, nidx), phase in
                 (forced.items() if apply_forced else [])
                 if layer in prober.relu_idx_of_preact},
                {v: k for k, v in prober.relu_idx_of_preact.items()})
        except ImportError as e:
            print(f'Phase probing: SAT layer unavailable ({e}), skipping.')
    if cfg['vivify_biccos']:
        model.phase_probing_implications_int = prober.implication_edges_int()
        model.phase_probing_forced_int = {
            (prober.relu_idx_of_preact[layer], nidx):
                (+1 if phase == 'active' else -1)
            for (layer, nidx), phase in
            (forced.items() if apply_forced else [])
            if layer in prober.relu_idx_of_preact
        }

    _summary(stats)
    return False, stats
