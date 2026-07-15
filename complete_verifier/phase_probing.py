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
        self._gurobi = None  # cached (grb, model, flat out vars) or False
        self._margin_gains = []  # per-escalated-probe margin improvements

        self.stats = {
            'probed_neurons': 0, 'forced_phases': 0,
            'forced_dropped_by_mip': 0, 'tightened_neurons': 0,
            'avg_tightening': 0.0, 'max_tightening': 0.0,
            'escalated': {r: 0 for r in _RUNGS[1:]},
            'implication_edges': 0, 'implied_cuts': 0, 'time': 0.0,
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
                self._collect_edges_and_cuts(layer_name, pairs, k, dl, du,
                                             width)
        for p, nidx in enumerate(pairs):
            key = (layer_name, nidx)
            self.pair_gain[key] = max(self.pair_gain.get(key, 0.),
                                      float(gain[p]))

    def _collect_edges_and_cuts(self, pin_layer, pairs, k, dl, du, width):
        """dl/du: [npairs, 2, ...] downstream bounds of layer k under the
        probes of pin_layer (dim 1: 0=active, 1=inactive). Restricted to
        the ORIGINALLY unstable neurons of layer k."""
        npairs = len(pairs)
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
                        src = (pin_layer, pairs[p], sign)
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
                key = (pin_layer, pairs[p], k, int(uidx[j]), kind)
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

    def run_pair_rung(self, rung, pairs_by_layer):
        """Run all probes of `rung` ('crown' or 'alpha') over
        pairs_by_layer: {layer_name: [nidx, ...]}. Updates probe_state."""
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
                scores = self.checker.score(lb_out, sel)
                for j, nidx in enumerate(chunk):
                    for off, sign in ((0, +1), (1, -1)):
                        s = float(scores[2 * j + off])
                        key = (layer_name, nidx, sign)
                        st = self.probe_state.setdefault(
                            key, {'score': float('-inf'), 'verified': False,
                                  'rung': rung})
                        if rung != 'crown' and st['score'] > float('-inf'):
                            # track how much the stronger oracle moved the
                            # probe's output margin (reporting only)
                            self._margin_gains.append(
                                max(0., s - st['score']))
                        st['score'] = max(st['score'], s)
                        st['verified'] |= s > 0
                        st['rung'] = rung
                    if self.probe_state[(layer_name, nidx, +1)]['verified'] \
                            and self.probe_state[(layer_name, nidx, -1)]['verified']:
                        self.both_verified = (layer_name, nidx)
                pos += len(chunk)
                if rung == 'crown':
                    self.stats['probed_neurons'] += len(chunk)
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
            f"time={stats['time']:.2f}s"
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
    candidates = []
    for name in prober.relu_preacts:
        l = prober.orig_lb[name].reshape(-1)
        u = prober.orig_ub[name].reshape(-1)
        unstable = prober.unstable_mask[name].nonzero().reshape(-1)
        if unstable.numel() == 0:
            continue
        # Instability score |lb*ub|/(ub-lb): high when both phases are wide
        # relative to the total range; neurons whose split is likely to
        # matter most are probed first when max_neurons > 0.
        score = (l[unstable] * u[unstable]).abs() \
            / (u[unstable] - l[unstable]).clamp(min=1e-12)
        for idx, s in zip(unstable.tolist(), score.tolist()):
            candidates.append((name, idx, s))
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
