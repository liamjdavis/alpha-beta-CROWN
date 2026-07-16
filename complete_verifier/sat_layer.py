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
"""CPU-side clause database over ReLU phase literals (PySAT / CaDiCaL).

A clausal mirror of everything the theory side has proven about phases:

  * phase-probing implication edges     (box-sound, valid everywhere)
  * spec-conditional forced phases      (valid for every OR group of THIS
                                         property, hence every BaB run)
  * BICCOS blocking clauses             (conditioned on ONE BaB run's
                                         clause -- see scoping below)
  * joint-pin vivified clauses          (same conditioning as BICCOS)

Duties (all CPU, zero GPU contention):

  a. PRE-GPU DOMAIN FILTERING: after every pick_out, each domain's phase
     assignment (its split history) is unit-propagated under assumptions;
     a conflict means every completion of the assignment falsifies some
     entailed clause, so the domain contains no counterexample and is
     pruned before any bound computation.
  b. FORCED-PHASE CLAMPING: literals implied by unit propagation must hold
     in every counterexample inside the domain, so the corresponding
     pre-activation bounds are clamped before bounding.
  c. CROSS-DOMAIN TRANSFER is LAZY: every domain passes through pick_out
     before it is bounded, so a clause learned at iteration k
     automatically filters every domain popped after k -- including
     domains created before the clause existed. No queue rescan needed.

CLAUSE SCOPING: BaB runs once per unverified OR group with a single-clause
specification; BICCOS clauses learned in a run are entailed w.r.t. THAT
clause only. The database is therefore split into a PERSISTENT part
(edges + forced phases) and a RUN part (BICCOS/vivified clauses) that is
flushed whenever the picked domains' (cs, thresholds) fingerprint changes.
The CaDiCaL solver is rebuilt on flush (clauses cannot be retracted from
an incremental solver); rebuild cost is microseconds at these sizes.

Inert by default: gated by solver:phase_probing:sat_layer.
"""

import time

import torch

import arguments


class PhaseSATLayer:
    """See module docstring. Constructed at probe time
    (phase_probing._probe_and_refine), stashed on the LiRPANet; consumed by
    bab.act_split_round (domain filtering) and BICCOS.update_cut (clause
    feeding)."""

    def __init__(self, model, implications_int, forced_int,
                 preact_of_relu_idx):
        from pysat.solvers import Cadical195
        self._solver_factory = Cadical195
        self.preact_of_relu_idx = dict(preact_of_relu_idx)
        self._var_of = {}       # (preact_name, neuron_idx) -> int (1-based)
        self.persistent = []    # clauses valid in every BaB run
        self.run_clauses = []   # clauses conditioned on the current run
        self.run_key = None     # fingerprint of the current run's (cs, rhs)
        self._seen = set()      # dedup over run clauses
        self._solver = None
        self.stats = {
            'domains_checked': 0, 'domains_pruned': 0, 'phases_clamped': 0,
            'clauses': 0, 'run_flushes': 0, 'time': 0.0,
        }
        # Persistent facts. Positive literal = phase ACTIVE (z = 1).
        for (ridx, nidx, sign), targets in (implications_int or {}).items():
            src = self._lit_int(ridx, nidx, sign)
            if src is None:
                continue
            for (tr, tn, ts) in targets:
                tgt = self._lit_int(tr, tn, ts)
                if tgt is not None:
                    self.persistent.append([-src, tgt])
        for (ridx, nidx), sign in (forced_int or {}).items():
            lit = self._lit_int(ridx, nidx, sign)
            if lit is not None:
                self.persistent.append([lit])
        print(f'Phase probing SAT layer: armed with {len(self.persistent)} '
              'persistent clauses (implication edges + forced phases).')

    # -- literal encoding -------------------------------------------------

    def _var(self, name, nidx):
        key = (name, nidx)
        v = self._var_of.get(key)
        if v is None:
            v = len(self._var_of) + 1
            self._var_of[key] = v
        return v

    def _lit(self, name, nidx, sign):
        v = self._var(name, nidx)
        return v if sign > 0 else -v

    def _lit_int(self, ridx, nidx, sign):
        name = self.preact_of_relu_idx.get(ridx)
        return None if name is None else self._lit(name, nidx, sign)

    # -- clause feeding ---------------------------------------------------

    @staticmethod
    def run_key_of(cs, rhs):
        """Fingerprint of one BaB run's specification. None when the batch
        mixes specs (jointly optimized OR groups -- no sound scoping)."""
        if not isinstance(cs, torch.Tensor) or not isinstance(
                rhs, torch.Tensor) or cs.numel() == 0:
            return None
        if not bool((cs == cs[:1]).all()) or not bool((rhs == rhs[:1]).all()):
            return None
        return (cs[0].detach().cpu().numpy().tobytes(),
                rhs[0].detach().cpu().numpy().tobytes())

    def _flush_run(self, run_key):
        if run_key != self.run_key:
            if self.run_clauses:
                self.stats['run_flushes'] += 1
            self.run_clauses, self._seen = [], set()
            self.run_key = run_key
            self._solver = None

    def add_blocking_cuts(self, cuts, run_key):
        """Mirror BICCOS blocking clauses (possibly vivified) into the RUN
        database. A blocking clause sum_i s_i z_i <= npos-1 is the
        disjunction OR_i (z_i = 0 if s_i > 0 else z_i = 1)."""
        if run_key is None:
            return
        self._flush_run(run_key)
        added = 0
        for cut in cuts:
            if cut.get('x_decision') or cut.get('relu_decision') \
                    or cut.get('pre_decision') or cut.get('c') != -1:
                continue
            signs = [1 if s > 0 else -1 for s in cut['arelu_coeffs']]
            npos = sum(1 for s in signs if s > 0)
            if abs(float(cut['bias']) - (npos - 1)) > 1e-6:
                continue
            clause = []
            ok = True
            for (ridx, nidx), s in zip(cut['arelu_decision'], signs):
                # literal = "z differs from the blocked phase" = phase(-s)
                lit = self._lit_int(ridx, nidx, -s)
                if lit is None:
                    ok = False
                    break
                clause.append(lit)
            if not ok or not clause:
                continue
            key = frozenset(clause)
            if key in self._seen:
                continue
            self._seen.add(key)
            self.run_clauses.append(sorted(clause))
            if self._solver is not None:
                self._solver.add_clause(sorted(clause))
            added += 1
        if added:
            self.stats['clauses'] += added

    # -- domain filtering -------------------------------------------------

    def _get_solver(self):
        if self._solver is None:
            self._solver = self._solver_factory(
                bootstrap_with=self.persistent + self.run_clauses)
        return self._solver

    def _assumptions_of(self, history):
        """Domain split history -> assumption literals, or None when the
        history contains non-phase splits (nonzero branching points)."""
        lits = []
        for name, entry in history.items():
            locs, signs = entry[0], entry[1]
            points = entry[2] if len(entry) > 2 else None
            n = len(locs)
            if n == 0:
                continue
            for i in range(n):
                if points is not None:
                    p = float(points[i])
                    if abs(p) > 1e-9:
                        return None  # non-zero branching point: not a phase
                s = int(signs[i])
                if s not in (1, -1):
                    return None
                lits.append(self._lit(name, int(locs[i]), s))
        return lits

    def process_picked_domains(self, d, run_key):
        """Duty (a) + (b) on a picked batch dict. Returns (keep_indices or
        None-if-nothing-pruned, num_pruned). Clamps implied phases in
        d's lower/upper bounds in place when the tensors are present."""
        t0 = time.time()
        histories = d.get('history')
        if not histories:
            return None, 0
        if run_key != self.run_key:
            # The batch belongs to a different BaB run than the stored run
            # clauses: flush them (persistent facts remain valid).
            self._flush_run(run_key)
        solver = self._get_solver()
        lb_dict = d.get('lower_bounds')
        ub_dict = d.get('upper_bounds')
        have_bounds = isinstance(lb_dict, dict) and isinstance(ub_dict, dict)
        name_of_var = None
        keep, pruned = [], 0
        for i, hist in enumerate(histories):
            self.stats['domains_checked'] += 1
            lits = self._assumptions_of(hist)
            if lits is None:
                keep.append(i)
                continue
            ok, implied = solver.propagate(assumptions=lits)
            if not ok:
                pruned += 1
                continue
            keep.append(i)
            if have_bounds and implied:
                assumed = set(lits)
                for lit in implied:
                    if lit in assumed:
                        continue
                    if name_of_var is None:
                        name_of_var = {v: k for k, v in self._var_of.items()}
                    name, nidx = name_of_var[abs(lit)]
                    t = lb_dict.get(name) if lit > 0 else ub_dict.get(name)
                    if not isinstance(t, torch.Tensor):
                        continue
                    flat = t.view(t.shape[0], -1)
                    if lit > 0:
                        if flat[i, nidx] < 0:
                            flat[i, nidx] = 0.  # implied ACTIVE: lb -> 0
                            self.stats['phases_clamped'] += 1
                    else:
                        if flat[i, nidx] > 0:
                            flat[i, nidx] = 0.  # implied INACTIVE: ub -> 0
                            self.stats['phases_clamped'] += 1
        self.stats['time'] += time.time() - t0
        st = self.stats
        self.stats['domains_pruned'] += pruned
        report = pruned > 0 or (
            st['domains_checked'] // 1000
            > (st['domains_checked'] - len(histories)) // 1000)
        if report:
            print(f'Phase probing SAT layer: pruned {pruned} of '
                  f'{len(histories)} picked domains by unit propagation '
                  f"(cumulative: checked={st['domains_checked']}, "
                  f"pruned={st['domains_pruned']}, "
                  f"clamped={st['phases_clamped']}, "
                  f"clauses={st['clauses']}, time={st['time']:.3f}s).")
        if pruned == 0:
            return None, 0
        return keep, pruned


def select_domain_batch(d, keep):
    """Restrict a picked-domain batch dict (bab.act_split_round's `d`) to
    the given batch indices, in place. Handles every key pick_out returns
    plus the bounds installed by construct_interm_bounds_in_d. Returns
    False (leaving d untouched beyond tensors already filtered) when an
    unknown structure is encountered -- callers must then use d unfiltered.
    """
    idx = torch.as_tensor(keep, dtype=torch.long)

    def sel(t):
        return t[idx.to(t.device)]

    # The bespoke objects first: they are the only structures that can
    # fail, and failing BEFORE any tensor filtering leaves d untouched.
    bfd = d.get('batch_first_branching_decisions')
    if bfd is not None:
        try:
            bfd.packed_branching_decision = [
                bfd.packed_branching_decision[i] for i in keep]
            bfd.packed_branching_points = [
                bfd.packed_branching_points[i] for i in keep]
            if isinstance(bfd.packed_branching_points_split_depth,
                          torch.Tensor):
                bfd.packed_branching_points_split_depth = \
                    bfd.packed_branching_points_split_depth[idx]
            bfd.batch_size = len(keep)
        except (TypeError, IndexError):
            return False
    scd = d.get('sub_domain_clip_decisions')
    if scd is not None and hasattr(scd, 'index_select'):
        d['sub_domain_clip_decisions'] = scd.index_select(idx)

    for key in ('global_lb', 'cs', 'thresholds', 'x_Ls', 'x_Us',
                'input_split_idx'):
        v = d.get(key)
        if isinstance(v, torch.Tensor):
            d[key] = sel(v)
    for key in ('lAs', 'final_lb', 'final_ub', 'lower_bounds',
                'upper_bounds'):
        v = d.get(key)
        if isinstance(v, dict):
            d[key] = {k: (sel(t) if isinstance(t, torch.Tensor) else t)
                      for k, t in v.items()}
    v = d.get('unstable_bounds')
    if isinstance(v, dict):
        d['unstable_bounds'] = {
            k: [sel(b[0]), sel(b[1])] for k, b in v.items()}
    v = d.get('alphas')
    if isinstance(v, dict):
        for sub in v.values():
            for kk in sub:
                t = sub[kk]
                sub[kk] = t[:, :, idx.to(t.device)]
    for key in ('betas', 'history', 'split_history', 'depths'):
        v = d.get(key)
        if isinstance(v, (list, tuple)):
            d[key] = [v[i] for i in keep]
    return True
