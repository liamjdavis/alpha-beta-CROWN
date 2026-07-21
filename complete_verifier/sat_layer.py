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
        self.mirror = arguments.Config['solver']['phase_probing']['mirror']
        self.fact_cuts_max = arguments.Config['solver']['phase_probing'][
            'fact_cuts_max']
        self._fact_cuts_emitted = 0  # per-run count against fact_cuts_max
        self.run_unsat = False  # per-run UNSAT certificate (mirror duty 2)
        self._run_unsat_reported = False
        self._flp_last_clauses = 0   # DB size at the last failed-lit pass
        self._flp_decided = set()    # literals already known unit this run
        self._new_unit_lits = []     # flp units not yet exported as cuts
        self._edges_exported = False  # persistent edges exported this run
        self._run_edge_cursor = 0     # run_clauses exported as cuts so far
        self._clause_score = {}       # frozenset(clause) -> strength score
        self.stats = {
            'domains_checked': 0, 'domains_pruned': 0, 'phases_clamped': 0,
            'clauses': 0, 'run_flushes': 0, 'time': 0.0,
            'mirror_calls': 0, 'mirror_unsat': 0, 'mirror_sat': 0,
            'mirror_unknown': 0, 'mirror_time': 0.0,
            'flp_solves': 0, 'flp_units': 0, 'flp_time': 0.0,
            'unsat_runs': 0,
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
        self._persistent_keys = {frozenset(c) for c in self.persistent}
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
            self.run_unsat = False
            self._run_unsat_reported = False
            self._flp_last_clauses = 0
            self._flp_decided = set()
            self._new_unit_lits = []
            self._edges_exported = False
            self._run_edge_cursor = 0
            self._clause_score = {}
            self._fact_cuts_emitted = 0

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
            if key in self._seen or key in self._persistent_keys:
                continue
            self._seen.add(key)
            self.run_clauses.append(sorted(clause))
            if self._solver is not None:
                self._solver.add_clause(sorted(clause))
            added += 1
        if added:
            self.stats['clauses'] += added
        if self.mirror and added:
            self.failed_literal_pass()

    # -- pin-set propagation (vivification BCP) ---------------------------

    def propagate_pins(self, pins, run_key):
        """Unit-propagate a vivification pin set [(relu_idx, neuron_idx,
        sign)] over the clause database scoped to run_key (persistent
        clauses + run clauses of THAT run; a mismatched stored run is
        flushed first, exactly like process_picked_domains).

        Returns (ok, implied):
          ok=False   -- the pins conflict with entailed clauses: the pinned
                        region contains no counterexample of this run's
                        specification (same argument as duty (a)).
          ok=True    -- implied is the list of propagation-implied phase
                        literals as (relu_idx, neuron_idx, sign) tuples,
                        excluding the pins themselves; each holds for every
                        counterexample inside the pinned region.
        Unmappable pins degrade gracefully to (True, [])."""
        if run_key is None:
            return True, []
        if run_key != self.run_key:
            self._flush_run(run_key)
        lits = []
        for (ridx, nidx, sign) in pins:
            lit = self._lit_int(ridx, nidx, sign)
            if lit is None:
                return True, []
            lits.append(lit)
        if not lits:
            # Same PySAT quirk process_picked_domains guards against:
            # propagate(assumptions=[]) reports ok=False on a clause-free
            # solver, and ok=False here claims the pinned region is
            # counterexample-free -- for an empty pin set that region is the
            # whole box. The live caller (vivification prefixes, j >= 1)
            # never passes empty pins; this guards the next caller that does.
            return True, []
        ok, implied = self._get_solver().propagate(assumptions=lits)
        if not ok:
            return False, []
        out = []
        if implied:
            assumed = set(lits)
            if getattr(self, '_name_of_var', None) is None \
                    or len(self._name_of_var) != len(self._var_of):
                self._name_of_var = {
                    v: k for k, v in self._var_of.items()}
            if getattr(self, '_ridx_of_name', None) is None:
                self._ridx_of_name = {
                    v: k for k, v in self.preact_of_relu_idx.items()}
            for lit in implied:
                if lit in assumed:
                    continue
                name, nidx = self._name_of_var[abs(lit)]
                ridx = self._ridx_of_name.get(name)
                if ridx is not None:
                    out.append((ridx, nidx, 1 if lit > 0 else -1))
        return True, out

    # -- mirror-oracle duties (Marabou port) -------------------------------

    def _mark_run_unsat(self):
        """The clause DB of this run is boolean-unsat. Every clause is
        satisfied by the phase assignment of any counterexample of this
        OR group's specification, so an inconsistent DB proves the group
        has no counterexample: the OR group is VERIFIED. Delivered by
        pruning every subsequently picked domain of this run."""
        if not self.run_unsat:
            self.run_unsat = True
            self.stats['unsat_runs'] += 1
            print('Phase probing SAT layer: mirror UNSAT certificate -- '
                  'the entailed clause set of this OR group is '
                  'boolean-unsat; the group is verified, pruning all '
                  'remaining domains.')

    def vivify_clause_pins(self, pins, run_key, conf_budget=200):
        """Assumption-core vivification of one blocking clause, given as
        its pin set [(relu_idx, neuron_idx, sign)] (the NEGATIONS of the
        clause literals). Conflict-bounded solve of DB + pins:

          UNSAT, empty core  -> ('unsat_run', None): per-run UNSAT
                                certificate (see _mark_run_unsat);
          UNSAT, core        -> ('core', kept): kept = indices into pins
                                whose literals form the failed-assumption
                                core; the disjunction of the corresponding
                                clause literals is entailed on its own, so
                                the clause shortens to them (full conflict
                                analysis, zero GPU);
          SAT                -> ('sat', None): no boolean shortening
                                possible against the current DB;
          budget exceeded    -> ('unknown', None).

        The clause itself must not be in the DB yet (callers mirror fresh
        clauses only AFTER their vivification attempt), so it cannot
        refute itself. Unmappable pins degrade to ('unmapped', None)."""
        if run_key is None:
            return 'unmapped', None
        if run_key != self.run_key:
            self._flush_run(run_key)
        if self.run_unsat:
            return 'unsat_run', None
        lits = []
        for (ridx, nidx, sign) in pins:
            lit = self._lit_int(ridx, nidx, sign)
            if lit is None:
                return 'unmapped', None
            lits.append(lit)
        t0 = time.time()
        solver = self._get_solver()
        solver.conf_budget(conf_budget)
        res = solver.solve_limited(assumptions=lits)
        self.stats['mirror_calls'] += 1
        self.stats['mirror_time'] += time.time() - t0
        if res is None:
            self.stats['mirror_unknown'] += 1
            return 'unknown', None
        if res:
            self.stats['mirror_sat'] += 1
            return 'sat', None
        self.stats['mirror_unsat'] += 1
        core = solver.get_core()
        if not core:
            self._mark_run_unsat()
            return 'unsat_run', None
        coreset = set(core)
        kept = [i for i, lit in enumerate(lits) if lit in coreset]
        if not kept:
            # Defensive: a nonempty core disjoint from the assumptions
            # should be impossible (cores are assumption subsets).
            self._mark_run_unsat()
            return 'unsat_run', None
        return 'core', kept

    def failed_literal_pass(self, max_time=1.0, growth_gate=32):
        """Boolean failed-literal probing over the clause DB: for every
        undecided phase variable, propagate each sign; a conflict makes
        the negation a run-scoped unit (a forced phase valid for this OR
        group), and both signs conflicting is a per-run UNSAT
        certificate. Pure propagation, microseconds per literal. Re-runs
        only when the DB has grown since the last pass (Marabou's gate)."""
        if self.run_unsat:
            return
        db_size = len(self.persistent) + len(self.run_clauses)
        if db_size < self._flp_last_clauses + growth_gate:
            return
        self._flp_last_clauses = db_size
        t0 = time.time()
        solver = self._get_solver()
        new_units = 0
        for v in list(self._var_of.values()):
            if v in self._flp_decided or -v in self._flp_decided:
                continue
            failed = []
            for lit in (v, -v):
                ok, _ = solver.propagate(assumptions=[lit])
                self.stats['flp_solves'] += 1
                if not ok:
                    failed.append(lit)
            if len(failed) == 2:
                self._mark_run_unsat()
                break
            if failed:
                unit = -failed[0]
                self._flp_decided.add(unit)
                self._seen.add(frozenset([unit]))
                self.run_clauses.append([unit])
                solver.add_clause([unit])
                self._new_unit_lits.append(unit)
                new_units += 1
                self.stats['flp_units'] += 1
            if time.time() - t0 > max_time:
                break
        self.stats['flp_time'] += time.time() - t0
        if new_units:
            print(f'Phase probing SAT layer: failed-literal probing found '
                  f'{new_units} run-scoped forced phases '
                  f"(cumulative: units={self.stats['flp_units']}, "
                  f"solves={self.stats['flp_solves']}, "
                  f"time={self.stats['flp_time']:.3f}s).")

    def add_run_edge(self, src_ridx, src_nidx, src_sign,
                     tgt_ridx, tgt_nidx, tgt_sign, run_key, score=0.0):
        """Install a run-scoped binary implication (src literal => tgt phase)
        derived by conditioned re-probing.

        Stored as the clause (-src OR tgt), the same encoding the ROOT probe's
        persistent implication edges use -- so the mirror oracle and unit
        propagation treat it identically. RUN-scoped for the same reason as
        add_run_unit: the edge is conditioned on this OR group's spec AND on
        the run's other forced phases, so it must die with the run's flush.

        Returns True if the clause is new.
        """
        if run_key is None or self.run_unsat:
            return False
        if run_key != self.run_key:
            self._flush_run(run_key)
        src = self._lit_int(src_ridx, src_nidx, src_sign)
        tgt = self._lit_int(tgt_ridx, tgt_nidx, tgt_sign)
        if src is None or tgt is None or src == tgt:
            return False
        clause = sorted([-src, tgt])
        key = frozenset(clause)
        if key in self._seen or key in self._persistent_keys:
            return False
        self._seen.add(key)
        self.run_clauses.append(clause)
        # Strength score decides which edges win the scarce cut budget (see
        # pop_new_cut_facts): every installed cut is a general-beta constraint
        # carried by EVERY subproblem's bounding call, so the cap is a cost
        # limit, not a quality one -- the Lagrangian would happily drive a
        # useless cut's multiplier to zero, but we would still pay for it on
        # every domain forever. Same sacrifice BICCOS makes with number_cuts.
        self._clause_score[key] = max(self._clause_score.get(key, 0.0),
                                      float(score))
        if self._solver is not None:
            self._solver.add_clause(clause)
        self.stats['clauses'] += 1
        return True

    def add_run_unit(self, ridx, nidx, sign, run_key):
        """Install a run-scoped forced phase derived outside the boolean
        layer (conditioned re-probing at depth; see
        ClauseVivifier.reprobe). RUN-scoped, never persistent: the fact is
        conditioned on this OR group's spec AND on the run's other forced
        phases, so it must die with the run's DB flush.

        Returns True if the unit is new. The literal also joins
        _new_unit_lits, so it rides the existing fact-cut channel into the
        GCP-CROWN pool exactly like a failed-literal unit.
        """
        if run_key is None or self.run_unsat:
            return False
        if run_key != self.run_key:
            self._flush_run(run_key)
        lit = self._lit_int(ridx, nidx, sign)
        if lit is None:
            return False
        key = frozenset([lit])
        if key in self._seen or lit in self._flp_decided:
            return False
        self._seen.add(key)
        self._flp_decided.add(lit)
        self.run_clauses.append([lit])
        if self._solver is not None:
            self._solver.add_clause([lit])
        self._new_unit_lits.append(lit)
        self.stats['reprobe_units'] = self.stats.get('reprobe_units', 0) + 1
        return True

    def pop_new_cut_facts(self):
        """SAT-derived facts as GCP-CROWN blocking-clause cuts for the
        current run's BICCOS pool, so every subproblem's relaxation gets
        them with optimized multipliers (instead of only lazily clamping
        picked domains): failed-literal units (run-scoped forced phases,
        1-literal cuts) and -- once per run -- the persistent implication
        edges (box-sound 2-literal clause cuts). Wire form of a clause
        OR_i phase(p_i): coefficient s_i = -p_i, sum s_i z_i <= npos - 1
        (the add_blocking_cuts mapping, inverted)."""
        if not self.mirror or self.fact_cuts_max <= 0:
            return []
        # Safety valve: every installed cut is a per-domain-optimized
        # general-beta constraint, so fact cuts respect a number_cuts-style
        # per-run budget. Units go first (a forced phase is the strongest
        # clause cut); edges fill whatever budget remains.
        budget = self.fact_cuts_max - self._fact_cuts_emitted
        if budget <= 0:
            return []
        lits_lists = [[l] for l in self._new_unit_lits[:budget]]
        self._new_unit_lits = self._new_unit_lits[len(lits_lists):]
        if len(lits_lists) < budget:
            # Binary clauses as 2-literal cuts. BOTH sources:
            #   persistent -- root-probe implication edges, box-sound, exported
            #                 once per run (they never change within a run);
            #   run_clauses -- edges harvested by conditioned re-probing, which
            #                 accumulate as the run progresses, so they are
            #                 exported incrementally (a cursor, not a flag).
            # Feeding the CUTTER is the point: a clause sitting in the boolean
            # DB only filters picked domains, and that was measured inert
            # (15,708 re-probe edges -> 0 change in domains visited). As a
            # GCP-CROWN cut the same clause enters every subproblem's
            # relaxation with an optimized multiplier, exactly like a BICCOS
            # blocking clause.
            edges = []
            if not self._edges_exported:
                self._edges_exported = True
                edges += [c for c in self.persistent if len(c) == 2]
            new_run_edges = self.run_clauses[self._run_edge_cursor:]
            self._run_edge_cursor = len(self.run_clauses)
            # STRONGEST FIRST. Re-probing can harvest thousands of edges per
            # pass; the budget takes tens. Insertion order is meaningless, so
            # rank by the harvest-time strength score (how much the source pin
            # actually moved the box) and spend the budget on the top of that
            # list.
            ranked = sorted(
                (c for c in new_run_edges if len(c) == 2),
                key=lambda c: -self._clause_score.get(frozenset(c), 0.0))
            edges += ranked
            lits_lists += edges[:budget - len(lits_lists)]
        if not lits_lists:
            return []
        self._fact_cuts_emitted += len(lits_lists)
        name_of_var = {v: k for k, v in self._var_of.items()}
        ridx_of_name = {v: k for k, v in self.preact_of_relu_idx.items()}
        out = []
        for lits in lits_lists:
            decision, coeffs, ok = [], [], True
            for lit in lits:
                name, nidx = name_of_var[abs(lit)]
                ridx = ridx_of_name.get(name)
                if ridx is None:
                    ok = False
                    break
                decision.append([ridx, int(nidx)])
                coeffs.append(-1.0 if lit > 0 else 1.0)
            if not ok:
                continue
            npos = sum(1 for c in coeffs if c > 0)
            out.append({
                'x_decision': [], 'x_coeffs': [],
                'relu_decision': [], 'relu_coeffs': [],
                'arelu_decision': decision, 'arelu_coeffs': coeffs,
                'pre_decision': [], 'pre_coeffs': [],
                'bias': float(npos - 1), 'c': -1,
            })
        return out

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

    def process_picked_domains(self, d, run_key, prune=True):
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
        if self.run_unsat and run_key is not None and prune:
            # Mirror UNSAT certificate for this run: every domain of the
            # OR group is counterexample-free (see _mark_run_unsat).
            n = len(histories)
            self.stats['domains_checked'] += n
            self.stats['domains_pruned'] += n
            self.stats['time'] += time.time() - t0
            if not self._run_unsat_reported:
                self._run_unsat_reported = True
                print(f'Phase probing SAT layer: UNSAT certificate prunes '
                      f'all {n} picked domains of this run.')
            return [], n
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
            if not lits:
                # No phase splits on this domain's path (the root domain, and
                # every domain until the first split). The empty assumption set
                # is trivially satisfiable, so there is nothing to refute --
                # but PySAT's propagate() reports ok=False for empty assumptions
                # on a clause-free solver:
                #     Cadical195().propagate(assumptions=[])  -> (False, [])
                #     Cadical195().propagate(assumptions=[1]) -> (True, [1])
                # Reading that as "refuted" prunes the root, empties the picked
                # batch, and leaves multi-tree BaB with no tree to restore
                # (assert best_node is not None, branching_domains.py). Only
                # bites when probing armed the DB with no facts, which is the
                # common case on instances with no edges/forced phases.
                keep.append(i)
                continue
            ok, implied = solver.propagate(assumptions=lits)
            if not ok:
                if prune:
                    pruned += 1
                    continue
                # prune=False (multi-tree call sites): the domain is refuted,
                # but emptying a batch there breaks MTS's tree restore
                # (restore_best_domains -> _generate_tree asserts on an empty
                # candidate set). Keep it and let the clamps below tighten it;
                # BaB will reach the same conclusion from the bounds.
                keep.append(i)
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
