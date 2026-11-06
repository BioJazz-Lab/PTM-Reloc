"""Optimize the reassignment weights (w_H, w_L, w_D) against the GS/FUN composite score.

Strategy
--------
1. **Monte Carlo search** (as in the manuscript): weights sampled log-uniformly in
   ``10**[lo, hi]``; every trial is scored by the composite Δ_REASSIGN on the
   peptides that would be reassigned (AScore <= threshold).
2. **Refinement** (optional): differential evolution in log10 space, seeded with the
   best Monte Carlo trials. The objective is piecewise constant (an arg-max), so
   a derivative-free population method is used.
3. **Robust estimate**: per-weight median of the top ``top_frac`` Monte Carlo trials
   (the manuscript's "median of the best 5%"), which is less sensitive to the
   plateau-y objective than the single best trial.

Optionally a fraction of proteins is held out, so the chosen weights can be
checked on peptides that were not used for fitting.

Note: S_F is invariant to multiplying all weights by a constant, so only the
*ratios* of the weights matter; ``normalized`` weights (summing to 1) are reported.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import as_weight_vector, choose
from .features import WEIGHT_NAMES, CandidateTable, FeatureConfig
from .metrics import evaluate, evaluate_by_ascore, trial_relative_composite

__version__ = "0.1.0"


def _wdict(w) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(WEIGHT_NAMES, as_weight_vector(w))}


def normalize_weights(w) -> Dict[str, float]:
    v = as_weight_vector(w)
    s = v.sum()
    return _wdict(v / s if s > 0 else v)


@dataclass
class OptimizationResult:
    weights: Dict[str, float]  # recommended (robust) weights
    best_weights: Dict[str, float]  # single best trial
    threshold: float
    config: FeatureConfig
    gs_weight: float
    trials: pd.DataFrame
    evaluation: pd.DataFrame  # candidate weight sets × split
    by_ascore: pd.DataFrame  # recommended weights, metrics per AScore bin (all rows)
    settings: Dict = field(default_factory=dict)

    def summary(self) -> str:
        cols = ["weights", "split", "n_rows", "n_rows_changed", "delta_gs", "random_delta_gs",
                "oracle_delta_gs", "delta_fun", "random_delta_fun", "oracle_delta_fun",
                "gs_gain", "fun_gain", "composite"]
        ev = self.evaluation[[c for c in cols if c in self.evaluation.columns]]
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
            return (f"Recommended weights: {self.weights}\n"
                    f"  normalized:        {normalize_weights(self.weights)}\n"
                    f"Best single trial:   {self.best_weights}\n"
                    f"AScore threshold:    {self.threshold}\n\n{ev.to_string(index=False)}")

    def weights_payload(self) -> Dict:
        ev = self.evaluation
        rec = ev[ev["weights"] == "recommended"].set_index("split")
        return {
            "ptmreloc_version": __version__,
            "weights": self.weights,
            "normalized_weights": normalize_weights(self.weights),
            "best_trial_weights": self.best_weights,
            "threshold": self.threshold,
            "feature_config": self.config.to_dict(),
            "gs_weight": self.gs_weight,
            "metrics": {s: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                            for k, v in r.items()} for s, r in rec.to_dict(orient="index").items()},
            "settings": self.settings,
        }

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "weights.json"), "w") as fh:
            json.dump(self.weights_payload(), fh, indent=2)
        self.trials.to_csv(os.path.join(outdir, "trials.csv"), index=False)
        self.evaluation.to_csv(os.path.join(outdir, "evaluation.csv"), index=False)
        self.by_ascore.to_csv(os.path.join(outdir, "metrics_by_ascore.csv"), index=False)


def split_by_protein(table: CandidateTable, holdout_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean row mask of held-out rows (whole proteins are held out)."""
    if holdout_frac <= 0:
        return np.zeros(table.n_rows, dtype=bool)
    prots = np.unique(table.r_uniprot)
    n_test = int(round(holdout_frac * len(prots)))
    test = set(rng.choice(prots, size=n_test, replace=False).tolist())
    return np.fromiter((u in test for u in table.r_uniprot), dtype=bool, count=table.n_rows)


def optimize_weights(
    table: CandidateTable,
    threshold: float = 30.0,
    n_samples: int = 2000,
    log10_bounds: Tuple[float, float] = (-3.0, 0.0),
    refine: bool = True,
    refine_maxiter: int = 30,
    top_frac: float = 0.05,
    holdout_frac: float = 0.2,
    gs_weight: float = 0.5,
    seed: Optional[int] = 0,
    verbose: bool = False,
) -> OptimizationResult:
    """Search for the weights maximizing the composite GS/FUN score.

    Parameters
    ----------
    table
        Output of :func:`ptmreloc.build_candidate_table` built with a reference
        that includes Gold Standard and/or Functional Landscape sites.
    threshold
        Only peptides with AScore <= threshold are relocalized (and scored).
    n_samples
        Number of Monte Carlo trials.
    log10_bounds
        Sampling range of each weight in log10 space (manuscript: [-3, 0]).
    refine
        Run differential evolution seeded from the best trials.
    top_frac
        Fraction of best Monte Carlo trials whose median gives the recommended weights.
    holdout_frac
        Fraction of proteins held out for validation (0 disables).
    gs_weight
        Weight of the GS gain in the composite (geometric mean); 0.5 = equal.
    """
    if not (table.c_gs.any() or table.c_fun.any()):
        raise ValueError("no Gold Standard / Functional sites among the candidates; "
                         "build the table with a ReferenceDB that includes them")
    rng = np.random.default_rng(seed)
    lo, hi = log10_bounds
    eligible = table.r_ascore <= threshold
    test_rows = split_by_protein(table, holdout_frac, rng)
    train_mask = eligible & ~test_rows
    test_mask = eligible & test_rows

    records = []

    def run(logw: np.ndarray, source: str) -> float:
        w = 10.0 ** np.asarray(logw, dtype=float)
        m = evaluate(table, choose(table, w), train_mask, gs_weight)
        records.append({"source": source, **_wdict(w), **m})
        return m["composite"]

    # anchors: HTP only, equal weights
    run(np.array([0.0, -np.inf, -np.inf]), "anchor_htp_only")
    run(np.zeros(3), "anchor_equal")
    samples = rng.uniform(lo, hi, size=(n_samples, len(WEIGHT_NAMES)))
    for i, logw in enumerate(samples):
        run(logw, "monte_carlo")
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            best = max(r["composite"] for r in records)
            print(f"  monte carlo {i + 1}/{n_samples}  best composite={best:.4f}", flush=True)

    mc = pd.DataFrame([r for r in records if r["source"] == "monte_carlo"])

    if refine and n_samples > 0:
        from scipy.optimize import differential_evolution

        n_init = min(len(mc), max(5, 5 * len(WEIGHT_NAMES)))
        init = np.log10(mc.nlargest(n_init, "composite")[list(WEIGHT_NAMES)].to_numpy())
        res = differential_evolution(
            lambda x: -run(x, "refine"), bounds=[(lo, hi)] * len(WEIGHT_NAMES), init=init,
            maxiter=refine_maxiter, polish=False, seed=int(rng.integers(2**31 - 1)), tol=1e-8,
        )
        if verbose:
            print(f"  refinement: {res.nfev} evaluations, best composite={-res.fun:.4f}", flush=True)

    trials = pd.DataFrame(records)
    trials.insert(0, "trial", np.arange(len(trials)))
    trials["composite_trial_relative"] = trial_relative_composite(trials)

    best_row = trials.loc[trials["composite"].idxmax()]
    best_w = _wdict(best_row[list(WEIGHT_NAMES)].to_numpy(dtype=float))
    if len(mc):
        k = max(1, int(math.ceil(top_frac * len(mc))))
        top = mc.nlargest(k, "composite")
        robust_w = _wdict(10.0 ** np.median(np.log10(top[list(WEIGHT_NAMES)].to_numpy()), axis=0))
    else:
        robust_w = best_w

    candidates = {"recommended": robust_w, "best_trial": best_w, "htp_only": {"w_H": 1.0, "w_L": 0.0, "w_D": 0.0}}
    splits = {"train": train_mask}
    if test_mask.any():
        splits["holdout"] = test_mask
    splits["all"] = eligible
    ev = []
    for name, w in candidates.items():
        sel = choose(table, w)
        for split, mask in splits.items():
            ev.append({"weights": name, "split": split, **w, **evaluate(table, sel, mask, gs_weight)})
    evaluation = pd.DataFrame(ev)
    by_ascore = evaluate_by_ascore(table, choose(table, robust_w), gs_weight=gs_weight)

    return OptimizationResult(
        weights=robust_w, best_weights=best_w, threshold=float(threshold), config=table.config,
        gs_weight=gs_weight, trials=trials, evaluation=evaluation, by_ascore=by_ascore,
        settings={"n_samples": n_samples, "log10_bounds": list(log10_bounds), "refine": refine,
                  "refine_maxiter": refine_maxiter, "top_frac": top_frac, "holdout_frac": holdout_frac,
                  "seed": seed},
    )
