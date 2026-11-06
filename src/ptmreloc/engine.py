"""Vectorized site selection: S_F = w_H*S_H + w_L*S_L + w_D*S_D, pick the top-n candidates per peptide.

Ties are broken in favor of the originally assigned site(s), so a peptide is
only relocalized when another site scores strictly higher (this also leaves
peptides without any PSP evidence untouched).
"""
from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

from .features import WEIGHT_NAMES, CandidateTable

WeightsLike = Union[Mapping[str, float], Sequence[float], np.ndarray]


def as_weight_vector(weights: WeightsLike) -> np.ndarray:
    if isinstance(weights, Mapping):
        w = np.array([float(weights.get(k, 0.0)) for k in WEIGHT_NAMES])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != (len(WEIGHT_NAMES),):
            raise ValueError(f"expected {len(WEIGHT_NAMES)} weights {WEIGHT_NAMES}, got {w.shape}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be finite and non-negative: {w}")
    return w


def site_scores(table: CandidateTable, weights: WeightsLike) -> np.ndarray:
    """Final reassignment score S_F for every candidate."""
    return table.X @ as_weight_vector(weights)


def choose(table: CandidateTable, weights: WeightsLike, rel_tol: float = 1e-9) -> np.ndarray:
    """Boolean mask over candidates: the sites selected for each peptide."""
    s = site_scores(table, weights)
    C = len(s)
    chosen = np.zeros(C, dtype=bool)
    if C == 0:
        return chosen
    c_v, starts, orig = table.c_v, table.v_start, table.c_orig
    mono_v = table.v_nsites == 1

    # --- mono-site peptides: arg-max with ties -> original site
    gmax = np.maximum.reduceat(s, starts)
    tol = rel_tol * np.maximum(1.0, np.abs(gmax))
    is_max = s >= (gmax - tol)[c_v]
    orig_is_max = np.bincount(c_v, weights=(is_max & orig), minlength=len(starts)) > 0
    first_max = np.minimum.reduceat(np.where(is_max, np.arange(C), C), starts)
    keep = mono_v & orig_is_max
    move = mono_v & ~orig_is_max
    chosen |= keep[c_v] & orig
    chosen[first_max[move]] = True

    # --- multi-site peptides: top-n with ties -> original sites, then sequence order
    if not mono_v.all():
        idx = np.flatnonzero(~mono_v[c_v])
        sv = s[idx]
        scale = np.maximum(1.0, np.abs(sv).max())
        q = np.round(sv / (rel_tol * scale))
        order = np.lexsort((table.c_pep_idx[idx], ~orig[idx], -q, c_v[idx]))
        gv = c_v[idx][order]
        rank = np.arange(len(gv)) - np.searchsorted(gv, gv, side="left")
        chosen[idx[order]] = rank < table.v_nsites[gv]
    return chosen


def selection_probability(table: CandidateTable, weights: WeightsLike, chosen: np.ndarray):
    """Share of the peptide's total S_F carried by the chosen and by the original sites.

    Returns per-valid-group arrays ``(p_chosen, p_original)``; uniform 1/n shares
    are used for peptides whose scores are all zero.
    """
    s = site_scores(table, weights)
    V = len(table.v_start)
    tot = np.bincount(table.c_v, weights=s, minlength=V)
    share = np.where(tot[table.c_v] > 0, s / np.where(tot > 0, tot, 1.0)[table.c_v],
                     1.0 / table.v_ncand[table.c_v])
    p_new = np.bincount(table.c_v, weights=share * chosen, minlength=V)
    p_old = np.bincount(table.c_v, weights=share * table.c_orig, minlength=V)
    return p_new, p_old, share
