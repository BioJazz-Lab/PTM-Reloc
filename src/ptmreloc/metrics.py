"""Reassignment quality metrics based on Gold Standard (GS) and Functional Landscape (FUN) sites.

For a set of input rows (PSMs/peptides), with ``n_sites`` assigned sites in total:

* ``delta_gs``  – (#GS sites after − #GS sites before) / n_sites          (Δ_GS)
* ``delta_fun`` – (Σ functional score after − before) / n_sites           (Δ_FUN)
* ``oracle_delta_*`` – the best achievable Δ (always choosing the best candidate)
* ``random_delta_*`` – the expected Δ of uniformly random reassignment
* ``gs_gain`` / ``fun_gain`` – Δ / oracle Δ: the fraction of the achievable
  improvement realized. These are the *unit-normalized* metrics; they do not
  depend on which other trials were run, so they can be optimized directly.
* ``composite`` – Δ_REASSIGN: weighted geometric mean of ``gs_gain`` and
  ``fun_gain`` when both are positive, otherwise the smaller of the two (so a
  loss in either metric is never hidden by a gain in the other).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .features import CandidateTable


def composite_score(gs_gain: float, fun_gain: float, gs_weight: float = 0.5) -> float:
    if gs_gain > 0 and fun_gain > 0:
        return float(gs_gain ** gs_weight * fun_gain ** (1.0 - gs_weight))
    return float(min(gs_gain, fun_gain))


def evaluate(
    table: CandidateTable,
    chosen: np.ndarray,
    row_mask: Optional[np.ndarray] = None,
    gs_weight: float = 0.5,
) -> Dict[str, float]:
    """Metrics of a selection (``chosen`` from :func:`ptmreloc.engine.choose`) on the given rows."""
    V = len(table.v_start)
    rows = np.ones(table.n_rows, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    valid_rows = rows & (table.r_v >= 0)
    w = np.bincount(table.r_v[valid_rows], minlength=V).astype(float)

    new_gs = np.bincount(table.c_v, weights=table.c_gs * chosen, minlength=V)
    new_fun = np.bincount(table.c_v, weights=table.c_fun * chosen, minlength=V)
    changed = np.bincount(table.c_v, weights=chosen & ~table.c_orig, minlength=V) > 0

    n_sites = float(w @ table.v_nsites)
    denom = n_sites if n_sites > 0 else np.nan

    def d(new, old):
        return float(w @ (new - old)) / denom

    out = {
        "n_rows": int(rows.sum()),
        "n_rows_unparsed": int((rows & (table.r_v < 0)).sum()),
        "n_sites": int(n_sites),
        "n_rows_changed": int(w @ changed),
        "delta_gs": d(new_gs, table.v_old_gs),
        "delta_fun": d(new_fun, table.v_old_fun),
        "random_delta_gs": d(table.v_rand_gs, table.v_old_gs),
        "random_delta_fun": d(table.v_rand_fun, table.v_old_fun),
        "oracle_delta_gs": d(table.v_oracle_gs, table.v_old_gs),
        "oracle_delta_fun": d(table.v_oracle_fun, table.v_old_fun),
    }
    out["frac_rows_changed"] = out["n_rows_changed"] / max(1, out["n_rows"] - out["n_rows_unparsed"])
    out["gs_gain"] = out["delta_gs"] / out["oracle_delta_gs"] if out["oracle_delta_gs"] > 0 else 0.0
    out["fun_gain"] = out["delta_fun"] / out["oracle_delta_fun"] if out["oracle_delta_fun"] > 0 else 0.0
    out["composite"] = composite_score(out["gs_gain"], out["fun_gain"], gs_weight)
    return out


DEFAULT_ASCORE_BINS = (-np.inf, 0, 10, 20, 30, 50, 100, np.inf)


def evaluate_by_ascore(
    table: CandidateTable,
    chosen: np.ndarray,
    bins: Sequence[float] = DEFAULT_ASCORE_BINS,
    row_mask: Optional[np.ndarray] = None,
    gs_weight: float = 0.5,
) -> pd.DataFrame:
    """Metrics per AScore bin ``(lo, hi]`` — the selection is applied to every bin
    regardless of threshold, to show where relocalization helps or hurts."""
    rows = np.ones(table.n_rows, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    recs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = rows & (table.r_ascore > lo) & (table.r_ascore <= hi)
        if not m.any():
            continue
        label = f"({lo:g}, {hi:g}]" if np.isfinite(lo) else f"<= {hi:g}"
        recs.append({"ascore_bin": label, **evaluate(table, chosen, m, gs_weight)})
    return pd.DataFrame(recs)


def scale_to_unit(x: np.ndarray) -> np.ndarray:
    """Min–max scaling across trials, as in the original analysis (``scale2unit``)."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmin(x) + 1e-4) / (np.nanmax(x) - np.nanmin(x) + 1e-4)


def trial_relative_composite(trials: pd.DataFrame) -> np.ndarray:
    """Manuscript-style Δ_REASSIGN: sqrt(scaled Δ_GS × scaled Δ_FUN), scaled across trials."""
    return np.sqrt(scale_to_unit(trials["delta_gs"]) * scale_to_unit(trials["delta_fun"]))
