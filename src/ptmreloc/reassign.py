"""Apply weights to relocalize low-confidence PTM sites in a peptide table."""
from __future__ import annotations

import json
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .defaults import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS
from .engine import WeightsLike, as_weight_vector, choose, selection_probability
from .features import DEFAULT_COLUMNS, WEIGHT_NAMES, CandidateTable, FeatureConfig, build_candidate_table
from .peptide import format_residue, parse_peptide, relabel
from .reference import ReferenceDB


def load_weights(path: str) -> Dict:
    """Read a ``weights.json`` written by :meth:`OptimizationResult.save`."""
    with open(path) as fh:
        payload = json.load(fh)
    if "weights" not in payload:
        raise ValueError(f"{path} has no 'weights' entry")
    return payload


def apply_selection(
    df: pd.DataFrame,
    table: CandidateTable,
    weights: WeightsLike,
    threshold: float,
    chosen: Optional[np.ndarray] = None,
    columns: Optional[Mapping[str, str]] = None,
    include_candidates: bool = True,
) -> pd.DataFrame:
    """Return ``df`` with relocalization columns appended (see :meth:`Reassigner.reassign`)."""
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    if chosen is None:
        chosen = choose(table, weights)
    label = table.config.ptm_label
    V = len(table.v_start)
    p_new, p_old, share = selection_probability(table, weights, chosen)

    changed_v = np.bincount(table.c_v, weights=chosen & ~table.c_orig, minlength=V) > 0
    new_pep_v = table.g_peptide[table.v_g].copy()
    new_res_v = table.g_residue[table.v_g].copy()
    for v in np.flatnonzero(changed_v):
        sl = slice(table.v_start[v], table.v_start[v] + table.v_ncand[v])
        sel = chosen[sl]
        pp = parse_peptide(table.g_peptide[table.v_g[v]], label)
        new_pep_v[v] = relabel(pp, table.c_pep_idx[sl][sel], label)
        new_res_v[v] = format_residue(list(zip(table.c_aa[sl][sel], table.c_pos[sl][sel])))

    cand_v = None
    if include_candidates and V:
        fmt = [f"{a}{p}:{x:.3f}" for a, p, x in zip(table.c_aa, table.c_pos, share)]
        cand_v = np.array([";".join(fmt[s:s + n]) for s, n in zip(table.v_start, table.v_ncand)], dtype=object)

    out = df.copy()
    rv = table.r_v
    ok = rv >= 0
    low = table.r_ascore <= threshold
    apply = ok & low
    rvi = np.where(ok, rv, 0)

    pep = df[cols["peptide"]].astype(str).to_numpy(dtype=object).copy()
    res = df[cols["residue"]].astype(str).to_numpy(dtype=object).copy()
    if V:
        pep[apply] = new_pep_v[rvi[apply]]
        res[apply] = new_res_v[rvi[apply]]
    out["NewPeptide"] = pep
    out["NewResidue"] = res
    out["Reassigned"] = apply & (changed_v[rvi] if V else False)
    status = np.where(~ok, "unparsed", np.where(low, "eligible", "above_threshold"))
    out["ReassignStatus"] = status
    out["OriginalSiteProb"] = np.where(ok, p_old[rvi] if V else np.nan, np.nan)
    out["NewSiteProb"] = np.where(apply, p_new[rvi] if V else np.nan, out["OriginalSiteProb"])
    if cand_v is not None:
        out["CandidateSiteProbs"] = np.where(ok, cand_v[rvi], "")
    return out


class Reassigner:
    """Relocalize PTM sites on peptides with AScore <= ``threshold`` using PSP priors.

    Example
    -------
    >>> ref = ptmreloc.load_reference(gold=False, fun=False)  # default PSP, or psp="my_psp.txt"
    >>> r = Reassigner(ref)  # default weights (DEFAULT_WEIGHTS), threshold 30
    >>> out = r.reassign(peptides_df)
    """

    def __init__(
        self,
        reference: ReferenceDB,
        weights: Optional[WeightsLike] = None,
        threshold: float = DEFAULT_THRESHOLD,
        config: FeatureConfig = FeatureConfig(),
        columns: Optional[Mapping[str, str]] = None,
    ):
        self.reference = reference
        if weights is None:
            weights = DEFAULT_WEIGHTS
        self.weights = {k: float(v) for k, v in zip(WEIGHT_NAMES, as_weight_vector(weights))}
        self.threshold = float(threshold)
        self.config = config
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}

    @classmethod
    def from_weights_file(cls, path: str, reference: ReferenceDB, threshold: Optional[float] = None,
                          columns: Optional[Mapping[str, str]] = None) -> "Reassigner":
        payload = load_weights(path)
        config = FeatureConfig(**payload.get("feature_config", {}))
        thr = payload.get("threshold", 30.0) if threshold is None else threshold
        return cls(reference, payload["weights"], thr, config, columns)

    def candidate_table(self, df: pd.DataFrame) -> CandidateTable:
        return build_candidate_table(df, self.reference, self.config, self.columns)

    def reassign(self, df: pd.DataFrame, include_candidates: bool = True) -> pd.DataFrame:
        """Return ``df`` with added columns:

        * ``NewPeptide`` / ``NewResidue`` – relocalized peptide and site annotation
          (unchanged for AScore > threshold or unparsable rows)
        * ``Reassigned`` – whether the site(s) moved
        * ``ReassignStatus`` – ``eligible`` / ``above_threshold`` / ``unparsed``
        * ``OriginalSiteProb`` / ``NewSiteProb`` – share of the peptide's S_F
          carried by the original / chosen site(s)
        * ``CandidateSiteProbs`` – every candidate site with its S_F share
        """
        table = self.candidate_table(df)
        return apply_selection(df, table, self.weights, self.threshold, columns=self.columns,
                               include_candidates=include_candidates)
