"""Build the candidate-site table: one row per (peptide, candidate residue) with its features.

The table is computed once per dataset. Reassignment with any weight vector is
then a vectorized arg-max/top-n over groups (see :mod:`ptmreloc.engine`), which
is what makes large-scale weight optimization cheap.

Features per candidate site (each normalized within its peptide):

* ``S_H`` – PSP high-throughput (MS) evidence
* ``S_L`` – PSP low-throughput (literature) evidence
* ``S_D`` – proximity to the originally assigned site(s)

Evaluation targets per candidate site (never used for reassignment itself):

* ``gs``  – 1 if the site is a Gold Standard site
* ``fun`` – Functional Landscape score (0 if absent)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .peptide import parse_peptide, parse_residue
from .reference import ReferenceDB

FEATURE_NAMES = ("S_H", "S_L", "S_D")
WEIGHT_NAMES = ("w_H", "w_L", "w_D")

DEFAULT_COLUMNS = {"uniprot": "UniProt", "peptide": "Peptide", "residue": "Residue", "ascore": "AScore"}


@dataclass(frozen=True)
class FeatureConfig:
    """How candidate features are computed.

    evidence_transform
        ``"legacy"`` – ``log10(log2(n + 1) + 1)`` as in the original sampling scripts (default);
        ``"log"`` – ``log(n + 1)``; ``"raw"`` – the counts themselves (manuscript text).
        The transformed values are then divided by their sum over the peptide's candidates.
    distance_mode
        ``"legacy"`` – ``1 - d / (d_max + 1)`` normalized to sum 1 within the peptide (original scripts);
        ``"inverse"`` – ``1 / (d + 1)`` (max-normalized), set to 0 when AScore == 0 (manuscript text).
        ``d`` is the residue distance to the nearest originally assigned site.
    """

    ptm_label: str = "*"
    candidate_residues: str = "STY"
    evidence_transform: str = "legacy"
    distance_mode: str = "legacy"

    def __post_init__(self):
        if self.evidence_transform not in ("legacy", "log", "raw"):
            raise ValueError(f"unknown evidence_transform {self.evidence_transform!r}")
        if self.distance_mode not in ("legacy", "inverse"):
            raise ValueError(f"unknown distance_mode {self.distance_mode!r}")

    def to_dict(self) -> Dict:
        return asdict(self)


def _transform(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "legacy":
        return np.log10(np.log2(x + 1.0) + 1.0)
    if mode == "log":
        return np.log1p(x)
    return x.astype(float)


@dataclass
class CandidateTable:
    """Flat, vectorized representation of every candidate site of every peptide.

    Indices: ``g`` = unique peptide group (all groups), ``v`` = valid group
    (compact index), ``c`` = candidate site, ``r`` = input row.
    Candidates of a valid group are contiguous, starting at ``v_start[v]``.
    """

    config: FeatureConfig
    # unique peptide groups
    g_uniprot: np.ndarray
    g_peptide: np.ndarray
    g_residue: np.ndarray
    g_valid: np.ndarray
    g_v: np.ndarray  # valid index of each group or -1
    # valid groups
    v_g: np.ndarray
    v_start: np.ndarray
    v_ncand: np.ndarray
    v_nsites: np.ndarray
    # candidates
    c_v: np.ndarray
    c_pep_idx: np.ndarray
    c_pos: np.ndarray
    c_aa: np.ndarray
    c_orig: np.ndarray
    X: np.ndarray  # (C, 3) features S_H, S_L, S_D
    c_gs: np.ndarray
    c_fun: np.ndarray
    # rows
    r_g: np.ndarray
    r_v: np.ndarray
    r_ascore: np.ndarray
    r_uniprot: np.ndarray
    # per valid group reference quantities for the targets
    v_old_gs: np.ndarray = None
    v_old_fun: np.ndarray = None
    v_oracle_gs: np.ndarray = None
    v_oracle_fun: np.ndarray = None
    v_rand_gs: np.ndarray = None
    v_rand_fun: np.ndarray = None

    @property
    def n_rows(self) -> int:
        return len(self.r_g)

    @property
    def n_candidates(self) -> int:
        return len(self.c_v)

    def candidate_frame(self) -> pd.DataFrame:
        """Candidate table as a DataFrame (for inspection / debugging)."""
        g = self.v_g[self.c_v]
        return pd.DataFrame({
            "UniProt": self.g_uniprot[g], "Peptide": self.g_peptide[g], "Residue": self.g_residue[g],
            "Site": [f"{a}{p}" for a, p in zip(self.c_aa, self.c_pos)], "IsOriginal": self.c_orig,
            "S_H": self.X[:, 0], "S_L": self.X[:, 1], "S_D": self.X[:, 2],
            "GoldStandard": self.c_gs.astype(bool), "FunScore": self.c_fun,
        })


def _group_sum(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(groups, weights=values, minlength=n)


def topn_sum(values: np.ndarray, c_v: np.ndarray, v_nsites: np.ndarray) -> np.ndarray:
    """Per group: sum of the ``v_nsites`` largest candidate values."""
    order = np.lexsort((-values, c_v))
    gv = c_v[order]
    rank = np.arange(len(gv)) - np.searchsorted(gv, gv, side="left")
    take = rank < v_nsites[gv]
    return _group_sum(np.where(take, values[order], 0.0), gv, len(v_nsites))


def build_candidate_table(
    df: pd.DataFrame,
    reference: ReferenceDB,
    config: FeatureConfig = FeatureConfig(),
    columns: Optional[Mapping[str, str]] = None,
) -> CandidateTable:
    """Compute candidate sites and features for every row of ``df``.

    ``df`` needs UniProt accession, modified peptide, residue annotation and
    AScore columns (names configurable via ``columns``). Rows that cannot be
    parsed, or whose residue annotation disagrees with the peptide, are marked
    invalid and are always passed through unchanged.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise KeyError(f"input is missing columns: {missing}")

    ascore = pd.to_numeric(df[cols["ascore"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    key = pd.DataFrame({
        "u": df[cols["uniprot"]].astype(str).to_numpy(),
        "p": df[cols["peptide"]].astype(str).to_numpy(),
        "r": df[cols["residue"]].astype(str).to_numpy(),
        "z": ascore == 0,
    })
    r_g = key.groupby(["u", "p", "r", "z"], sort=False).ngroup().to_numpy()
    uniq = key.drop_duplicates()
    G = len(uniq)

    label, cand_res, = config.ptm_label, set(config.candidate_residues)
    htp, ltp, gold, fun = reference.htp, reference.ltp, reference.gold, reference.fun

    g_valid = np.zeros(G, dtype=bool)
    v_g, v_nsites, v_ncand, v_zero = [], [], [], []
    c_pep_idx, c_pos, c_aa, c_orig, c_dist = [], [], [], [], []
    c_htp, c_ltp, c_gs, c_fun = [], [], [], []

    for g, (u, p, r, z) in enumerate(uniq.itertuples(index=False, name=None)):
        try:
            pp = parse_peptide(p, label)
            sites = parse_residue(r)
        except ValueError:
            continue
        lab = pp.labeled
        if not lab or len(sites) != len(lab):
            continue
        offsets = {pos - i for (aa, pos), i in zip(sorted(sites, key=lambda s: s[1]), lab)}
        if len(offsets) != 1 or any(pp.aas[i] != aa for (aa, _), i in zip(sorted(sites, key=lambda s: s[1]), lab)):
            continue
        off = offsets.pop()
        cands = sorted({i for i, a in enumerate(pp.aas) if a in cand_res} | set(lab))
        lab_arr = np.asarray(lab)
        g_valid[g] = True
        v_g.append(g)
        v_nsites.append(len(lab))
        v_ncand.append(len(cands))
        v_zero.append(z)
        for i in cands:
            k = (u, off + i)
            c_pep_idx.append(i)
            c_pos.append(off + i)
            c_aa.append(pp.aas[i])
            c_orig.append(i in lab)
            c_dist.append(int(np.abs(lab_arr - i).min()))
            c_htp.append(htp.get(k, 0.0))
            c_ltp.append(ltp.get(k, 0.0))
            c_gs.append(k in gold)
            c_fun.append(fun.get(k, 0.0))

    V = len(v_g)
    v_g = np.asarray(v_g, dtype=np.int64)
    v_ncand = np.asarray(v_ncand, dtype=np.int64)
    v_nsites = np.asarray(v_nsites, dtype=np.int64)
    v_zero = np.asarray(v_zero, dtype=bool)
    v_start = np.concatenate([[0], np.cumsum(v_ncand)[:-1]]).astype(np.int64) if V else np.zeros(0, np.int64)
    c_v = np.repeat(np.arange(V), v_ncand)
    g_v = np.full(G, -1, dtype=np.int64)
    g_v[v_g] = np.arange(V)

    c_dist = np.asarray(c_dist, dtype=float)
    c_orig = np.asarray(c_orig, dtype=bool)
    X = np.zeros((len(c_v), 3))

    def normalize_sum(x):
        s = _group_sum(x, c_v, V)[c_v]
        return np.divide(x, s, out=np.zeros_like(x), where=s > 0)

    X[:, 0] = normalize_sum(_transform(np.asarray(c_htp, dtype=float), config.evidence_transform))
    X[:, 1] = normalize_sum(_transform(np.asarray(c_ltp, dtype=float), config.evidence_transform))
    if V:
        if config.distance_mode == "legacy":
            dmax = np.maximum.reduceat(c_dist, v_start)[c_v]
            X[:, 2] = normalize_sum(1.0 - c_dist / (dmax + 1.0))
        else:
            X[:, 2] = np.where(v_zero[c_v], 0.0, 1.0 / (c_dist + 1.0))

    c_gs = np.asarray(c_gs, dtype=float)
    c_fun = np.asarray(c_fun, dtype=float)

    table = CandidateTable(
        config=config,
        g_uniprot=uniq["u"].to_numpy(), g_peptide=uniq["p"].to_numpy(), g_residue=uniq["r"].to_numpy(),
        g_valid=g_valid, g_v=g_v,
        v_g=v_g, v_start=v_start, v_ncand=v_ncand, v_nsites=v_nsites,
        c_v=c_v, c_pep_idx=np.asarray(c_pep_idx, dtype=np.int64), c_pos=np.asarray(c_pos, dtype=np.int64),
        c_aa=np.asarray(c_aa, dtype=object), c_orig=c_orig, X=X, c_gs=c_gs, c_fun=c_fun,
        r_g=r_g, r_v=g_v[r_g], r_ascore=ascore, r_uniprot=key["u"].to_numpy(),
    )
    # reference quantities for evaluation (independent of weights)
    table.v_old_gs = _group_sum(c_gs * c_orig, c_v, V)
    table.v_old_fun = _group_sum(c_fun * c_orig, c_v, V)
    table.v_oracle_gs = topn_sum(c_gs, c_v, v_nsites) if V else np.zeros(0)
    table.v_oracle_fun = topn_sum(c_fun, c_v, v_nsites) if V else np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        table.v_rand_gs = v_nsites * _group_sum(c_gs, c_v, V) / np.maximum(v_ncand, 1)
        table.v_rand_fun = v_nsites * _group_sum(c_fun, c_v, V) / np.maximum(v_ncand, 1)
    return table
