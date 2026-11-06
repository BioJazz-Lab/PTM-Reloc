"""High-level library API: one call each for optimize / reassign / evaluate.

Reference data defaults to the human reference database (PhosphoSitePlus,
Gold Standard, Functional Landscape, UniProt FASTA), downloaded once on first use. To use your own data, pass
DataFrames or file paths (pkl/csv/tsv/parquet/xlsx) as ``psp=``, ``gold=``, ``fun=``,
``fasta=``. Anything left as ``None`` falls back to the default file; ``False``
disables an optional table. A prebuilt :class:`ReferenceDB` can be passed as ``psp``
(its own GS/FUN sites are used unless ``gold``/``fun`` are given explicitly).

If the peptide table has no residue column, residues are annotated from the
protein sequences (``fasta=``, the default human proteome).

Weights (``weights=``) may be any of:

* ``None`` – the package defaults (:data:`ptmreloc.DEFAULT_WEIGHTS`)
* a preset name – ``"medians"``, ``"fun_medians"``, ``"gs_medians"``, ``"htp_only"``
* a path to a ``weights.json`` written by :func:`optimize` (its threshold and
  feature settings are used too, unless overridden)
* an :class:`OptimizationResult`
* a mapping ``{"w_H": ..., "w_L": ..., "w_D": ...}`` or a sequence ``(w_H, w_L, w_D)``
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .database import annotate_residues, database_dir, database_file, default_reference
from .defaults import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, WEIGHT_PRESETS
from .engine import as_weight_vector, choose
from .features import DEFAULT_COLUMNS, WEIGHT_NAMES, FeatureConfig, build_candidate_table
from .metrics import DEFAULT_ASCORE_BINS, evaluate as _evaluate_selection, evaluate_by_ascore
from .optimize import OptimizationResult, optimize_weights
from .reassign import Reassigner, load_weights
from .reference import ReferenceDB, TableLike, read_table, write_table

RefSpec = Union[None, TableLike, ReferenceDB]
OptTable = Union[None, bool, TableLike]
WeightsSpec = Union[None, str, os.PathLike, OptimizationResult, Mapping[str, float], Sequence[float], np.ndarray]


# ------------------------------------------------------------------ helpers
def resolve_weights(weights: WeightsSpec = None) -> Tuple[Dict[str, float], Optional[float], Optional[FeatureConfig]]:
    """Turn any accepted weights spec into ``(weights, threshold or None, FeatureConfig or None)``."""
    if weights is None:
        return dict(DEFAULT_WEIGHTS), None, None
    if isinstance(weights, OptimizationResult):
        return dict(weights.weights), weights.threshold, weights.config
    if isinstance(weights, (str, os.PathLike)):
        key = os.fspath(weights)
        if key.lower() in WEIGHT_PRESETS:
            return dict(WEIGHT_PRESETS[key.lower()]), None, None
        if os.path.isfile(key):
            payload = load_weights(key)
            cfg = payload.get("feature_config")
            return dict(payload["weights"]), payload.get("threshold"), FeatureConfig(**cfg) if cfg else None
        raise ValueError(f"{key!r} is neither a weights file nor a preset {sorted(WEIGHT_PRESETS)}")
    w = as_weight_vector(weights)
    return dict(zip(WEIGHT_NAMES, map(float, w))), None, None


def load_reference(
    psp: RefSpec = None,
    gold: OptTable = None,
    fun: OptTable = None,
    organism: Optional[str] = None,
    database: Optional[Union[str, os.PathLike]] = None,
) -> ReferenceDB:
    """Build the reference used for scoring (PSP) and for evaluation (GS, FUN).

    Each argument may be a DataFrame or file path (your own data), ``None`` (the
    default dataset from ``database`` / the default database folder) or ``False``
    (``gold``/``fun`` only: do not load). If ``psp`` is a :class:`ReferenceDB`, it is
    returned as is, or as a copy extended with explicitly given ``gold``/``fun`` tables.

    >>> ref = load_reference()                          # all default
    >>> ref = load_reference(psp="my_psp.tsv")          # own PSP, default GS/FUN
    >>> ref = load_reference(gold=False, fun=False)     # default PSP only
    """
    if isinstance(psp, ReferenceDB):
        if gold is None and fun is None or gold is False and fun is False:
            return psp
        ref = copy.copy(psp)
        if gold is not None and gold is not False:
            ref.gold = set(psp.gold)
            ref.load_gold(read_table(gold))
        if fun is not None and fun is not False:
            ref.fun = dict(psp.fun)
            ref.load_fun(read_table(fun))
        return ref

    if psp is None and (gold is None or gold is False) and (fun is None or fun is False):
        return default_reference(gold=gold is None, fun=fun is None, organism=organism, path=database)

    def pick(x, name):
        if x is False:
            return None
        return database_file(name, database) if x is None else x

    return ReferenceDB.from_sources(pick(psp, "psp"), pick(gold, "gold"), pick(fun, "fun"), organism=organism)


def _settings(weights, threshold, config):
    w, thr_w, cfg_w = resolve_weights(weights)
    thr = threshold if threshold is not None else (thr_w if thr_w is not None else DEFAULT_THRESHOLD)
    cfg = config if config is not None else (cfg_w if cfg_w is not None else FeatureConfig())
    return w, float(thr), cfg


def _peptides(peptides: TableLike, columns=None, fasta=None, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    df = read_table(peptides)
    res_col = {**DEFAULT_COLUMNS, **(columns or {})}["residue"]
    if res_col not in df.columns:
        label = config.ptm_label if config is not None else "*"
        df = annotate_residues(df, fasta, columns, label)
    return df


# ------------------------------------------------------------------ optimize
def optimize(
    peptides: TableLike,
    psp: RefSpec = None,
    gold: OptTable = None,
    fun: OptTable = None,
    threshold: float = DEFAULT_THRESHOLD,
    n_samples: int = 2000,
    config: Optional[FeatureConfig] = None,
    columns: Optional[Mapping[str, str]] = None,
    out_dir: Optional[str] = None,
    organism: Optional[str] = None,
    fasta: Optional[Union[TableLike, Mapping[str, str]]] = None,
    **kwargs,
) -> OptimizationResult:
    """Optimize (w_H, w_L, w_D) against Gold Standard / Functional Landscape sites.

    Extra keyword arguments go to :func:`ptmreloc.optimize_weights` (``refine``,
    ``holdout_frac``, ``top_frac``, ``gs_weight``, ``log10_bounds``, ``seed``, ``verbose`` ...).
    If ``out_dir`` is given, ``weights.json`` and the trial/evaluation tables are written there.

    >>> res = ptmreloc.optimize("peptides.tsv")                      # default reference data
    >>> res = ptmreloc.optimize(df, psp="my_psp.tsv", gold="my_gs.tsv")  # own data (FUN default)
    >>> res.weights, res.summary()
    """
    ref = load_reference(psp, gold, fun, organism)
    if not ref.has_targets:
        raise ValueError("optimize needs Gold Standard and/or Functional Landscape sites (gold=, fun=)")
    cfg = config or FeatureConfig()
    table = build_candidate_table(_peptides(peptides, columns, fasta, cfg), ref, cfg, columns)
    res = optimize_weights(table, threshold=threshold, n_samples=n_samples, **kwargs)
    if out_dir:
        res.save(out_dir)
    return res


# ------------------------------------------------------------------ reassign
def reassign(
    peptides: TableLike,
    psp: RefSpec = None,
    weights: WeightsSpec = None,
    threshold: Optional[float] = None,
    config: Optional[FeatureConfig] = None,
    columns: Optional[Mapping[str, str]] = None,
    include_candidates: bool = True,
    output: Optional[str] = None,
    organism: Optional[str] = None,
    fasta: Optional[Union[TableLike, Mapping[str, str]]] = None,
) -> pd.DataFrame:
    """Relocalize sites of peptides with AScore <= threshold. Only PSP is required.

    Without ``weights`` the package defaults are used (threshold 30); without
    ``psp`` the default PhosphoSitePlus table is used.
    Returns the input table plus ``NewPeptide``, ``NewResidue``, ``Reassigned``,
    ``ReassignStatus``, ``OriginalSiteProb``, ``NewSiteProb`` (and ``CandidateSiteProbs``).

    >>> out = ptmreloc.reassign(df)                                    # default PSP, default weights
    >>> out = ptmreloc.reassign(df, psp="my_psp.tsv", weights="opt/weights.json")
    """
    w, thr, cfg = _settings(weights, threshold, config)
    ref = load_reference(psp, gold=False, fun=False, organism=organism)
    df = _peptides(peptides, columns, fasta, cfg)
    out = Reassigner(ref, w, thr, cfg, columns).reassign(df, include_candidates)
    if output:
        write_table(out, output)
    return out


# ------------------------------------------------------------------ evaluate
@dataclass
class EvaluationReport:
    weights: Dict[str, float]
    threshold: float
    overall: Dict[str, float]  # metrics on rows with AScore <= threshold
    by_ascore: pd.DataFrame  # metrics per AScore bin (selection applied to every bin)
    config: FeatureConfig = field(default_factory=FeatureConfig)

    def summary(self) -> str:
        keys = ["n_rows", "n_rows_changed", "delta_gs", "random_delta_gs", "oracle_delta_gs", "delta_fun",
                "random_delta_fun", "oracle_delta_fun", "gs_gain", "fun_gain", "composite"]
        head = "\n".join(f"  {k:<18}{self.overall[k]:.4g}" for k in keys)
        cols = ["ascore_bin", "n_rows", "frac_rows_changed", "delta_gs", "random_delta_gs", "delta_fun",
                "random_delta_fun", "gs_gain", "fun_gain", "composite"]
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
            return (f"weights {self.weights}, AScore <= {self.threshold:g}\n{head}\n\n"
                    f"{self.by_ascore[cols].to_string(index=False)}")


def evaluate(
    peptides: TableLike,
    psp: RefSpec = None,
    gold: OptTable = None,
    fun: OptTable = None,
    weights: WeightsSpec = None,
    threshold: Optional[float] = None,
    bins: Sequence[float] = DEFAULT_ASCORE_BINS,
    config: Optional[FeatureConfig] = None,
    columns: Optional[Mapping[str, str]] = None,
    gs_weight: float = 0.5,
    organism: Optional[str] = None,
    fasta: Optional[Union[TableLike, Mapping[str, str]]] = None,
) -> EvaluationReport:
    """Score a weight set against GS/FUN sites, overall and per AScore bin.

    >>> rep = ptmreloc.evaluate(df)                                    # default data, default weights
    >>> print(rep.summary())
    """
    w, thr, cfg = _settings(weights, threshold, config)
    ref = load_reference(psp, gold, fun, organism)
    if not ref.has_targets:
        raise ValueError("evaluate needs Gold Standard and/or Functional Landscape sites (gold=, fun=)")
    table = build_candidate_table(_peptides(peptides, columns, fasta, cfg), ref, cfg, columns)
    sel = choose(table, w)
    overall = _evaluate_selection(table, sel, table.r_ascore <= thr, gs_weight)
    by_bin = evaluate_by_ascore(table, sel, bins=bins, gs_weight=gs_weight)
    return EvaluationReport(w, thr, overall, by_bin, cfg)


def compare_weights(
    peptides: TableLike,
    psp: RefSpec = None,
    gold: OptTable = None,
    fun: OptTable = None,
    weight_sets: Optional[Mapping[str, WeightsSpec]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    config: Optional[FeatureConfig] = None,
    columns: Optional[Mapping[str, str]] = None,
    gs_weight: float = 0.5,
    organism: Optional[str] = None,
    fasta: Optional[Union[TableLike, Mapping[str, str]]] = None,
) -> pd.DataFrame:
    """Evaluate several weight sets on the same data (features built once).

    ``weight_sets`` defaults to all presets. Returns one row per weight set.
    All sets share ``threshold`` and ``config``.
    """
    weight_sets = weight_sets if weight_sets is not None else {k: k for k in WEIGHT_PRESETS}
    ref = load_reference(psp, gold, fun, organism)
    cfg = config or FeatureConfig()
    table = build_candidate_table(_peptides(peptides, columns, fasta, cfg), ref, cfg, columns)
    mask = table.r_ascore <= threshold
    recs = []
    for name, spec in weight_sets.items():
        w, _, _ = resolve_weights(spec)
        recs.append({"weights": name, **w, **_evaluate_selection(table, choose(table, w), mask, gs_weight)})
    return pd.DataFrame(recs)
