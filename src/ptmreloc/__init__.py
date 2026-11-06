"""ptmreloc — prior-knowledge guided relocalization of low-confidence PTM sites.

Quick start. The bundled human reference database (PhosphoSitePlus, Gold Standard,
Functional Landscape, UniProt FASTA) is used unless you pass your own tables::

    import ptmreloc as pr

    out = pr.reassign(peptides)                          # default weights, bundled PSP
    res = pr.optimize(peptides, out_dir="opt/")          # optimize against bundled GS/FUN
    out = pr.reassign(peptides, weights=res)             # or weights="opt/weights.json"
    rep = pr.evaluate(peptides, weights=res); print(rep.summary())

    # own data: any of psp= / gold= / fun= / fasta= (DataFrame or file path)
    out = pr.reassign(peptides, psp="my_psp.tsv")

The lower-level building blocks (``build_candidate_table``, ``choose``,
``evaluate_selection``, ``optimize_weights``, ``Reassigner``) are exported too.
"""
from .api import EvaluationReport, compare_weights, evaluate, load_reference, optimize, reassign, resolve_weights
from .database import (DATABASE_FILES, annotate_residues, database_dir, database_file,
                       database_info, default_proteome, default_reference, read_fasta)
from .defaults import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, WEIGHT_PRESETS
from .engine import choose, site_scores
from .features import FEATURE_NAMES, WEIGHT_NAMES, CandidateTable, FeatureConfig, build_candidate_table
from .metrics import composite_score, evaluate_by_ascore, trial_relative_composite
from .metrics import evaluate as evaluate_selection
from .optimize import OptimizationResult, normalize_weights, optimize_weights
from .peptide import format_residue, parse_peptide, parse_residue, relabel
from .reassign import Reassigner, apply_selection, load_weights
from .reference import ReferenceDB, read_table, write_table

__version__ = "0.1.0"

__all__ = [
    # high-level API
    "optimize", "reassign", "evaluate", "compare_weights", "load_reference", "resolve_weights",
    "EvaluationReport", "OptimizationResult",
    # bundled database
    "DATABASE_FILES", "database_dir", "database_file", "database_info", "default_reference",
    "default_proteome", "read_fasta", "annotate_residues",
    # defaults
    "DEFAULT_WEIGHTS", "DEFAULT_THRESHOLD", "WEIGHT_PRESETS",
    # building blocks
    "ReferenceDB", "read_table", "write_table", "FeatureConfig", "CandidateTable", "build_candidate_table",
    "FEATURE_NAMES", "WEIGHT_NAMES", "choose", "site_scores", "evaluate_selection", "evaluate_by_ascore",
    "composite_score", "trial_relative_composite", "optimize_weights", "normalize_weights",
    "Reassigner", "apply_selection", "load_weights", "parse_peptide", "parse_residue", "format_residue",
    "relabel",
]
