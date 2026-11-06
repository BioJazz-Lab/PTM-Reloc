"""Command-line interface: ``ptmreloc optimize | reassign | evaluate``."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import api
from .defaults import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, WEIGHT_PRESETS
from .features import FeatureConfig
from .reference import read_table, write_table


def _add_common(p: argparse.ArgumentParser, targets: bool) -> None:
    p.add_argument("-i", "--input", required=True, help="peptide table (pkl/csv/tsv/parquet/xlsx)")
    g = p.add_argument_group("reference data (default: human reference database, downloaded on first use)")
    g.add_argument("--psp", default=None, help="own PhosphoSitePlus site table")
    if targets:
        g.add_argument("--gold", default=None, help="own Gold Standard site table ('none' to disable)")
        g.add_argument("--fun", default=None, help="own Functional Landscape site table ('none' to disable)")
    g.add_argument("--fasta", default=None,
                   help="own protein FASTA, used only when the input has no residue column")
    g.add_argument("--database", default=None, help="folder with reference files (same names) to use instead")
    g.add_argument("--organism", default=None, help="filter own PSP by ORGANISM (e.g. human)")
    p.add_argument("--sheet", default=None, help="Excel sheet name for --input")
    g = p.add_argument_group("input columns")
    g.add_argument("--col-uniprot", default="UniProt")
    g.add_argument("--col-peptide", default="Peptide")
    g.add_argument("--col-residue", default="Residue")
    g.add_argument("--col-ascore", default="AScore")


def _add_features(p: argparse.ArgumentParser, inherit: bool) -> None:
    g = p.add_argument_group("features" + (" (default: from --weights file, else legacy)" if inherit else ""))
    d = None if inherit else argparse.SUPPRESS
    g.add_argument("--ptm-label", default=d)
    g.add_argument("--candidate-residues", default=d)
    g.add_argument("--evidence-transform", choices=["legacy", "log", "raw"], default=d)
    g.add_argument("--distance-mode", choices=["legacy", "inverse"], default=d)


def _config(a, base: FeatureConfig = FeatureConfig()):
    """Explicit feature flags override ``base``; returns None when nothing was given and base is None."""
    over = {k: getattr(a, k, None) for k in ("ptm_label", "candidate_residues", "evidence_transform", "distance_mode")}
    over = {k: v for k, v in over.items() if v is not None}
    return FeatureConfig(**{**base.to_dict(), **over})


def _opt_table(x):
    return False if isinstance(x, str) and x.lower() == "none" else x


def _ref(a, targets: bool):
    gold = _opt_table(a.gold) if targets else False
    fun = _opt_table(a.fun) if targets else False
    return api.load_reference(a.psp, gold, fun, organism=a.organism, database=a.database)


def _columns(a):
    return {"uniprot": a.col_uniprot, "peptide": a.col_peptide, "residue": a.col_residue, "ascore": a.col_ascore}


def _read_input(a):
    kw = {"sheet_name": a.sheet} if a.sheet else {}
    return read_table(a.input, **kw)


def _weights_spec(a):
    if a.w is not None:
        return list(a.w)
    return a.weights  # file, preset name, or None -> defaults


def _resolved(a):
    w, thr, cfg = api.resolve_weights(_weights_spec(a))
    thr = a.threshold if a.threshold is not None else (thr if thr is not None else DEFAULT_THRESHOLD)
    return w, thr, _config(a, cfg or FeatureConfig())


def cmd_optimize(a) -> int:
    t0 = time.time()
    res = api.optimize(
        _read_input(a), _ref(a, True), fasta=a.fasta, threshold=a.threshold, n_samples=a.n_samples,
        config=_config(a), columns=_columns(a), out_dir=a.out, organism=a.organism,
        log10_bounds=(a.log10_low, a.log10_high), refine=not a.no_refine, refine_maxiter=a.refine_maxiter,
        top_frac=a.top_frac, holdout_frac=a.holdout, gs_weight=a.gs_weight, seed=a.seed, verbose=True,
    )
    print(res.summary())
    print(f"\nwrote {a.out}/weights.json, trials.csv, evaluation.csv, metrics_by_ascore.csv "
          f"({time.time() - t0:.1f}s)")
    return 0


def cmd_reassign(a) -> int:
    w, thr, cfg = _resolved(a)
    out = api.reassign(_read_input(a), _ref(a, False), fasta=a.fasta, weights=w, threshold=thr, config=cfg, columns=_columns(a),
                       include_candidates=not a.no_candidates, organism=a.organism)
    write_table(out, a.output)
    n_elig = int((out["ReassignStatus"] == "eligible").sum())
    print(f"weights {w}, AScore <= {thr:g}")
    print(f"{len(out)} rows; {n_elig} eligible; {int(out['Reassigned'].sum())} relocalized; "
          f"{int((out['ReassignStatus'] == 'unparsed').sum())} unparsed -> {a.output}")
    return 0


def cmd_evaluate(a) -> int:
    w, thr, cfg = _resolved(a)
    bins = [-np.inf] + [float(x) for x in a.bins.split(",")]
    rep = api.evaluate(_read_input(a), _ref(a, True), fasta=a.fasta, weights=w, threshold=thr, bins=bins,
                       config=cfg, columns=_columns(a), organism=a.organism)
    print(rep.summary())
    if a.output:
        write_table(rep.by_ascore, a.output)
    return 0


def cmd_info(a) -> int:
    from .database import _database_url, cache_dir, database_info

    with __import__("pandas").option_context("display.width", 200, "display.max_colwidth", 120):
        print(database_info(a.database).to_string(index=False))
    print(f"\ndownload cache: {cache_dir()}\ndownload URL:   {_database_url() or '(not configured)'}")
    print(f"\ndefault weights: {DEFAULT_WEIGHTS}, AScore threshold {DEFAULT_THRESHOLD:g}")
    print(f"presets: {', '.join(WEIGHT_PRESETS)}")
    return 0


def cmd_download(a) -> int:
    from .database import download_database

    dest = download_database(url=a.url, dest=a.dest, force=a.force, verify=not a.no_verify)
    print(f"reference data ready in {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ptmreloc", description="Prior-knowledge guided PTM site relocalization")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="show the reference data files in use and the default weights")
    p.add_argument("--database", default=None)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("download", help="download the reference data (otherwise fetched on first use)")
    p.add_argument("--url", default=None, help="base URL of the data files (default: the package's)")
    p.add_argument("--dest", default=None, help="target folder (default: the per-user cache)")
    p.add_argument("--force", action="store_true", help="download again even if present")
    p.add_argument("--no-verify", action="store_true", help="skip checksum verification")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("optimize", help="optimize weights against Gold Standard / Functional sites")
    _add_common(p, targets=True)
    _add_features(p, inherit=True)
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="AScore cutoff (reassign AScore <= cutoff)")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--log10-low", type=float, default=-3.0)
    p.add_argument("--log10-high", type=float, default=0.0)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--refine-maxiter", type=int, default=30)
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--holdout", type=float, default=0.2, help="fraction of proteins held out")
    p.add_argument("--gs-weight", type=float, default=0.5, help="GS weight in the composite score")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_optimize)

    wh = (f"weights.json from `ptmreloc optimize`, or a preset ({', '.join(WEIGHT_PRESETS)}). "
          f"Default: {DEFAULT_WEIGHTS}")
    for name, func, targets, help_ in [
        ("reassign", cmd_reassign, False, "relocalize sites (default weights unless --weights/--w)"),
        ("evaluate", cmd_evaluate, True, "score weights against GS/FUN, per AScore bin"),
    ]:
        p = sub.add_parser(name, help=help_)
        _add_common(p, targets=targets)
        _add_features(p, inherit=True)
        p.add_argument("--weights", default=None, help=wh)
        p.add_argument("--w", nargs=3, type=float, metavar=("W_H", "W_L", "W_D"), help="explicit weights")
        p.add_argument("--threshold", type=float, default=None,
                       help=f"AScore cutoff (default: from weights file, else {DEFAULT_THRESHOLD:g})")
        if name == "reassign":
            p.add_argument("-o", "--output", required=True, help="output table (.tsv/.csv/.pkl/.parquet/.xlsx)")
            p.add_argument("--no-candidates", action="store_true", help="omit CandidateSiteProbs column")
        else:
            p.add_argument("--bins", default="0,10,20,30,50,100,inf", help="AScore bin edges")
            p.add_argument("-o", "--output", default=None, help="write per-bin metrics table")
        p.set_defaults(func=func)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
