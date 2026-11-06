"""Bundled reference database (human PhosphoSitePlus, Gold Standard, Functional Landscape, UniProt FASTA).

The files live in the repository's ``database/`` folder and are installed as the
``ptmreloc._data`` package. Every API function uses them when you don't pass your
own tables. Resolution order of the database folder:

1. ``database_dir(path)`` / the ``PTMRELOC_DATABASE`` environment variable
2. the installed ``ptmreloc._data`` package
3. ``<repo>/database`` next to the source tree (development checkouts)
"""
from __future__ import annotations

import functools
import importlib.util
import os
import warnings
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pandas as pd

from .peptide import format_residue, parse_peptide
from .reference import ReferenceDB, read_table

ENV_VAR = "PTMRELOC_DATABASE"

#: file names of the bundled datasets
DATABASE_FILES: Dict[str, str] = {
    "psp": "phospho_site_plus_human_clustered_indexed_by_name.csv",
    "gold": "gold_standard_sites_reindexed.tsv",
    "fun": "functional_landscape_sites_reindexed.tsv",
    "fasta": "UP000005640_9606.fasta",
}


def database_dir(path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Folder holding the reference datasets (see module docstring for the search order)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    try:
        spec = importlib.util.find_spec("ptmreloc._data")
        if spec is not None and spec.submodule_search_locations:
            candidates.extend(Path(p) for p in spec.submodule_search_locations)
    except (ImportError, ValueError):
        pass
    candidates.append(Path(__file__).resolve().parents[2] / "database")
    for c in candidates:
        if c.is_dir() and any((c / f).is_file() for f in DATABASE_FILES.values()):
            return c
    raise FileNotFoundError(
        "bundled reference database not found; pass your own tables or set "
        f"{ENV_VAR} to a folder containing {sorted(DATABASE_FILES.values())}")


def database_file(name: str, path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Path of a bundled dataset: ``"psp"``, ``"gold"``, ``"fun"`` or ``"fasta"``."""
    if name not in DATABASE_FILES:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATABASE_FILES)}")
    f = database_dir(path) / DATABASE_FILES[name]
    if not f.is_file():
        raise FileNotFoundError(f)
    return f


def database_info(path: Optional[Union[str, os.PathLike]] = None) -> pd.DataFrame:
    """Which bundled datasets are available, with sizes."""
    d = database_dir(path)
    return pd.DataFrame([{"dataset": k, "file": str(d / v), "present": (d / v).is_file(),
                          "size_MB": round((d / v).stat().st_size / 1e6, 1) if (d / v).is_file() else None}
                         for k, v in DATABASE_FILES.items()])


# ------------------------------------------------------------------ reference tables
@functools.lru_cache(maxsize=4)
def _cached_reference(dirpath: str, with_gold: bool, with_fun: bool, organism: Optional[str]) -> ReferenceDB:
    d = Path(dirpath)
    return ReferenceDB.from_sources(
        d / DATABASE_FILES["psp"],
        d / DATABASE_FILES["gold"] if with_gold else None,
        d / DATABASE_FILES["fun"] if with_fun else None,
        organism=organism)


def default_reference(gold: bool = True, fun: bool = True, organism: Optional[str] = None,
                      path: Optional[Union[str, os.PathLike]] = None) -> ReferenceDB:
    """The bundled reference (cached; treat it as read-only)."""
    return _cached_reference(str(database_dir(path)), gold, fun, organism)


# ------------------------------------------------------------------ protein sequences
def read_fasta(path: Union[str, os.PathLike]) -> Dict[str, str]:
    """Read a FASTA file into ``{id: sequence}``.

    UniProt headers (``>sp|P12345|NAME_HUMAN ...``) are indexed by both the
    accession (``P12345``) and the entry name (``NAME_HUMAN``); other headers by
    their first word.
    """
    seqs: Dict[str, str] = {}
    keys, buf = [], []

    def flush():
        if keys:
            s = "".join(buf)
            for k in keys:
                seqs.setdefault(k, s)

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                head = line[1:].split()[0] if len(line) > 1 else ""
                parts = head.split("|")
                keys = [parts[1], parts[2]] if len(parts) >= 3 else [head]
                buf = []
            elif line:
                buf.append(line)
    flush()
    return seqs


@functools.lru_cache(maxsize=2)
def _cached_fasta(path: str) -> Dict[str, str]:
    return read_fasta(path)


def default_proteome(path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, str]:
    """Bundled human proteome ``{accession or entry name: sequence}`` (cached)."""
    return _cached_fasta(str(database_file("fasta", path)))


def annotate_residues(
    df: pd.DataFrame,
    proteins: Optional[Union[Mapping[str, str], str, os.PathLike]] = None,
    columns: Optional[Mapping[str, str]] = None,
    ptm_label: str = "*",
) -> pd.DataFrame:
    """Add the residue annotation (e.g. ``S8sT10t``) by locating each peptide in its protein.

    ``proteins`` is a ``{id: sequence}`` mapping or a FASTA path (default: the
    bundled human proteome). The first occurrence of the peptide in the protein is
    used; rows whose protein or peptide is not found get an empty annotation (and
    are later passed through unchanged).
    """
    from .features import DEFAULT_COLUMNS

    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    if proteins is None:
        proteins = default_proteome()
    elif not isinstance(proteins, Mapping):
        proteins = read_fasta(proteins)

    cache: Dict[tuple, str] = {}
    out = []
    for u, p in zip(df[cols["uniprot"]].astype(str), df[cols["peptide"]].astype(str)):
        key = (u, p)
        if key not in cache:
            res = ""
            seq = proteins.get(u)
            if seq is not None:
                try:
                    pp = parse_peptide(p, ptm_label)
                    i = seq.find(pp.aas)
                    if i >= 0 and pp.labeled:
                        res = format_residue([(pp.aas[j], i + j + 1) for j in pp.labeled])
                except ValueError:
                    pass
            cache[key] = res
        out.append(cache[key])
    df = df.copy()
    df[cols["residue"]] = out
    missing = sum(1 for r in out if not r)
    if missing:
        warnings.warn(f"{missing} of {len(out)} rows could not be located in the protein sequences")
    return df
