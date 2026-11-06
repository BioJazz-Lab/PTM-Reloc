"""Reference database (human PhosphoSitePlus, Gold Standard, Functional Landscape, UniProt FASTA).

The data files are not part of the installed package. They are downloaded once,
on first use, into a per-user cache folder, and verified against SHA-256 checksums
(see :func:`download_database`). Every API function uses them when you don't
pass your own tables.

Search order for each file:

1. an explicit folder (``database=`` / ``path=`` arguments), used exclusively
2. the ``PTMRELOC_DATABASE`` environment variable
3. ``<repo>/database`` next to the source tree (development checkouts)
4. the download cache (``PTMRELOC_CACHE_DIR``, default: the platform's user cache folder)

If a file is in none of them it is downloaded from :data:`DATABASE_URL` (or
``PTMRELOC_DATABASE_URL``) into the cache. Set ``PTMRELOC_AUTO_DOWNLOAD=0`` to
disable automatic downloads.
"""
from __future__ import annotations

import functools
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd

from .peptide import format_residue, parse_peptide
from .reference import ReferenceDB, read_table

ENV_VAR = "PTMRELOC_DATABASE"
CACHE_ENV_VAR = "PTMRELOC_CACHE_DIR"
URL_ENV_VAR = "PTMRELOC_DATABASE_URL"
AUTO_DOWNLOAD_ENV_VAR = "PTMRELOC_AUTO_DOWNLOAD"

#: Base URL of the hosted data; each file is fetched from ``<DATABASE_URL>/<file name>``.
#: Zenodo record "Datasets used for PTM-Reloc", DOI 10.5281/zenodo.23997751
DATABASE_URL: Optional[str] = "https://zenodo.org/records/23997751/files"
DATABASE_DOI = "10.5281/zenodo.23997751"

#: file names of the reference datasets
DATABASE_FILES: Dict[str, str] = {
    "psp": "phospho_site_plus_human_clustered_indexed_by_name.csv",
    "gold": "gold_standard_sites_reindexed.tsv",
    "fun": "functional_landscape_sites_reindexed.tsv",
    "fasta": "UP000005640_9606.fasta",
}

#: SHA-256 checksums of the reference files (verified after download)
DATABASE_SHA256: Dict[str, str] = {
    "psp": "e688f5a0cf26864a8f9d132b368e01aabaf9eef2c7d33afafc9e7136093dff40",
    "gold": "f7d4bde85dea2acb64daf8c76bd84840586a33fd7de10dd7db11c676d70b58a6",
    "fun": "9de99746e3b491479ca8181745b76344b655e16d9611674960f53feb5844c2b6",
    "fasta": "6227dc7449e3cc56957e6ad69a0c587bf1d38a8ab8ad4da565e27b201fd249e0",
}

PathLike = Union[str, os.PathLike]


# ------------------------------------------------------------------ locations
def cache_dir() -> Path:
    """Per-user folder the reference data is downloaded into."""
    if os.environ.get(CACHE_ENV_VAR):
        return Path(os.environ[CACHE_ENV_VAR])
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    return base / "ptmreloc"


def _source_tree_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "database"


def _search_dirs(path: Optional[PathLike] = None) -> List[Path]:
    if path is not None:
        return [Path(path)]
    dirs = []
    if os.environ.get(ENV_VAR):
        dirs.append(Path(os.environ[ENV_VAR]))
    src = _source_tree_dir()
    if (src.parent / "pyproject.toml").is_file():  # only in a source checkout
        dirs.append(src)
    dirs.append(cache_dir())
    return dirs


def _database_url(url: Optional[str] = None) -> Optional[str]:
    return url or os.environ.get(URL_ENV_VAR) or DATABASE_URL


def _auto_download() -> bool:
    return os.environ.get(AUTO_DOWNLOAD_ENV_VAR, "1").strip().lower() not in ("0", "false", "no", "off")


def _check_name(name: str) -> None:
    if name not in DATABASE_FILES:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATABASE_FILES)}")


def _not_found(name: str, dirs: Iterable[Path]) -> FileNotFoundError:
    return FileNotFoundError(
        f"reference file {DATABASE_FILES[name]!r} not found in {[str(d) for d in dirs]}. "
        f"Run `ptmreloc download` / ptmreloc.download_database(), set {ENV_VAR} to a folder "
        f"holding the files, or pass your own table (psp=, gold=, fun=, fasta=).")


# ------------------------------------------------------------------ download
def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_database(
    url: Optional[str] = None,
    dest: Optional[PathLike] = None,
    datasets: Optional[Iterable[str]] = None,
    force: bool = False,
    verify: bool = True,
    quiet: bool = False,
) -> Path:
    """Download the reference files into ``dest`` (default: :func:`cache_dir`).

    Files already present with the expected checksum are skipped unless ``force``.
    ``url`` is the base URL holding the files (default: ``PTMRELOC_DATABASE_URL``
    or :data:`DATABASE_URL`). With ``verify``, each download must match
    :data:`DATABASE_SHA256` (use ``verify=False`` for a self-hosted, different version).
    Returns the destination folder.
    """
    base = _database_url(url)
    if not base:
        raise RuntimeError(
            f"no download location configured; pass url= or set {URL_ENV_VAR}")
    base = base.rstrip("/")
    dest = Path(dest) if dest is not None else cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    names = list(datasets) if datasets is not None else list(DATABASE_FILES)
    for name in names:
        _check_name(name)
        target = dest / DATABASE_FILES[name]
        expected = DATABASE_SHA256.get(name) if verify else None
        if target.is_file() and not force and (expected is None or _sha256(target) == expected):
            continue
        src = f"{base}/{DATABASE_FILES[name]}"
        if not quiet:
            print(f"ptmreloc: downloading {DATABASE_FILES[name]} -> {dest}", file=sys.stderr, flush=True)
        fd, tmp = tempfile.mkstemp(dir=dest, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(src) as resp:
                shutil.copyfileobj(resp, out, 1 << 20)
            if expected is not None and _sha256(Path(tmp)) != expected:
                raise ValueError(f"checksum mismatch for {src}; file not saved")
            os.chmod(tmp, 0o666 & ~_umask())  # mkstemp creates 0600; use normal file permissions
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return dest


# ------------------------------------------------------------------ lookup
def database_file(name: str, path: Optional[PathLike] = None, download: bool = True) -> Path:
    """Path of a reference dataset (``"psp"``, ``"gold"``, ``"fun"`` or ``"fasta"``).

    Downloads it into the cache if it is not found and downloading is allowed.
    """
    _check_name(name)
    dirs = _search_dirs(path)
    for d in dirs:
        f = d / DATABASE_FILES[name]
        if f.is_file():
            return f
    if path is None and download and _auto_download() and _database_url():
        return download_database(datasets=[name]) / DATABASE_FILES[name]
    raise _not_found(name, dirs)


def database_dir(path: Optional[PathLike] = None) -> Path:
    """First folder (in search order) that holds any reference file; no download."""
    dirs = _search_dirs(path)
    for d in dirs:
        if d.is_dir() and any((d / f).is_file() for f in DATABASE_FILES.values()):
            return d
    raise FileNotFoundError(f"no reference files found in {[str(d) for d in dirs]}")


def database_info(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Where each reference dataset would be read from (no download)."""
    rows = []
    for name, fname in DATABASE_FILES.items():
        try:
            f = database_file(name, path, download=False)
            rows.append({"dataset": name, "file": str(f), "present": True,
                         "size_MB": round(f.stat().st_size / 1e6, 1)})
        except FileNotFoundError:
            rows.append({"dataset": name, "file": fname, "present": False, "size_MB": None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reference tables
@functools.lru_cache(maxsize=4)
def _cached_reference(psp: str, gold: Optional[str], fun: Optional[str], organism: Optional[str]) -> ReferenceDB:
    return ReferenceDB.from_sources(psp, gold, fun, organism=organism)


def default_reference(gold: bool = True, fun: bool = True, organism: Optional[str] = None,
                      path: Optional[PathLike] = None) -> ReferenceDB:
    """The reference built from the default data files (cached; treat it as read-only)."""
    return _cached_reference(
        str(database_file("psp", path)),
        str(database_file("gold", path)) if gold else None,
        str(database_file("fun", path)) if fun else None,
        organism)


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
    default human proteome). The first occurrence of the peptide in the protein is
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
