"""Parsing and relabeling of modified peptide strings.

Peptides are expected in the common search-engine notation ``K.PEPT*IDE.R``:
optional single-residue flanks separated by ``.``, upper-case residues, and
modification symbols (e.g. ``*`` for phosphorylation) placed *after* the
modified residue. Residue annotations follow ``S8sT10t`` (amino acid, protein
position, lower-case amino acid), one block per modified site.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

_RESIDUE_RE = re.compile(r"([A-Z])(\d+)[a-z]?")


@dataclass
class ParsedPeptide:
    """A peptide split into residues and the modification symbols attached to them."""

    prefix: str  # e.g. "K." (left flank incl. dot) or ""
    suffix: str  # e.g. ".R" (right flank incl. dot) or ""
    nterm: str  # modification symbols before the first residue
    aas: str  # bare residue sequence of the core peptide
    mods: List[str] = field(default_factory=list)  # symbols following each residue
    labeled: List[int] = field(default_factory=list)  # 0-based indices carrying the PTM label


def parse_peptide(peptide: str, ptm_label: str = "*") -> ParsedPeptide:
    """Parse ``K.PEPT*IDE.R`` into a :class:`ParsedPeptide`."""
    if not isinstance(peptide, str) or not peptide:
        raise ValueError(f"invalid peptide: {peptide!r}")
    parts = peptide.split(".")
    if len(parts) >= 3:
        prefix, core, suffix = parts[0] + ".", ".".join(parts[1:-1]), "." + parts[-1]
    else:
        prefix, core, suffix = "", peptide, ""

    nterm = ""
    aas: List[str] = []
    mods: List[str] = []
    for ch in core:
        if "A" <= ch <= "Z":
            aas.append(ch)
            mods.append("")
        elif aas:
            mods[-1] += ch
        else:
            nterm += ch
    if not aas:
        raise ValueError(f"peptide has no residues: {peptide!r}")
    labeled = [i for i, m in enumerate(mods) if ptm_label in m]
    return ParsedPeptide(prefix, suffix, nterm, "".join(aas), mods, labeled)


def relabel(parsed: ParsedPeptide, new_indices: Sequence[int], ptm_label: str = "*") -> str:
    """Return the peptide string with ``ptm_label`` moved to ``new_indices``.

    Any other modification symbols are preserved in place.
    """
    new = set(int(i) for i in new_indices)
    out = [parsed.prefix, parsed.nterm]
    for i, (aa, m) in enumerate(zip(parsed.aas, parsed.mods)):
        m = m.replace(ptm_label, "")
        if i in new:
            m = ptm_label + m
        out.append(aa + m)
    out.append(parsed.suffix)
    return "".join(out)


def parse_residue(residue: str) -> List[Tuple[str, int]]:
    """Parse ``S8sT10t`` into ``[("S", 8), ("T", 10)]``."""
    if not isinstance(residue, str):
        raise ValueError(f"invalid residue annotation: {residue!r}")
    sites = [(aa, int(pos)) for aa, pos in _RESIDUE_RE.findall(residue)]
    if not sites:
        raise ValueError(f"invalid residue annotation: {residue!r}")
    return sites


def format_residue(sites: Sequence[Tuple[str, int]]) -> str:
    """Inverse of :func:`parse_residue`."""
    return "".join(f"{aa}{pos}{aa.lower()}" for aa, pos in sorted(sites, key=lambda s: s[1]))
