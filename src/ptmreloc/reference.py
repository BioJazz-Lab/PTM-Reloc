"""Reference (prior-knowledge) site tables: PhosphoSitePlus, Gold Standard, Functional Landscape."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

Key = Tuple[str, int]
TableLike = Union[str, os.PathLike, pd.DataFrame]


def read_table(src: TableLike, **kwargs) -> pd.DataFrame:
    """Read a DataFrame from a pickle, CSV, TSV/TXT, parquet or Excel file (or pass one through).

    Raw PhosphoSitePlus downloads start with a few lines of license text; for
    ``.txt``/``.tsv`` files the header row is located automatically.
    """
    if isinstance(src, pd.DataFrame):
        return src
    path = os.fspath(src)
    ext = os.path.splitext(path.lower().removesuffix(".gz"))[1]
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if ext == ".csv":
        return pd.read_csv(path, **kwargs)
    if ext in (".tsv", ".txt", ".tab"):
        if "skiprows" not in kwargs:
            kwargs["skiprows"] = _find_header_row(path)
        return pd.read_csv(path, sep="\t", **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"unsupported file type: {path}")


def _find_header_row(path: str, max_lines: int = 20) -> int:
    opener = __import__("gzip").open if path.lower().endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            if "\t" in line and ("MOD_RSD" in line or "UniProt" in line or "ACC_ID" in line):
                return i
    return 0


def _first_col(df: pd.DataFrame, candidates, what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"could not find {what} column; tried {list(candidates)}")


@dataclass
class ReferenceDB:
    """Site-level lookups keyed by ``(UniProt accession, protein position)``.

    * ``htp``  – high-throughput (MS) observation counts in PSP (MS_LIT + MS_CST)
    * ``ltp``  – low-throughput (literature) observation counts in PSP (LT_LIT)
    * ``gold`` – set of Gold Standard sites (used only for evaluation/optimization)
    * ``fun``  – Functional Landscape scores (used only for evaluation/optimization)
    """

    htp: Dict[Key, float] = field(default_factory=dict)
    ltp: Dict[Key, float] = field(default_factory=dict)
    gold: Set[Key] = field(default_factory=set)
    fun: Dict[Key, float] = field(default_factory=dict)

    @property
    def has_targets(self) -> bool:
        return bool(self.gold) or bool(self.fun)

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_sources(
        cls,
        psp: TableLike,
        gold: Optional[TableLike] = None,
        fun: Optional[TableLike] = None,
        organism: Optional[str] = None,
    ) -> "ReferenceDB":
        db = cls()
        db.load_psp(read_table(psp), organism=organism)
        if gold is not None:
            db.load_gold(read_table(gold))
        if fun is not None:
            db.load_fun(read_table(fun))
        return db

    def load_psp(self, df: pd.DataFrame, organism: Optional[str] = None) -> None:
        if organism is not None and "ORGANISM" in df.columns:
            df = df[df["ORGANISM"].astype(str).str.lower() == organism.lower()]
        if "MOD_RSD" in df.columns:
            mod = df["MOD_RSD"].astype(str)
            df = df[~mod.str.contains("-", regex=False) | mod.str.endswith("-p")]
        acc = _first_col(df, ["UniProt", "ACC_ID", "ACC#"], "PSP accession")
        if "Residual Position" in df.columns:
            pos = df["Residual Position"]
        else:
            pos = df["MOD_RSD"].astype(str).str.extract(r"(\d+)")[0]
        pos = pd.to_numeric(pos, errors="coerce")

        def col(name):
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else 0.0

        tab = pd.DataFrame(
            {"acc": df[acc].astype(str).to_numpy(), "pos": pos.to_numpy(),
             "htp": (col("MS_LIT") + col("MS_CST")), "ltp": col("LT_LIT")}
        ).dropna(subset=["pos"])
        tab["pos"] = tab["pos"].astype(int)
        # duplicated (acc, pos) rows: keep the strongest evidence
        tab = tab.groupby(["acc", "pos"], sort=False)[["htp", "ltp"]].max()
        self.htp.update(zip(tab.index, tab["htp"].to_numpy(dtype=float)))
        self.ltp.update(zip(tab.index, tab["ltp"].to_numpy(dtype=float)))

    def load_gold(self, df: pd.DataFrame) -> None:
        acc = _first_col(df, ["UniProt", "ACC_ID", "Protein accession"], "gold-standard accession")
        pos = _first_col(df, ["Position in target protein", "Position", "Residual Position"], "gold-standard position")
        p = pd.to_numeric(df[pos], errors="coerce")
        ok = p.notna().to_numpy()
        self.gold.update(zip(df[acc].astype(str).to_numpy()[ok], p.to_numpy()[ok].astype(int)))

    def load_fun(self, df: pd.DataFrame) -> None:
        acc = _first_col(df, ["UniProt", "ACC_ID"], "functional-score accession")
        pos = _first_col(df, ["Position", "Position in target protein", "Residual Position"], "functional-score position")
        score = _first_col(df, ["Functional_score", "FunctionalScore", "functional_score", "FuncScore"], "functional score")
        tab = pd.DataFrame({"acc": df[acc].astype(str), "pos": pd.to_numeric(df[pos], errors="coerce"),
                            "s": pd.to_numeric(df[score], errors="coerce")}).dropna()
        tab["pos"] = tab["pos"].astype(int)
        tab = tab.groupby(["acc", "pos"], sort=False)["s"].max()
        self.fun.update(zip(tab.index, tab.to_numpy(dtype=float)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ReferenceDB(psp_sites={len(self.htp)}, gold_sites={len(self.gold)}, "
                f"functional_sites={len(self.fun)})")


def write_table(df: pd.DataFrame, path: Union[str, os.PathLike]) -> None:
    """Write a DataFrame; format from extension (.pkl/.csv/.parquet/.xlsx, otherwise TSV)."""
    path = os.fspath(path)
    ext = os.path.splitext(path.lower())[1]
    if ext in (".pkl", ".pickle"):
        df.to_pickle(path)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)
