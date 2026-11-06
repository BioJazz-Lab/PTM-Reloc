"""Example: optimize the reassignment weights on the MCF10A EGF time-course phosphoproteome.

The weights (w_H, w_L, w_D) are tuned so that relocalized sites agree as much as
possible with the Gold Standard and Functional Landscape sites in the bundled
reference database. The best weights are written to ``results/<sheet>/weights.json``,
which ``reassign_sites.py`` can use.

Usage:
    python optimize_weights.py                                # MCF10A, all peptides
    python optimize_weights.py --sheet HS578T_phospho         # the other cell line
    python optimize_weights.py --mono-only --n-samples 10000  # manuscript setting
"""
import argparse
from pathlib import Path

import pandas as pd

import ptmreloc as pr

HERE = Path(__file__).resolve().parent
DATA = HERE / "MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="MCF10A_phospho", choices=["MCF10A_phospho", "HS578T_phospho"])
    ap.add_argument("--threshold", type=float, default=30, help="only peptides with AScore <= threshold are relocalized")
    ap.add_argument("--n-samples", type=int, default=2000, help="number of Monte Carlo weight samples")
    ap.add_argument("--mono-only", action="store_true", help="optimize on monophosphorylated peptides only")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: results/<sheet>)")
    args = ap.parse_args()
    out_dir = args.out or HERE / "results" / args.sheet

    # 1. Load the phosphopeptide table: needs UniProt, Peptide, Residue and AScore columns.
    peptides = pd.read_excel(DATA, sheet_name=args.sheet,
                             usecols=["Peptide", "Site", "UniProt", "Protein", "Residue", "AScore"])
    if args.mono_only:
        peptides = peptides[peptides["Residue"].str.count(r"[A-Z]\d+") == 1]
    print(f"{args.sheet}: {len(peptides)} phosphopeptides, "
          f"{(peptides['AScore'] <= args.threshold).sum()} with AScore <= {args.threshold:g}")

    # 2. Optimize. The bundled PhosphoSitePlus / Gold Standard / Functional Landscape
    #    tables are used; pass psp=, gold=, fun= to use your own.
    result = pr.optimize(
        peptides,
        threshold=args.threshold,
        n_samples=args.n_samples,
        holdout_frac=0.2,      # 20% of proteins held out to check for overfitting
        seed=0,
        verbose=True,
        out_dir=out_dir,       # weights.json, trials.csv, evaluation.csv, metrics_by_ascore.csv
    )
    print()
    print(result.summary())

    # 3. Compare with the package defaults and the presets on the same peptides.
    print("\nComparison with the built-in weight sets:")
    comparison = pr.compare_weights(
        peptides, threshold=args.threshold,
        weight_sets={"optimized": result, "default": None, "htp_only": "htp_only"},
    )
    cols = ["weights", "w_H", "w_L", "w_D", "n_rows_changed", "delta_gs", "delta_fun", "gs_gain", "fun_gain", "composite"]
    print(comparison[cols].round(4).to_string(index=False))

    # 4. How much the relocalization helps per AScore range (applied to every range).
    print("\nOptimized weights by AScore range:")
    by_bin = result.by_ascore[["ascore_bin", "n_rows", "frac_rows_changed", "delta_gs", "random_delta_gs",
                               "delta_fun", "random_delta_fun", "composite"]]
    print(by_bin.round(4).to_string(index=False))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
