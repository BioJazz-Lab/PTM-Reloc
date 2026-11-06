"""Example: relocalize low-confidence phosphosites in the MCF10A / HS578T EGF time-course data.

Peptides with AScore <= 30 get their phosphosite(s) moved to the candidate residues
best supported by PhosphoSitePlus. The output keeps every column of the input
(including the TMT quantification) and adds:

    NewPeptide, NewResidue, NewSite   relocalized peptide / residue / site label
    Reassigned                        True if the site moved
    ReassignStatus                    eligible / above_threshold / unparsed
    OriginalSiteProb, NewSiteProb     share of the reassignment score at the old / new site
    CandidateSiteProbs                all candidate residues with their score share

Usage:
    python reassign_sites.py                                       # default weights, both cell lines
    python reassign_sites.py --weights results/MCF10A_phospho/weights.json
    python reassign_sites.py --weights htp_only --threshold 20     # preset weights, other cutoff
"""
import argparse
from pathlib import Path

import pandas as pd

import ptmreloc as pr

HERE = Path(__file__).resolve().parent
DATA = HERE / "MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx"
SHEETS = ["MCF10A_phospho", "HS578T_phospho"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="weights.json from optimize_weights.py or a preset name "
                         "(medians, fun_medians, gs_medians, htp_only); default: package defaults")
    ap.add_argument("--threshold", type=float, default=None,
                    help="AScore cutoff (default: from the weights file, else 30)")
    ap.add_argument("--out", type=Path, default=HERE / "results", help="output folder")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    weights, threshold, _ = pr.resolve_weights(args.weights)
    if args.threshold is not None:
        threshold = args.threshold
    elif threshold is None:
        threshold = pr.DEFAULT_THRESHOLD
    print(f"weights: {weights}\nAScore threshold: {threshold:g}\n")

    # Load the default PhosphoSitePlus table once and reuse it for both cell lines.
    # (Use pr.load_reference(psp="my_psp.txt", ...) for your own reference data.)
    reference = pr.load_reference(gold=False, fun=False)

    for sheet in SHEETS:
        peptides = pd.read_excel(DATA, sheet_name=sheet)
        out = pr.reassign(peptides, reference, weights=args.weights, threshold=threshold)

        # Site label in the same style as the input "Site" column, e.g. NUD4B_HUMAN-S149s
        out.insert(out.columns.get_loc("Residue") + 1, "NewSite", out["Protein"] + "-" + out["NewResidue"])
        added = ["NewPeptide", "NewResidue", "Reassigned", "ReassignStatus",
                 "OriginalSiteProb", "NewSiteProb", "CandidateSiteProbs"]
        out = out[[c for c in out.columns if c not in added] + added]  # new columns next to the originals

        path = args.out / f"{sheet}_reassigned.tsv"
        out.to_csv(path, sep="\t", index=False)

        eligible = out["ReassignStatus"] == "eligible"
        moved = out[out["Reassigned"]]
        print(f"== {sheet}")
        print(f"  phosphopeptides:           {len(out)}")
        print(f"  AScore <= {threshold:g} (eligible):   {eligible.sum()}")
        print(f"  relocalized:               {len(moved)} ({len(moved) / max(1, eligible.sum()):.1%} of eligible)")
        print(f"  unique sites before/after: {out['Site'].nunique()} -> {out['NewSite'].nunique()}")
        print(f"  written to {path}")
        print(moved[["Peptide", "AScore", "NewPeptide", "Site", "NewSite", "NewSiteProb"]]
              .head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
