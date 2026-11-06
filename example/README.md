# Examples

These examples use the EGF time-course phosphoproteomics data of MCF10A and HS578T cells,
`MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx`. The sheets `MCF10A_phospho` and
`HS578T_phospho` hold about 45k phosphopeptides each, with `UniProt`, `Peptide`, `Residue` and
`AScore` columns plus the TMT quantification.

Both scripts assume `ptmreloc` is installed with Excel support (`pip install "ptmreloc[excel]"`, or `pip install ".[excel]"` from
the repository root, which adds `openpyxl`) and
use the default reference database (downloaded automatically on first use). They can be run from any folder.

## 1. Optimize the weights: `optimize_weights.py`

```bash
python optimize_weights.py                                # MCF10A, about 1 min
python optimize_weights.py --sheet HS578T_phospho
python optimize_weights.py --mono-only --n-samples 10000  # manuscript setting
```

The script tunes (w_H, w_L, w_D) against the Gold Standard and Functional Landscape sites,
with 20% of proteins held out. It compares the result with the default weights, prints the
gain per AScore range, and writes `results/<sheet>/weights.json` plus the trial and
evaluation tables.

## 2. Relocalize sites: `reassign_sites.py`

```bash
python reassign_sites.py                                             # package default weights
python reassign_sites.py --weights results/MCF10A_phospho/weights.json
python reassign_sites.py --weights htp_only --threshold 20
```

For both cell lines, the script relocalizes peptides with AScore <= 30. It writes
`results/<sheet>_reassigned.tsv`, which contains all input columns plus `NewPeptide`,
`NewResidue`, `NewSite`, `Reassigned`, `ReassignStatus`, `OriginalSiteProb`, `NewSiteProb` and
`CandidateSiteProbs`.

## Same thing from the command line

```bash
ptmreloc optimize -i MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx --sheet MCF10A_phospho -o results/MCF10A_phospho
ptmreloc reassign -i MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx --sheet MCF10A_phospho \
    --weights results/MCF10A_phospho/weights.json -o results/MCF10A_phospho_reassigned.tsv
```
