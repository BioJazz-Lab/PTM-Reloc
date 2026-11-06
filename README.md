# ptmreloc

Prior-knowledge guided relocalization of low-confidence PTM (phosphorylation) sites on peptides.

When the MS2 fragment evidence cannot localize a site (low AScore), `ptmreloc` reassigns it
to the candidate residue best supported by PhosphoSitePlus (PSP). Each candidate site gets a score

```
S_F = w_H * S_H + w_L * S_L + w_D * S_D
```

* `S_H` – PSP high-throughput (MS) evidence, normalized within the peptide
* `S_L` – PSP low-throughput (literature) evidence, normalized within the peptide
* `S_D` – proximity to the originally assigned site

The top-n scoring sites (n = number of sites originally assigned) are kept for every
peptide with `AScore <= threshold`. Ties go to the original site, so peptides without
PSP evidence are left alone.

The weights are optimized against two independent references: **Gold Standard** sites (GS)
and **Functional Landscape** scores (FUN). The objective is a composite **Δ_REASSIGN**,
the geometric mean of the fraction of achievable GS improvement and the fraction of
achievable FUN improvement that a weight set realizes.

## Install

```bash
pip install -e .            # numpy, pandas, scipy
pip install -e ".[test]"    # + pytest
```

## Input

A peptide table with (column names configurable):

| column  | example                    |
|---------|----------------------------|
| UniProt | `Q93015`                   |
| Peptide | `-.MELILSTS*PAELTLDPACQPK.L` (PTM label after the residue; other mod symbols are preserved) |
| Residue | `S8s` (multi-site: `S8sT10t`); optional, see below |
| AScore  | `14.04`                    |

If there is no `Residue` column, protein positions are annotated by locating each peptide in
its protein sequence (first occurrence). This uses the bundled human proteome or your own
`fasta=`. Rows that cannot be located are passed through unchanged.

## Reference database

The package ships a ready-to-use human reference database (the `database/` folder,
installed as `ptmreloc._data`). Nothing needs to be preprocessed:

| dataset | file | used for |
|---|---|---|
| `psp`   | `phospho_site_plus_human_clustered_indexed_by_name.csv` | PhosphoSitePlus HTP/LTP evidence (scoring) |
| `gold`  | `gold_standard_sites_reindexed.tsv` | Gold Standard sites (optimization / evaluation) |
| `fun`   | `functional_landscape_sites_reindexed.tsv` | Functional Landscape scores (optimization / evaluation) |
| `fasta` | `UP000005640_9606.fasta` | UniProt human proteome (residue annotation) |

Each dataset can be replaced with your own, individually, as a file path or DataFrame
(pkl / csv / tsv / parquet / xlsx; raw PSP `.txt` downloads work):

* PSP: `UniProt` or `ACC_ID`, `MOD_RSD` (e.g. `S8-p`), `MS_LIT`, `MS_CST`, `LT_LIT`
* Gold Standard: `UniProt`, `Position in target protein`
* Functional Landscape: `UniProt`, `Position`, `Functional_score`
* FASTA: UniProt-style headers (`>sp|P12345|NAME_HUMAN`) or plain IDs

```python
pr.load_reference()                                   # everything bundled
pr.load_reference(psp="my_psp.txt")                   # own PSP, bundled GS/FUN
pr.load_reference(gold="my_gs.tsv", fun=False)        # bundled PSP, own GS, no FUN
pr.load_reference(database="path/to/other_db/")       # a folder with the same file names
```

The database folder can also be set with the `PTMRELOC_DATABASE` environment variable, and
`ptmreloc info` / `pr.database_info()` show which files are in use.

## Default weights

You don't have to run the optimization. Without explicit weights, `reassign` and `evaluate` use
the weights found by Monte Carlo sampling in `sampling/Optimizing_Reassignment.py`
(the median of the best 5% of trials) with an AScore threshold of 30:

| `DEFAULT_WEIGHTS` | w_H | w_L | w_D |
|---|---|---|---|
| `"medians"` (default) | 0.40936 | 0.0028834 | 0.0044392 |

Other presets you can pass by name: `"fun_medians"`, `"gs_medians"` (best 5% ranked by FUN
or by GS alone) and `"htp_only"` (1, 0, 0). They were derived with the default (`legacy`)
feature settings.

## Python library

```python
import ptmreloc as pr

# Tables may be pandas DataFrames or file paths. Reference data defaults to the bundled database.

# 1. reassign with the default weights - no optimization, no reference files needed
out = pr.reassign(peptides)                            # == pr.reassign(peptides, weights="medians")

# 2. optimize weights for your data (against bundled GS/FUN)
res = pr.optimize(peptides, n_samples=2000, threshold=30, out_dir="opt/")
print(res.summary())                                   # res.weights, res.trials, res.evaluation, res.by_ascore

# 3. reassign / evaluate with the optimized weights
out = pr.reassign(peptides, weights=res, output="reassigned.tsv")
rep = pr.evaluate(peptides, weights="opt/weights.json")
print(rep.summary())                                   # rep.overall (dict), rep.by_ascore (DataFrame)

# compare several weight sets on the same data (default: all presets)
pr.compare_weights(peptides, weight_sets={"default": None, "mine": res, "htp": "htp_only"})

# own reference data: pass any of psp= / gold= / fun= / fasta=, or a prebuilt reference
ref = pr.load_reference(psp="my_psp.txt")              # build once, reuse across calls
out = pr.reassign(peptides, ref)
```

| function | needs | returns |
|---|---|---|
| `reassign(peptides, psp=None, weights=None, threshold=None, ...)` | PSP | input table + relocalization columns |
| `optimize(peptides, psp=None, gold=None, fun=None, threshold=30, n_samples=2000, out_dir=None, ...)` | PSP, GS and/or FUN | `OptimizationResult` |
| `evaluate(peptides, psp=None, gold=None, fun=None, weights=None, threshold=None, bins=...)` | PSP, GS and/or FUN | `EvaluationReport` |
| `compare_weights(peptides, psp=None, gold=None, fun=None, weight_sets=None)` | PSP, GS and/or FUN | DataFrame |
| `load_reference(psp=None, gold=None, fun=None, database=None)` | | `ReferenceDB` |

`None` means the bundled dataset and `False` disables GS or FUN. All functions also take `fasta=`.

`weights` can be `None` (defaults), a preset name, a `weights.json` path, an `OptimizationResult`,
a dict `{"w_H", "w_L", "w_D"}` or a 3-tuple. A weights file or result also supplies its threshold
and feature settings, unless you override them with `threshold=` or `config=`. Column names
can be changed with `columns={"uniprot": ..., "peptide": ..., "residue": ..., "ascore": ...}`.

`reassign` appends `NewPeptide`, `NewResidue`, `Reassigned`, `ReassignStatus`
(`eligible` / `above_threshold` / `unparsed`), `OriginalSiteProb`, `NewSiteProb` and
`CandidateSiteProbs` (every candidate with its share of S_F).

Lower-level building blocks are exported too: `build_candidate_table`, `choose`,
`evaluate_selection`, `evaluate_by_ascore`, `optimize_weights` and `Reassigner`.

## Command line

```bash
ptmreloc info                                          # bundled database files, default weights

# reassign with the default weights and the bundled database
ptmreloc reassign -i peptides.tsv -o reassigned.tsv

# optimize weights (Monte Carlo + differential-evolution refinement, 20% proteins held out)
ptmreloc optimize -i peptides.tsv --threshold 30 --n-samples 2000 -o opt/
#   -> opt/weights.json, trials.csv, evaluation.csv, metrics_by_ascore.csv

# reassign with optimized weights, a preset, or explicit values
ptmreloc reassign -i peptides.tsv --weights opt/weights.json -o reassigned.tsv
ptmreloc reassign -i peptides.tsv --weights htp_only -o reassigned.tsv
ptmreloc reassign -i peptides.tsv --w 1 0 0.01 --threshold 20 -o reassigned.tsv

# evaluate weights (default if none given) per AScore bin
ptmreloc evaluate -i peptides.tsv

# own reference data: any of --psp / --gold / --fun / --fasta (or --database DIR);
# use --gold none / --fun none to leave one out
ptmreloc optimize -i peptides.tsv --psp my_psp.txt --gold my_gs.tsv --fun none -o opt/
```

## Optimization details

* **Search.** Weights are sampled log-uniformly in `10^[-3, 0]`, as in the manuscript.
  Differential evolution, seeded with the best samples, then refines the result. The
  objective is an arg-max and therefore piecewise constant, so the search is derivative-free.
* **Recommended weights.** These are the per-weight median of the top 5% Monte Carlo trials
  (the manuscript's "median of best 5%"). That is more robust than the single best trial,
  and `evaluation.csv` reports both on the training and held-out proteins.
* **Scale invariance.** S_F rankings are unchanged when all weights are multiplied by the
  same constant, so only the ratios matter. `normalized_weights` in `weights.json` sum to 1.
* **Metrics.** `delta_gs` and `delta_fun` are changes per site. `random_delta_*` is the
  exact expectation of uniform random reassignment, and `oracle_delta_*` is the best
  achievable change. `gs_gain` and `fun_gain` equal Δ / oracle Δ. `composite_trial_relative`
  in `trials.csv` reproduces the manuscript's across-trial min-max scaled version.
* **Speed.** Candidate features are built once (about 1 s for 120k PSMs). Each weight
  trial is then a vectorized group arg-max (a few ms), so thousands of trials take seconds.

### Feature options (`FeatureConfig` / CLI flags)

| option | `legacy` (default) | alternative |
|---|---|---|
| `evidence_transform` | `log10(log2(n+1)+1)`, sum-normalized (original scripts) | `log`: `log(n+1)`, `raw`: `n` (manuscript text) |
| `distance_mode` | `1 - d/(d_max+1)`, sum-normalized (original scripts) | `inverse`: `1/(d+1)`, zero when AScore = 0 (manuscript text) |

The defaults reproduce `sampling/Optimizing_Reassignment.py`. Note that in both distance
variants, multiplying by AScore cancels out once the score is normalized within the
peptide. AScore only enters through the threshold, plus the AScore = 0 case in `inverse` mode.
