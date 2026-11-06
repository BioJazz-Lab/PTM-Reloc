import json

import numpy as np
import pandas as pd
import pytest

import ptmreloc as pr
from ptmreloc.cli import main


# ---------------------------------------------------------------- peptide parsing
def test_parse_and_relabel_roundtrip():
    pp = pr.parse_peptide("K.AS*PTM@Y.R")
    assert pp.aas == "ASPTMY" and pp.labeled == [1]
    assert pr.relabel(pp, [1]) == "K.AS*PTM@Y.R"
    assert pr.relabel(pp, [5]) == "K.ASPTM@Y*.R"  # other mods preserved
    assert pr.parse_peptide("PEPS*").prefix == ""


def test_residue_roundtrip():
    assert pr.parse_residue("S8sT10t") == [("S", 8), ("T", 10)]
    assert pr.format_residue([("T", 10), ("S", 8)]) == "S8sT10t"
    with pytest.raises(ValueError):
        pr.parse_residue("nonsense")


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def reference():
    # protein P1: peptide "AS*AATAAY" starts at position 10 -> S11, T14, Y17
    psp = pd.DataFrame({
        "UniProt": ["P1", "P1", "P1", "P2", "P2"],
        "MOD_RSD": ["S11-p", "T14-p", "Y17-p", "S3-p", "S5-p"],
        "MS_LIT": [1, 50, 0, 10, 10], "MS_CST": [0, 10, 0, 0, 0], "LT_LIT": [0, 0, 20, 0, 0],
    })
    gold = pd.DataFrame({"UniProt": ["P1", "P2"], "Position in target protein": [14, 5]})
    fun = pd.DataFrame({"UniProt": ["P1", "P1", "P2"], "Position": [14, 11, 5], "Functional_score": [0.9, 0.1, 0.5]})
    return pr.ReferenceDB.from_sources(psp, gold, fun)


@pytest.fixture
def peptides():
    return pd.DataFrame({
        "UniProt": ["P1", "P1", "P1", "P2", "P3", "P2"],
        "Peptide": ["K.AS*AATAAY.R", "K.AS*AATAAY.R", "K.AS*AATAAY.R", "R.AS*AS*A.K", "K.S*AT.R", "R.AS*ASA.K"],
        "Residue": ["S11s", "S11s", "S11s", "S3sS5s", "S1s", "S3s"],
        "AScore": [0.0, 12.0, 500.0, 5.0, 3.0, 5.0],
    })


# ---------------------------------------------------------------- features / engine
def test_candidate_table(reference, peptides):
    t = pr.build_candidate_table(peptides, reference)
    assert t.n_rows == 6
    cf = t.candidate_frame()
    p1 = cf[cf.Peptide == "K.AS*AATAAY.R"].iloc[:3]  # AScore==0 rows form their own group
    assert p1.Site.tolist() == ["S11", "T14", "Y17"]
    assert np.isclose(p1.S_H.sum(), 1.0) and p1.S_H.idxmax() == p1.index[1]  # T14 has most HTP
    assert np.isclose(p1.S_L.sum(), 1.0) and p1.S_L.idxmax() == p1.index[2]  # Y17 has LTP
    assert p1.S_D.idxmax() == p1.index[0]  # original site is closest


def test_htp_weight_moves_site_and_ties_keep_original(reference, peptides):
    t = pr.build_candidate_table(peptides, reference)
    sel = pr.choose(t, [1, 0, 0])
    cf = t.candidate_frame()[sel]
    assert "T14" in cf[cf.Peptide == "K.AS*AATAAY.R"].Site.tolist()
    # P3 has no PSP evidence -> all-zero scores -> original kept
    assert cf[cf.UniProt == "P3"].Site.tolist() == ["S1"]
    # P2 mono-site: S3 and S5 have equal HTP -> tie -> original S3 kept
    assert cf[cf.Peptide == "R.AS*ASA.K"].Site.tolist() == ["S3"]
    # w_D only -> nothing moves
    assert (pr.choose(t, [0, 0, 1]) == t.c_orig).all()


def test_multisite_topn(reference):
    df = pd.DataFrame({"UniProt": ["P1"], "Peptide": ["K.AS*AAT*AAY.R"], "Residue": ["S11sT14t"], "AScore": [1.0]})
    t = pr.build_candidate_table(df, reference)
    sel = pr.choose(t, [0, 1, 0])  # LTP favours Y17; other site tie between S11/T14 -> keep originals
    assert sorted(t.candidate_frame()[sel].Site) == ["S11", "Y17"]  # tie -> sequence order
    out = pr.apply_selection(df, t, [0, 1, 0], threshold=30)
    assert out.NewPeptide[0] == "K.AS*AATAAY*.R" and out.NewResidue[0] == "S11sY17y"


def test_metrics_and_oracle(reference, peptides):
    t = pr.build_candidate_table(peptides, reference)
    low = t.r_ascore <= 30
    m0 = pr.evaluate_selection(t, t.c_orig, low)
    assert m0["delta_gs"] == 0 and m0["n_rows_changed"] == 0
    m = pr.evaluate_selection(t, pr.choose(t, [1, 0, 0]), low)
    assert m["delta_gs"] > 0 and m["delta_fun"] > 0
    assert m["gs_gain"] <= 1 + 1e-12 and m["fun_gain"] <= 1 + 1e-12
    assert m["oracle_delta_gs"] >= m["delta_gs"]
    assert pr.composite_score(0.5, -0.1) == -0.1
    assert np.isclose(pr.composite_score(0.25, 1.0), 0.5)


def test_unparsable_rows_pass_through(reference):
    df = pd.DataFrame({"UniProt": ["P1", "P1"], "Peptide": ["K.AS*AATAAY.R", "K.AS*AATAAY.R"],
                       "Residue": ["S11s", "T11t"], "AScore": [1.0, 1.0]})  # 2nd: residue mismatch
    out = pr.Reassigner(reference, [1, 0, 0]).reassign(df)
    assert out.ReassignStatus.tolist() == ["eligible", "unparsed"]
    assert out.NewPeptide.tolist() == ["K.ASAAT*AAY.R", "K.AS*AATAAY.R"]
    assert out.NewResidue.tolist()[0] == "T14t"


def test_reassigner_threshold(reference, peptides):
    out = pr.Reassigner(reference, {"w_H": 1, "w_L": 0, "w_D": 0}, threshold=30).reassign(peptides)
    assert out.loc[2, "ReassignStatus"] == "above_threshold"
    assert out.loc[2, "NewPeptide"] == "K.AS*AATAAY.R"
    assert out.loc[0, "NewPeptide"] == "K.ASAAT*AAY.R" and out.loc[0, "Reassigned"]
    assert out.loc[3, "NewPeptide"] == "R.AS*AS*A.K"  # both sites already assigned


# ---------------------------------------------------------------- optimization
def _synthetic(n_prot=60, seed=1):
    """Peptides whose true (GS / high-FUN) site is the one with most HTP evidence."""
    rng = np.random.default_rng(seed)
    psp, gold, fun, rows = [], [], [], []
    for k in range(n_prot):
        acc = f"Q{k}"
        seq = "AASAATAASAAK"  # S3 T6 S9 -> protein positions offset+...
        cands = [2, 5, 8]
        true = rng.choice(cands)
        for c in cands:
            pos = 100 + c + 1
            htp = 40 if c == true else rng.integers(0, 5)
            ltp = rng.integers(0, 30)  # uninformative
            psp.append({"UniProt": acc, "MOD_RSD": f"{seq[c]}{pos}-p", "MS_LIT": htp, "MS_CST": 0, "LT_LIT": ltp})
            fun.append({"UniProt": acc, "Position": pos, "Functional_score": 0.8 if c == true else 0.1})
        gold.append({"UniProt": acc, "Position in target protein": 100 + true + 1})
        assigned = rng.choice(cands)
        pep = "K." + "".join(a + ("*" if i == assigned else "") for i, a in enumerate(seq)) + ".R"
        rows.append({"UniProt": acc, "Peptide": pep, "Residue": f"{seq[assigned]}{100 + assigned + 1}{seq[assigned].lower()}",
                     "AScore": float(rng.uniform(0, 25))})
    ref = pr.ReferenceDB.from_sources(pd.DataFrame(psp), pd.DataFrame(gold), pd.DataFrame(fun))
    return ref, pd.DataFrame(rows)


def test_optimize_prefers_htp():
    ref, df = _synthetic()
    t = pr.build_candidate_table(df, ref)
    res = pr.optimize_weights(t, threshold=30, n_samples=200, refine=True, refine_maxiter=5,
                              holdout_frac=0.25, seed=0)
    w = pr.normalize_weights(res.weights)
    assert w["w_H"] > w["w_L"] and w["w_H"] > w["w_D"]
    rec = res.evaluation.query("weights == 'recommended' and split == 'all'").iloc[0]
    assert rec["gs_gain"] > 0.9 and rec["composite"] > 0.9
    assert {"monte_carlo", "refine", "anchor_htp_only"} <= set(res.trials.source)


def test_cli_end_to_end(tmp_path):
    ref_src, df = _synthetic(n_prot=30)
    # write inputs
    rng = np.random.default_rng(0)
    psp = pd.DataFrame([{"UniProt": u, "MOD_RSD": f"X{p}-p", "MS_LIT": h, "MS_CST": 0, "LT_LIT": ref_src.ltp[(u, p)]}
                        for (u, p), h in ref_src.htp.items()])
    gold = pd.DataFrame([{"UniProt": u, "Position in target protein": p} for u, p in ref_src.gold])
    fun = pd.DataFrame([{"UniProt": u, "Position": p, "Functional_score": s} for (u, p), s in ref_src.fun.items()])
    for name, d in [("pep", df), ("psp", psp), ("gold", gold), ("fun", fun)]:
        d.to_csv(tmp_path / f"{name}.csv", index=False)
    common = ["-i", str(tmp_path / "pep.csv"), "--psp", str(tmp_path / "psp.csv")]
    targets = ["--gold", str(tmp_path / "gold.csv"), "--fun", str(tmp_path / "fun.csv")]
    assert main(["optimize", *common, *targets, "-o", str(tmp_path / "opt"), "--n-samples", "50",
                 "--no-refine", "--holdout", "0"]) == 0
    payload = json.loads((tmp_path / "opt" / "weights.json").read_text())
    assert set(payload["weights"]) == {"w_H", "w_L", "w_D"}
    assert main(["reassign", *common, "--weights", str(tmp_path / "opt" / "weights.json"),
                 "-o", str(tmp_path / "out.tsv")]) == 0
    out = pd.read_csv(tmp_path / "out.tsv", sep="\t")
    assert {"NewPeptide", "NewResidue", "Reassigned", "CandidateSiteProbs"} <= set(out.columns)
    assert main(["evaluate", *common, *targets, "--w", "1", "0", "0", "--threshold", "30"]) == 0


# ---------------------------------------------------------------- high-level API
def test_default_weights_match_sampling_script():
    assert pr.DEFAULT_WEIGHTS == {"w_H": 0.40935512896496834, "w_L": 0.0028834480249440245,
                                  "w_D": 0.004439231598796614}
    assert pr.DEFAULT_THRESHOLD == 30.0
    assert pr.WEIGHT_PRESETS["medians"] == pr.DEFAULT_WEIGHTS


def test_resolve_weights_variants(tmp_path):
    assert pr.resolve_weights(None)[0] == pr.DEFAULT_WEIGHTS
    assert pr.resolve_weights("HTP_only")[0] == {"w_H": 1.0, "w_L": 0.0, "w_D": 0.0}
    assert pr.resolve_weights([1, 2, 3])[0] == {"w_H": 1.0, "w_L": 2.0, "w_D": 3.0}
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"weights": {"w_H": 1, "w_L": 0, "w_D": 0.5}, "threshold": 20,
                             "feature_config": {"distance_mode": "inverse"}}))
    w, thr, cfg = pr.resolve_weights(str(f))
    assert w["w_D"] == 0.5 and thr == 20 and cfg.distance_mode == "inverse"
    with pytest.raises(ValueError):
        pr.resolve_weights("no_such_preset")


def test_api_reassign_defaults(reference, peptides):
    out = pr.reassign(peptides, reference)  # default weights, threshold 30
    assert out.loc[0, "NewPeptide"] == "K.ASAAT*AAY.R"
    assert out.loc[2, "ReassignStatus"] == "above_threshold"
    same = pr.Reassigner(reference).reassign(peptides)
    pd.testing.assert_frame_equal(out, same)
    # threshold override
    assert not pr.reassign(peptides, reference, threshold=-1)["Reassigned"].any()


def test_api_optimize_evaluate_compare(tmp_path):
    ref, df = _synthetic()
    res = pr.optimize(df, ref, n_samples=100, refine=False, holdout_frac=0, out_dir=str(tmp_path / "opt"))
    assert (tmp_path / "opt" / "weights.json").exists()
    rep = pr.evaluate(df, ref, weights=str(tmp_path / "opt" / "weights.json"))
    assert rep.threshold == 30 and rep.overall["composite"] > 0.9
    assert "ascore_bin" in rep.by_ascore.columns and "composite" in rep.summary()
    rep2 = pr.evaluate(df, ref, weights=res)
    assert rep2.overall == rep.overall
    out = pr.reassign(df, ref, weights=res, output=str(tmp_path / "out.csv"))
    assert (tmp_path / "out.csv").exists() and out["Reassigned"].any()
    cmp_ = pr.compare_weights(df, ref)
    assert set(cmp_.weights) == set(pr.WEIGHT_PRESETS)


def test_api_requires_targets(reference, peptides):
    psp_only = pr.ReferenceDB(htp=reference.htp, ltp=reference.ltp)
    with pytest.raises(ValueError):
        pr.evaluate(peptides, psp_only)


def test_load_reference_extends_copy(reference):
    base = pr.ReferenceDB(htp=reference.htp, ltp=reference.ltp)
    ext = pr.load_reference(base, gold=pd.DataFrame({"UniProt": ["P9"], "Position in target protein": [1]}))
    assert ("P9", 1) in ext.gold and not base.gold


def test_cli_reassign_default_weights(tmp_path, reference, peptides):
    psp = pd.DataFrame([{"UniProt": u, "MOD_RSD": f"X{p}-p", "MS_LIT": h, "MS_CST": 0,
                         "LT_LIT": reference.ltp[(u, p)]} for (u, p), h in reference.htp.items()])
    peptides.to_csv(tmp_path / "pep.csv", index=False)
    psp.to_csv(tmp_path / "psp.csv", index=False)
    common = ["-i", str(tmp_path / "pep.csv"), "--psp", str(tmp_path / "psp.csv")]
    assert main(["reassign", *common, "-o", str(tmp_path / "a.tsv")]) == 0
    assert main(["reassign", *common, "--weights", "htp_only", "-o", str(tmp_path / "b.tsv")]) == 0
    a = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    assert a.loc[0, "NewPeptide"] == "K.ASAAT*AAY.R"


# ---------------------------------------------------------------- bundled database / own data
def _has_db():
    try:
        pr.database_dir()
        return True
    except FileNotFoundError:
        return False


needs_db = pytest.mark.skipif(not _has_db(), reason="bundled database not available")


@needs_db
def test_bundled_reference_defaults():
    ref = pr.load_reference()
    assert len(ref.htp) > 100_000 and ref.gold and ref.fun
    assert pr.load_reference() is ref  # cached
    psp_only = pr.load_reference(gold=False, fun=False)
    assert psp_only.htp and not psp_only.gold and not psp_only.fun
    assert set(pr.database_info().dataset) == {"psp", "gold", "fun", "fasta"}


@needs_db
def test_bundled_reassign_and_evaluate():
    df = pd.DataFrame({"UniProt": ["Q93015"], "Peptide": ["-.MELILSTS*PAELTLDPACQPK.L"],
                       "Residue": ["S8s"], "AScore": [14.0]})
    out = pr.reassign(df)  # bundled PSP + default weights
    assert out.ReassignStatus[0] == "eligible"
    assert pr.evaluate(df).overall["n_rows"] == 1
    # residue column missing -> annotated from the bundled proteome
    out2 = pr.reassign(df.drop(columns="Residue"))
    assert out2.NewPeptide[0] == out.NewPeptide[0] and out2.Residue[0] == "S8s"


@needs_db
def test_own_tables_override_bundled(reference, peptides):
    own_psp = pd.DataFrame({"UniProt": ["P1"], "MOD_RSD": ["T14-p"], "MS_LIT": [5], "MS_CST": [0], "LT_LIT": [0]})
    ref = pr.load_reference(psp=own_psp)  # own PSP, bundled GS/FUN
    assert ref.htp == {("P1", 14): 5.0} and len(ref.gold) > 1000
    out = pr.reassign(peptides, psp=own_psp)
    assert out.loc[0, "NewPeptide"] == "K.ASAAT*AAY.R"


def test_annotate_residues_with_own_fasta(tmp_path):
    fa = tmp_path / "p.fasta"
    fa.write_text(">sp|P1|PROT1_HUMAN test\nMMMMMMMMMMAS\nAATAAYKK\n>other\nSSS\n")
    seqs = pr.read_fasta(fa)
    assert seqs["P1"] == seqs["PROT1_HUMAN"] == "MMMMMMMMMMASAATAAYKK" and seqs["other"] == "SSS"
    df = pd.DataFrame({"UniProt": ["P1", "PX"], "Peptide": ["K.AS*AAT*AAY.R", "K.S*S.R"]})
    with pytest.warns(UserWarning):
        ann = pr.annotate_residues(df, fa)
    assert ann.Residue.tolist() == ["S12sT15t", ""]


def test_database_dir_env_var(tmp_path, monkeypatch, reference):
    (tmp_path / pr.DATABASE_FILES["psp"]).write_text("UniProt,MOD_RSD,MS_LIT,MS_CST,LT_LIT\nP1,S11-p,99,0,0\n")
    monkeypatch.setenv("PTMRELOC_DATABASE", str(tmp_path))
    assert pr.database_dir() == tmp_path
    ref = pr.load_reference(gold=False, fun=False)
    assert ref.htp == {("P1", 11): 99.0}


def test_cli_info():
    if _has_db():
        assert main(["info"]) == 0
