# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: Finding the optimized parameters for reassignment
# format: html
# ---

# %%
from utils import *

import random
random.seed(int(datetime.datetime.now().timestamp()*1e3))

warnings.filterwarnings("ignore")
# home = str(Path.home())
plt.rcParams['figure.figsize'] = [7, 5] 
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
plt.style.use('default')

# %%
from sklearn import preprocessing as preproc

# %% [markdown]
# ## Loading input data

# %%
# Load the data
# fasta_file = home + "/Dropbox/Data/Uniprot/Human/UP000005640_9606.fasta"
fasta_file = "./UP000005640_9606.fasta"
proteome_seqs = {uniprot_item.id.split('|')[2] : str(uniprot_item.seq) for uniprot_item in SeqIO.parse(fasta_file, "fasta")}
PSP_sites = pd.read_pickle("./phospho_site_plus_human_clustered_indexed_by_name.pkl")
GS_sites = pd.read_pickle("./gold_standard_sites_reindexed.pkl")
FUN_sites = pd.read_pickle("./functional_landscape_sites_reindexed.pkl")
results_dir = "./"

# mcf10a_ts_pept = pd.read_excel("../data/_phospho_proteomics/MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx", sheet_name="MCF10A_phospho")
# hs578t_ts_pept = pd.read_excel("../data/_phospho_proteomics/MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx", sheet_name="HS578T_phospho")

# %%
# mcf10a_ts_pept = pd.read_excel("./Optimizing_Reassignment/MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx", sheet_name="MCF10A_phospho")
# df = mcf10a_ts_pept[["UniProt", "Protein", "Peptide", "Residue", "AScore"]].copy()
# hs578t_ts_pept = pd.read_excel("./Optimizing_Reassignment/MCF10A_HS578T_EGF_timecourse_data_20211112.xlsx", sheet_name="HS578T_phospho")
# df = hs578t_ts_pept[["UniProt", "Protein", "Peptide", "Residue", "AScore"]].copy()
df = pd.read_pickle("./Optimizing_Reassignment/df4reassignment.pkl")


# %%
def reassignment_summary(params, df = df, ptm_label = '*', PSP_sites = PSP_sites, GS_sites = GS_sites, FUN_sites = FUN_sites):
# def reassignment_summary(thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W, df = df, ptm_label = '*', PSP_sites = PSP_sites, GS_sites = GS_sites, FUN_sites = FUN_sites):

    (thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W) = params
    ids = list(set(df["UniProt"]))

    bad_pept_num = 0
    bad_site_num = 0

    protein_names = []
    uniprot_ids = []
    pho_num = []
    ascores = []
    old_pepts = []
    # old_sites = []
    old_gs_nums = []
    old_gs_scores = [] 
    old_fun_scores = []
    old_psp_ms_scores = []
    old_psp_lt_scores = []
    old_psp_scores = []
    old_exp_scores = []
    old_dist_scores = []
    old_comb_scores = []
    new_pepts = []
    # new_sites = []
    new_gs_nums = []
    new_gs_scores = []
    new_fun_scores = []
    new_psp_ms_scores = []
    new_psp_lt_scores = []
    new_psp_scores = []
    new_exp_scores = []
    new_dist_scores = []
    new_comb_scores = []
    gs_p2p = []
    gs_p2n = []
    gs_n2p = []
    gs_n2n = []
    p2p = []
    p2n = []
    n2p = []
    n2n = []
    for idi in ids:
        dfi = df[df["UniProt"] == idi].copy()
        prot_name = dfi["Protein"].to_list()[0]
        pepts = dfi["Peptide"].to_list()
        reses = dfi["Residue"].to_list()
        dfi_ascore = dfi["AScore"].to_list()
        poses = get_res_pos(reses)
        dfi["PhosphoNum"] = [len(res) for res in poses]
        bad_dfi = dfi[dfi["AScore"] <= thr].copy()
        bad_pept_num += bad_dfi.shape[0]
        if bad_dfi.shape[0] > 0:
            good_dfi = dfi[(dfi["AScore"] > thr) & (dfi["PhosphoNum"] <= 1)].copy()
            goodi = get_phosphositeplus_pos(good_dfi["Residue"])
            bad_pepts = bad_dfi["Peptide"].to_list()
            bad_reses = bad_dfi["Residue"].to_list()
            bad_ascores = bad_dfi["AScore"].to_list()
            bad_poses = get_res_pos(bad_reses)
            bad_site_num += np.sum([len(res) for res in bad_poses])
            bad_names = get_res_names(bad_reses)
            pspi = PSP_sites[PSP_sites["UniProt"] == idi].copy()
            gsi = GS_sites[GS_sites["UniProt"] == idi].copy()
            funi = FUN_sites[FUN_sites["UniProt"] == idi].copy()
            pspi_ids = list(set(pspi["Site"]))
            gsi_ids = list(set(gsi["Site"]))
            funi_ids = list(set(funi["Site"]))
            gsi_poses = list(set(gsi["Position in target protein"]))
            funi_poses = list(set(funi["Position"]))
            if pspi.shape[0] > 0:
                # pspevi = np.array([list(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0)), list(pspi["LT_LIT"].fillna(0))]).transpose()
                pspevi = np.array([list(np.log2(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0) + 1)), list(np.log2(pspi["LT_LIT"].fillna(0) + 1))]).transpose()
                psppos = get_phosphositeplus_pos(pspi["MOD_RSD"])
                pspthis = [1 if i in goodi else 0 for i in psppos]
                for i, res in enumerate(bad_poses):
                    ptm_info = get_ptm_info(bad_pepts[i], bad_reses[i])
                    possible_ptm_pos = [ptm[0] for ptm in ptm_info]
                    possible_ptm_scores = []
                    possible_exp_scores = []
                    possible_gs_nums = []
                    possible_gs_scores = []
                    possible_fun_scores = []
                    for ptm_pos in possible_ptm_pos:
                        if ptm_pos in psppos:
                            possible_ptm_scores.append(np.log10(pspevi[psppos.index(ptm_pos)] + 1))
                            possible_exp_scores.append(pspthis[psppos.index(ptm_pos)])
                        elif ptm_pos in goodi:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(1)
                        else:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(0)
                        if ptm_pos in gsi_poses:
                            possible_gs_nums.append(1)
                        else:
                            possible_gs_nums.append(0)
                        if ptm_pos in funi_poses:
                            possible_fun_scores.append(funi[funi["Position"] == ptm_pos]["Functional_score"].to_list()[0])
                        else:
                            possible_fun_scores.append(0)
                    possible_ptm_scores /= (np.array(possible_ptm_scores).sum(axis=0, keepdims=True) + 1e-12)
                    possible_exp_scores = np.array(possible_exp_scores)
                    true_res_pos = get_ptm_pos_in_pept(bad_pepts[i], ptm_label)
                    ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
                    ptm_dist_scores = bad_ascores[i] * (1 - (ptm_dists/ (ptm_dists.max() + 1)))
                    ptm_dist_scores /= (ptm_dist_scores.sum() + 1e-12)
                    ptm_gs_nums = np.array(possible_gs_nums)
                    ptm_gs_scores = np.array(possible_gs_nums) / (np.array(possible_gs_nums).sum() + 1e-12)
                    ptm_fun_scores = np.array(possible_fun_scores) / (np.array(possible_fun_scores).sum() + 1e-12)
                    ptm_final_scores = possible_ptm_scores[:,0] * MS_W + possible_ptm_scores[:,1] * LT_W + ptm_dist_scores * DT_W + possible_exp_scores * MP_W + ptm_gs_scores * GS_W + ptm_fun_scores * FL_W
                    num_ptm = len(res)
                    if ptm_final_scores.sum() == 0.0:
                    #     print(bad_pepts[i] + " does not have PSP sites!")
                        continue
                    new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
                    new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
                    new_residue = []
                    new_res_pos = []
                    for new_ptm in new_ptm_info:
                        new_residue.append(new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower())
                        new_res_pos.append(new_ptm[2])
                    new_peptide = relable_pept(bad_pepts[i], sorted(new_res_pos), ptm_label)
                    new_psp_ms_score = np.sum([possible_ptm_scores[k][0] for k in new_ptm_indices])
                    new_psp_ms_scores.append(new_psp_ms_score) 
                    new_psp_lt_score = np.sum([possible_ptm_scores[k][1] for k in new_ptm_indices])
                    new_psp_lt_scores.append(new_psp_lt_score) 
                    new_psp_score = np.sum([possible_ptm_scores[k][0] * MS_W + possible_ptm_scores[k][1] * LT_W for k in new_ptm_indices])
                    new_psp_scores.append(new_psp_score)
                    new_exp_score = np.sum([possible_exp_scores[k] for k in new_ptm_indices])
                    new_exp_scores.append(new_exp_score)
                    new_dist_score = np.sum([ptm_dist_scores[k] for k in new_ptm_indices])
                    new_dist_scores.append(new_dist_score)
                    new_gs_num = np.sum([ptm_gs_nums[k] for k in new_ptm_indices])
                    new_gs_nums.append(new_gs_num)
                    new_gs_score = np.sum([ptm_gs_scores[k] for k in new_ptm_indices])
                    new_gs_scores.append(new_gs_score)
                    new_fun_score = np.sum([ptm_fun_scores[k] for k in new_ptm_indices])
                    new_fun_scores.append(new_fun_score)
                    new_comb_score = np.sum([ptm_final_scores[k] for k in new_ptm_indices])
                    new_comb_scores.append(new_comb_score)
                    ascores.append(bad_ascores[i])
                    old_psp_ms_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] for j in range(num_ptm)])
                    old_psp_ms_scores.append(old_psp_ms_score)
                    old_psp_lt_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][1] for j in range(num_ptm)])
                    old_psp_lt_scores.append(old_psp_lt_score)
                    old_psp_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W for j in range(num_ptm)])
                    old_psp_scores.append(old_psp_score)
                    old_exp_score = np.sum([possible_exp_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_exp_scores.append(old_exp_score)
                    old_dist_score = np.sum([ptm_dist_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_dist_scores.append(old_dist_score)
                    old_gs_num = np.sum([ptm_gs_nums[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_gs_nums.append(old_gs_num)
                    old_gs_score = np.sum([ptm_gs_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_gs_scores.append(old_gs_score)
                    old_fun_score = np.sum([ptm_fun_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_fun_scores.append(old_fun_score)
                    old_comb_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W + possible_exp_scores[possible_ptm_pos.index(res[j])] * MP_W + ptm_dist_scores[possible_ptm_pos.index(res[j])] * DT_W + ptm_gs_scores[possible_ptm_pos.index(res[j])] * GS_W + ptm_fun_scores[possible_ptm_pos.index(res[j])] * FL_W for j in range(num_ptm)])
                    old_comb_scores.append(old_comb_score)
                    
                    if old_exp_score < new_exp_score:
                        p2p.append(old_exp_score)
                        n2p.append(new_exp_score - old_exp_score)
                        p2n.append(0)
                        n2n.append(num_ptm - new_exp_score)
                    else:
                        p2p.append(new_exp_score)
                        n2p.append(0)
                        p2n.append(old_exp_score - new_exp_score)
                        n2n.append(num_ptm - old_exp_score)

                    if old_gs_num < new_gs_num:
                        gs_p2p.append(old_gs_num)
                        gs_n2p.append(new_gs_num - old_gs_num)
                        gs_p2n.append(0)
                        gs_n2n.append(num_ptm - new_gs_num)
                    else:
                        gs_p2p.append(new_gs_num)
                        gs_n2p.append(0)
                        gs_p2n.append(old_gs_num - new_gs_num)
                        gs_n2n.append(num_ptm - old_gs_num)

                    protein_names.append(prot_name)
                    uniprot_ids.append(idi)
                    pho_num.append(num_ptm)
                    old_pepts.append(bad_pepts[i])
                    # old_sites.append(old_site)
                    new_pepts.append(new_peptide)
                    # new_sites.append(new_site)
                    
    reassignment = pd.DataFrame({
        "Protein": protein_names,
        "UniProt": uniprot_ids,
        "PhoNum": pho_num,
        "OldPeptide": old_pepts,
        "NewPeptide": new_pepts,
        # "OldSite": old_sites,
        # "NewSite": new_sites,
        "AScore": ascores,
        "OldGsNum": old_gs_nums,
        "NewGsNum": new_gs_nums,
        "OldGsScore": old_gs_scores,
        "NewGsScore": new_gs_scores,
        "OldFunScore": old_fun_scores,
        "NewFunScore": new_fun_scores,
        "OldPspMsScore": old_psp_ms_scores,
        "NewPspMsScore": new_psp_ms_scores,
        "OldPspLtScore": old_psp_lt_scores,
        "NewPspLtScore": new_psp_lt_scores,
        "OldPspScore": old_psp_scores,
        "NewPspScore": new_psp_scores,
        "OldExpScore": old_exp_scores,
        "NewExpScore": new_exp_scores,
        "OldDistScore": old_dist_scores,
        "NewDistScore": new_dist_scores,
        "OldCombScore": old_comb_scores,
        "NewCombScore": new_comb_scores,
        "P2P": p2p,
        "N2P": n2p,
        "P2N": p2n,
        "N2N": n2n,
        "GS_P2P": gs_p2p,
        "GS_N2P": gs_n2p,
        "GS_P2N": gs_p2n,
        "GS_N2N": gs_n2n,
    })

    # Option: save the reassignment summary to a file
    # if save_file:
    #     # reassignment.to_csv(save_file + ".tsv", sep="\t", index=False)
    #     reassignment.to_pickle(save_file + ".pkl")

    # Now return a summarized list of sums of individual column
    summary = [thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W, np.sum(reassignment["AScore"]), bad_pept_num, bad_site_num, np.sum(reassignment["PhoNum"]), np.sum(reassignment["OldGsNum"]), np.sum(reassignment["NewGsNum"]), np.sum(reassignment["OldGsScore"]), np.sum(reassignment["NewGsScore"]), np.sum(reassignment["OldFunScore"]), np.sum(reassignment["NewFunScore"]), np.sum(reassignment["OldPspMsScore"]), np.sum(reassignment["NewPspMsScore"]), np.sum(reassignment["OldPspLtScore"]), np.sum(reassignment["NewPspLtScore"]), np.sum(reassignment["OldPspScore"]), np.sum(reassignment["NewPspScore"]), np.sum(reassignment["OldExpScore"]), np.sum(reassignment["NewExpScore"]), np.sum(reassignment["OldDistScore"]), np.sum(reassignment["NewDistScore"]), np.sum(reassignment["OldCombScore"]), np.sum(reassignment["NewCombScore"]), np.sum(reassignment["P2P"]), np.sum(reassignment["N2P"]), np.sum(reassignment["P2N"]), np.sum(reassignment["N2N"]), np.sum(reassignment["GS_P2P"]), np.sum(reassignment["GS_N2P"]), np.sum(reassignment["GS_P2N"]), np.sum(reassignment["GS_N2N"])]

    # return(reassignment, summary)
    return(summary)



def reassignment(params, df = df, ptm_label = '*', PSP_sites = PSP_sites, GS_sites = GS_sites, FUN_sites = FUN_sites):
# def reassignment_summary(thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W, df = df, ptm_label = '*', PSP_sites = PSP_sites, GS_sites = GS_sites, FUN_sites = FUN_sites):

    (thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W) = params
    ids = list(set(df["UniProt"]))

    bad_pept_num = 0
    bad_site_num = 0

    protein_names = []
    uniprot_ids = []
    pho_num = []
    ascores = []
    old_pepts = []
    # old_sites = []
    old_gs_nums = []
    old_gs_scores = []
    old_fun_scores = []
    old_psp_ms_scores = []
    old_psp_lt_scores = []
    old_psp_scores = []
    old_exp_scores = []
    old_dist_scores = []
    old_comb_scores = []
    new_pepts = []
    # new_sites = []
    new_gs_nums = []
    new_gs_scores = []
    new_fun_scores = []
    new_psp_ms_scores = []
    new_psp_lt_scores = []
    new_psp_scores = []
    new_exp_scores = []
    new_dist_scores = []
    new_comb_scores = []
    gs_p2p = []
    gs_p2n = []
    gs_n2p = []
    gs_n2n = []
    p2p = []
    p2n = []
    n2p = []
    n2n = []
    for idi in ids:
        dfi = df[df["UniProt"] == idi].copy()
        prot_name = dfi["Protein"].to_list()[0]
        pepts = dfi["Peptide"].to_list()
        reses = dfi["Residue"].to_list()
        dfi_ascore = dfi["AScore"].to_list()
        poses = get_res_pos(reses)
        dfi["PhosphoNum"] = [len(res) for res in poses]
        bad_dfi = dfi[dfi["AScore"] <= thr].copy()
        bad_pept_num += bad_dfi.shape[0]
        if bad_dfi.shape[0] > 0:
            good_dfi = dfi[(dfi["AScore"] > thr) & (dfi["PhosphoNum"] <= 1)].copy()
            goodi = get_phosphositeplus_pos(good_dfi["Residue"])
            bad_pepts = bad_dfi["Peptide"].to_list()
            bad_reses = bad_dfi["Residue"].to_list()
            bad_ascores = bad_dfi["AScore"].to_list()
            bad_poses = get_res_pos(bad_reses)
            bad_site_num += np.sum([len(res) for res in bad_poses])
            bad_names = get_res_names(bad_reses)
            pspi = PSP_sites[PSP_sites["UniProt"] == idi].copy()
            gsi = GS_sites[GS_sites["UniProt"] == idi].copy()
            funi = FUN_sites[FUN_sites["UniProt"] == idi].copy()
            pspi_ids = list(set(pspi["Site"]))
            gsi_ids = list(set(gsi["Site"]))
            funi_ids = list(set(funi["Site"]))
            gsi_poses = list(set(gsi["Position in target protein"]))
            funi_poses = list(set(funi["Position"]))
            if pspi.shape[0] > 0:
                # pspevi = np.array([list(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0)), list(pspi["LT_LIT"].fillna(0))]).transpose()
                pspevi = np.array([list(np.log2(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0) + 1)), list(np.log2(pspi["LT_LIT"].fillna(0) + 1))]).transpose()
                psppos = get_phosphositeplus_pos(pspi["MOD_RSD"])
                pspthis = [1 if i in goodi else 0 for i in psppos]
                for i, res in enumerate(bad_poses):
                    ptm_info = get_ptm_info(bad_pepts[i], bad_reses[i])
                    possible_ptm_pos = [ptm[0] for ptm in ptm_info]
                    possible_ptm_scores = []
                    possible_exp_scores = []
                    possible_gs_nums = []
                    possible_gs_scores = []
                    possible_fun_scores = []
                    for ptm_pos in possible_ptm_pos:
                        if ptm_pos in psppos:
                            possible_ptm_scores.append(np.log10(pspevi[psppos.index(ptm_pos)] + 1))
                            possible_exp_scores.append(pspthis[psppos.index(ptm_pos)])
                        elif ptm_pos in goodi:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(1)
                        else:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(0)
                        if ptm_pos in gsi_poses:
                            possible_gs_nums.append(1)
                        else:
                            possible_gs_nums.append(0)
                        if ptm_pos in funi_poses:
                            possible_fun_scores.append(funi[funi["Position"] == ptm_pos]["Functional_score"].to_list()[0])
                        else:
                            possible_fun_scores.append(0)
                    possible_ptm_scores /= (np.array(possible_ptm_scores).sum(axis=0, keepdims=True) + 1e-12)
                    possible_exp_scores = np.array(possible_exp_scores)
                    true_res_pos = get_ptm_pos_in_pept(bad_pepts[i], ptm_label)
                    ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
                    ptm_dist_scores = bad_ascores[i] * (1 - (ptm_dists/ (ptm_dists.max() + 1)))
                    ptm_dist_scores /= (ptm_dist_scores.sum() + 1e-12)
                    ptm_gs_nums = np.array(possible_gs_nums)
                    ptm_gs_scores = np.array(possible_gs_nums) / (np.array(possible_gs_nums).sum() + 1e-12)
                    ptm_fun_scores = np.array(possible_fun_scores) / (np.array(possible_fun_scores).sum() + 1e-12)
                    ptm_final_scores = possible_ptm_scores[:,0] * MS_W + possible_ptm_scores[:,1] * LT_W + ptm_dist_scores * DT_W + possible_exp_scores * MP_W + ptm_gs_scores * GS_W + ptm_fun_scores * FL_W
                    num_ptm = len(res)
                    if ptm_final_scores.sum() == 0.0:
                    #     print(bad_pepts[i] + " does not have PSP sites!")
                        continue
                    new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
                    new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
                    new_residue = []
                    new_res_pos = []
                    for new_ptm in new_ptm_info:
                        new_residue.append(new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower())
                        new_res_pos.append(new_ptm[2])
                    new_peptide = relable_pept(bad_pepts[i], sorted(new_res_pos), ptm_label)
                    new_psp_ms_score = np.sum([possible_ptm_scores[k][0] for k in new_ptm_indices])
                    new_psp_ms_scores.append(new_psp_ms_score) 
                    new_psp_lt_score = np.sum([possible_ptm_scores[k][1] for k in new_ptm_indices])
                    new_psp_lt_scores.append(new_psp_lt_score) 
                    new_psp_score = np.sum([possible_ptm_scores[k][0] * MS_W + possible_ptm_scores[k][1] * LT_W for k in new_ptm_indices])
                    new_psp_scores.append(new_psp_score)
                    new_exp_score = np.sum([possible_exp_scores[k] for k in new_ptm_indices])
                    new_exp_scores.append(new_exp_score)
                    new_dist_score = np.sum([ptm_dist_scores[k] for k in new_ptm_indices])
                    new_dist_scores.append(new_dist_score)
                    new_gs_num = np.sum([ptm_gs_nums[k] for k in new_ptm_indices])
                    new_gs_nums.append(new_gs_num)
                    new_gs_score = np.sum([ptm_gs_scores[k] for k in new_ptm_indices])
                    new_gs_scores.append(new_gs_score)
                    new_fun_score = np.sum([ptm_fun_scores[k] for k in new_ptm_indices])
                    new_fun_scores.append(new_fun_score)
                    new_comb_score = np.sum([ptm_final_scores[k] for k in new_ptm_indices])
                    new_comb_scores.append(new_comb_score)
                    ascores.append(bad_ascores[i])
                    old_psp_ms_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] for j in range(num_ptm)])
                    old_psp_ms_scores.append(old_psp_ms_score)
                    old_psp_lt_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][1] for j in range(num_ptm)])
                    old_psp_lt_scores.append(old_psp_lt_score)
                    old_psp_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W for j in range(num_ptm)])
                    old_psp_scores.append(old_psp_score)
                    old_exp_score = np.sum([possible_exp_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_exp_scores.append(old_exp_score)
                    old_dist_score = np.sum([ptm_dist_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_dist_scores.append(old_dist_score)
                    old_gs_num = np.sum([ptm_gs_nums[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_gs_nums.append(old_gs_num)
                    old_gs_score = np.sum([ptm_gs_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_gs_scores.append(old_gs_score)
                    old_fun_score = np.sum([ptm_fun_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)])
                    old_fun_scores.append(old_fun_score)
                    old_comb_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W + possible_exp_scores[possible_ptm_pos.index(res[j])] * MP_W + ptm_dist_scores[possible_ptm_pos.index(res[j])] * DT_W + ptm_gs_scores[possible_ptm_pos.index(res[j])] * GS_W + ptm_fun_scores[possible_ptm_pos.index(res[j])] * FL_W for j in range(num_ptm)])
                    old_comb_scores.append(old_comb_score)
                    
                    if old_exp_score < new_exp_score:
                        p2p.append(old_exp_score)
                        n2p.append(new_exp_score - old_exp_score)
                        p2n.append(0)
                        n2n.append(num_ptm - new_exp_score)
                    else:
                        p2p.append(new_exp_score)
                        n2p.append(0)
                        p2n.append(old_exp_score - new_exp_score)
                        n2n.append(num_ptm - old_exp_score)

                    if old_gs_num < new_gs_num:
                        gs_p2p.append(old_gs_num)
                        gs_n2p.append(new_gs_num - old_gs_num)
                        gs_p2n.append(0)
                        gs_n2n.append(num_ptm - new_gs_num)
                    else:
                        gs_p2p.append(new_gs_num)
                        gs_n2p.append(0)
                        gs_p2n.append(old_gs_num - new_gs_num)
                        gs_n2n.append(num_ptm - old_gs_num)

                    protein_names.append(prot_name)
                    uniprot_ids.append(idi)
                    pho_num.append(num_ptm)
                    old_pepts.append(bad_pepts[i])
                    # old_sites.append(old_site)
                    new_pepts.append(new_peptide)
                    # new_sites.append(new_site)
                    
    reassignment = pd.DataFrame({
        "Protein": protein_names,
        "UniProt": uniprot_ids,
        "PhoNum": pho_num,
        "OldPeptide": old_pepts,
        "NewPeptide": new_pepts,
        # "OldSite": old_sites,
        # "NewSite": new_sites,
        "AScore": ascores,
        "OldGsNum": old_gs_nums,
        "NewGsNum": new_gs_nums,
        "OldGsScore": old_gs_scores,
        "NewGsScore": new_gs_scores,
        "OldFunScore": old_fun_scores,
        "NewFunScore": new_fun_scores,
        "OldPspMsScore": old_psp_ms_scores,
        "NewPspMsScore": new_psp_ms_scores,
        "OldPspLtScore": old_psp_lt_scores,
        "NewPspLtScore": new_psp_lt_scores,
        "OldPspScore": old_psp_scores,
        "NewPspScore": new_psp_scores,
        "OldExpScore": old_exp_scores,
        "NewExpScore": new_exp_scores,
        "OldDistScore": old_dist_scores,
        "NewDistScore": new_dist_scores,
        "OldCombScore": old_comb_scores,
        "NewCombScore": new_comb_scores,
        "P2P": p2p,
        "N2P": n2p,
        "P2N": p2n,
        "N2N": n2n,
        "GS_P2P": gs_p2p,
        "GS_N2P": gs_n2p,
        "GS_P2N": gs_p2n,
        "GS_N2N": gs_n2n,
    })

    # Option: save the reassignment summary to a file
    # if save_file:
    #     # reassignment.to_csv(save_file + ".tsv", sep="\t", index=False)
    #     reassignment.to_pickle(save_file + ".pkl")

    # Now return a summarized list of sums of individual column
    # summary = [thr, MS_W, LT_W, DT_W, MP_W, GS_W, FL_W, np.sum(reassignment["AScore"]), bad_pept_num, bad_site_num, np.sum(reassignment["PhoNum"]), np.sum(reassignment["OldGsNum"]), np.sum(reassignment["NewGsNum"]), np.sum(reassignment["OldGsScore"]), np.sum(reassignment["NewGsScore"]), np.sum(reassignment["OldFunScore"]), np.sum(reassignment["NewFunScore"]), np.sum(reassignment["OldPspMsScore"]), np.sum(reassignment["NewPspMsScore"]), np.sum(reassignment["OldPspLtScore"]), np.sum(reassignment["NewPspLtScore"]), np.sum(reassignment["OldPspScore"]), np.sum(reassignment["NewPspScore"]), np.sum(reassignment["OldExpScore"]), np.sum(reassignment["NewExpScore"]), np.sum(reassignment["OldDistScore"]), np.sum(reassignment["NewDistScore"]), np.sum(reassignment["OldCombScore"]), np.sum(reassignment["NewCombScore"]), np.sum(reassignment["P2P"]), np.sum(reassignment["N2P"]), np.sum(reassignment["P2N"]), np.sum(reassignment["N2N"]), np.sum(reassignment["GS_P2P"]), np.sum(reassignment["GS_N2P"]), np.sum(reassignment["GS_P2N"]), np.sum(reassignment["GS_N2N"])]

    return(reassignment)
    # return(summary)


# %%
def reassignment_and_summary(params = None, df = df, ptm_label = '*', PSP_sites = PSP_sites, GS_sites = GS_sites, FUN_sites = FUN_sites, thr = None, exp_par = None):
    
    (prefix, instance, MS_W, LT_W, DT_W) = params
    np.random.seed(random.randint(instance * 10000, (instance + 1) * 10000))
    dfin = df.copy()
    ids = list(set(dfin["UniProt"]))
    new_dat = []

    for idi in ids:
        dfi = dfin[dfin["UniProt"] == idi].copy()
        pepts = dfi["Peptide"].to_list()
        reses = dfi["Residue"].to_list()
        dfi_ascore = dfi["AScore"].to_list()
        poses = get_res_pos(reses)

        bad_pepts = dfi["Peptide"].to_list()
        bad_reses = dfi["Residue"].to_list()
        bad_ascores = dfi["AScore"].to_list()
        bad_poses = get_res_pos(bad_reses)
        bad_names = get_res_names(bad_reses)
        pspi = PSP_sites[PSP_sites["UniProt"] == idi].copy()
        gsi = GS_sites[GS_sites["UniProt"] == idi].copy()
        funi = FUN_sites[FUN_sites["UniProt"] == idi].copy()
        pspi_ids = list(set(pspi["Site"]))
        gsi_ids = list(set(gsi["Site"]))
        funi_ids = list(set(funi["Site"]))
        gsi_poses = list(set(gsi["Position in target protein"]))
        funi_poses = list(set(funi["Position"]))
        if pspi.shape[0] > 0:
            # pspevi = np.array([list(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0)), list(pspi["LT_LIT"].fillna(0))]).transpose()
            pspevi = np.array([list(np.log2(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0) + 1)), list(np.log2(pspi["LT_LIT"].fillna(0) + 1))]).transpose()
            psppos = get_phosphositeplus_pos(pspi["MOD_RSD"])

            param_prob = []
            old_param_prob = []
            old_psp_ms_scores = []
            old_psp_lt_scores = []
            old_psp_scores = []
            old_dist_scores = []
            old_comb_scores = []
            old_gs_nums = []
            old_gs_scores = []
            old_fun_scores = []
            new_pepts = []
            new_reses = []
            new_psp_ms_scores = []
            new_psp_lt_scores = []
            new_psp_scores = []
            new_dist_scores = []
            new_comb_scores = []
            new_gs_nums = []
            # new_gs_scores = []
            new_fun_scores = []

            for i, pos in enumerate(bad_poses):
                # print(pos)
                ptm_info = get_ptm_info(bad_pepts[i], bad_reses[i])
                possible_ptm_pos = [ptm[0] for ptm in ptm_info]
                possible_ptm_scores = []
                possible_gs_nums = []
                possible_gs_scores = []
                possible_fun_scores = []
                for ptm_pos in possible_ptm_pos:
                    if ptm_pos in psppos:
                        possible_ptm_scores.append(np.log10(pspevi[psppos.index(ptm_pos)] + 1))
                    else:
                        possible_ptm_scores.append(np.zeros(2))
                    if ptm_pos in gsi_poses:
                        possible_gs_nums.append(1)
                    else:
                        possible_gs_nums.append(0)
                    if ptm_pos in funi_poses:
                        possible_fun_scores.append(funi[funi["Position"] == ptm_pos]["Functional_score"].to_list()[0])
                    else:
                        possible_fun_scores.append(0)
                possible_ptm_scores /= (np.array(possible_ptm_scores).sum(axis=0, keepdims=True) + 1e-12)
                true_res_pos = get_ptm_pos_in_pept(bad_pepts[i], ptm_label)
                ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
                ptm_dist_scores = (bad_ascores[i] + 1e-12) * (1 - (ptm_dists/ (ptm_dists.max() + 1)))
                ptm_dist_scores /= ptm_dist_scores.sum()
                ptm_gs_nums = np.array(possible_gs_nums)
                # ptm_gs_scores = np.array(possible_gs_nums) #/ (np.array(possible_gs_nums).sum() + 1e-12)
                ptm_fun_scores = np.array(possible_fun_scores)# / (np.array(possible_fun_scores).sum() + 1e-12)
                ptm_final_scores = possible_ptm_scores[:,0] * MS_W + possible_ptm_scores[:,1] * LT_W + ptm_dist_scores * DT_W 
                num_ptm = len(pos)
                reassign_prob = ptm_final_scores/np.sum(ptm_final_scores)
                if prefix == "random":
                    # Reassignment based on random selection
                    new_ptm_indices = np.random.choice(len(ptm_final_scores), num_ptm, replace=False)
                else:
                # elif prefix == "params":
                    # Reassignment based on the algorithm with weights
                    new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
                # else:
                #     raise NotImplementedError("Error: prefix must be either 'random' or 'params'!")
                #     # return(None)

                new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
                new_residue = []
                new_res_pos = []
                for new_ptm in new_ptm_info:
                    new_residue.append(new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower())
                    new_res_pos.append(new_ptm[2])
                new_peptide = relable_pept(bad_pepts[i], sorted(new_res_pos), ptm_label)
                new_pepts.append(new_peptide)
                new_reses.append(''.join(new_residue))
                if prefix == "random":
                    # Reassignment based on random selection
                    for j, k in enumerate(new_ptm_indices):
                        if j == 0:
                            p = reassign_prob[k]
                            q = p
                        else:
                            p *= reassign_prob[k]/(1 - q)
                            q += reassign_prob[k]
                    param_prob.append(p)
                else:
                # elif prefix == "params":
                    # Reassignment based on the algorithm with weights
                    param_prob.append(np.max(reassign_prob))
                # else:
                #     raise NotImplementedError("Error: prefix must be either 'random' or 'params'!")
                
                new_psp_ms_score = np.sum([possible_ptm_scores[k][0] for k in new_ptm_indices])
                new_psp_ms_scores.append(new_psp_ms_score) 
                new_psp_lt_score = np.sum([possible_ptm_scores[k][1] for k in new_ptm_indices])
                new_psp_lt_scores.append(new_psp_lt_score) 
                new_psp_score = np.sum([possible_ptm_scores[k][0] * MS_W + possible_ptm_scores[k][1] * LT_W for k in new_ptm_indices])
                new_psp_scores.append(new_psp_score)
                new_dist_score = np.sum([ptm_dist_scores[k] for k in new_ptm_indices])
                new_dist_scores.append(new_dist_score)
                new_gs_num = np.sum([ptm_gs_nums[k] for k in new_ptm_indices])
                new_gs_nums.append(new_gs_num)
                # new_gs_score = np.sum([ptm_gs_scores[k] for k in new_ptm_indices])
                # new_gs_scores.append(new_gs_score)
                new_fun_score = np.sum([ptm_fun_scores[k] for k in new_ptm_indices])
                new_fun_scores.append(new_fun_score)
                new_comb_score = np.sum([ptm_final_scores[k] for k in new_ptm_indices])
                new_comb_scores.append(new_comb_score)

                old_psp_ms_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(pos[j])][0] for j in range(num_ptm)])
                old_psp_ms_scores.append(old_psp_ms_score)
                old_psp_lt_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(pos[j])][1] for j in range(num_ptm)])
                old_psp_lt_scores.append(old_psp_lt_score)
                old_psp_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(pos[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(pos[j])][1] * LT_W for j in range(num_ptm)])
                old_psp_scores.append(old_psp_score)
                old_dist_score = np.sum([ptm_dist_scores[possible_ptm_pos.index(pos[j])] for j in range(num_ptm)])
                old_dist_scores.append(old_dist_score)
                old_gs_num = np.sum([ptm_gs_nums[possible_ptm_pos.index(pos[j])] for j in range(num_ptm)])
                old_gs_nums.append(old_gs_num)
                # old_gs_score = np.sum([ptm_gs_scores[possible_ptm_pos.index(pos[j])] for j in range(num_ptm)])
                # old_gs_scores.append(old_gs_score)
                old_fun_score = np.sum([ptm_fun_scores[possible_ptm_pos.index(pos[j])] for j in range(num_ptm)])
                old_fun_scores.append(old_fun_score)
                old_comb_score = np.sum([possible_ptm_scores[possible_ptm_pos.index(pos[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(pos[j])][1] * LT_W + ptm_dist_scores[possible_ptm_pos.index(pos[j])] * DT_W for j in range(num_ptm)])
                old_comb_scores.append(old_comb_score)
                old_param_prob.append(old_comb_score/np.sum(ptm_final_scores))
            maxAscore = dfi[["Residue", "AScore"]].groupby("Residue").max()
            dfi["ParamProb"] = param_prob
            dfi["OldParamProb"] = old_param_prob
            dfi["NewPeptide"] = new_pepts
            dfi["NewResidue"] = new_reses
            dfi["OldMaxAscore"] = maxAscore.loc[dfi["Residue"], "AScore"].to_list()
            dfi["NewMaxAscore"] = [maxAscore.loc[new_res, "AScore"] if new_res in maxAscore.index else np.nan for new_res in dfi["NewResidue"]]
            dfi["OldPspMsScore"] = old_psp_ms_scores
            dfi["NewPspMsScore"] = new_psp_ms_scores
            dfi["OldPspLtScore"] = old_psp_lt_scores
            dfi["NewPspLtScore"] = new_psp_lt_scores
            dfi["OldPspScore"] = old_psp_scores
            dfi["NewPspScore"] = new_psp_scores
            dfi["OldDistScore"] = old_dist_scores
            dfi["NewDistScore"] = new_dist_scores
            dfi["OldGsNum"] = old_gs_nums
            dfi["NewGsNum"] = new_gs_nums
            # dfi["OldGsScore"] = old_gs_scores
            # dfi["NewGsScore"] = new_gs_scores
            dfi["OldFunScore"] = old_fun_scores
            dfi["NewFunScore"] = new_fun_scores
            dfi["OldCombScore"] = old_comb_scores
            dfi["NewCombScore"] = new_comb_scores
            new_dat.append(dfi)
    new_dat = pd.concat(new_dat)
    new_dat = new_dat.reset_index(drop=True)
    # Now return a summarized list of sums of individual column
    new_dat["log2ProbFC"] = np.log2(new_dat["ParamProb"] / new_dat["RandProb"])
    new_dat["log2OldProbFC"] = np.log2(new_dat["OldParamProb"] / new_dat["RandProb"])
    new_dat["deltaMaxAscore"] = new_dat["NewMaxAscore"] - new_dat["OldMaxAscore"]
    new_dat["deltaPspMsScore"] = new_dat["NewPspMsScore"] - new_dat["OldPspMsScore"]
    new_dat["deltaPspLtScore"] = new_dat["NewPspLtScore"] - new_dat["OldPspLtScore"]
    new_dat["deltaPspScore"] = new_dat["NewPspScore"] - new_dat["OldPspScore"]
    new_dat["deltaDistScore"] = new_dat["NewDistScore"] - new_dat["OldDistScore"]
    new_dat["deltaGsNum"] = new_dat["NewGsNum"] - new_dat["OldGsNum"]
    # new_dat["deltaGsScore"] = new_dat["NewGsScore"] - new_dat["OldGsScore"]
    new_dat["deltaFunScore"] = new_dat["NewFunScore"] - new_dat["OldFunScore"]
    new_dat["deltaCombScore"] = new_dat["NewCombScore"] - new_dat["OldCombScore"]
    new_dat["deltaProb"] = np.log2(new_dat["ParamProb"] / new_dat["RandProb"])
    # Option: save the reassigned data to a file
    new_dat.to_pickle("{}/{}_reassignment_{}.pkl".format(results_dir, prefix, instance))

    MeanLog2ProbFC = np.mean(new_dat["log2ProbFC"])
    StdLog2ProbFC = np.std(new_dat["log2ProbFC"])
    MeanLog2OldProbFC = np.mean(new_dat["log2OldProbFC"])
    StdLog2OldProbFC = np.std(new_dat["log2OldProbFC"])
    PosDeltaMaxAscoreNum = np.sum(new_dat["deltaMaxAscore"] > 0)
    NegDeltaMaxAscoreNum = np.sum(new_dat["deltaMaxAscore"] < 0)
    PosDeltaPspMsScoreNum = np.sum(new_dat["deltaPspMsScore"] > 0)
    NegDeltaPspMsScoreNum = np.sum(new_dat["deltaPspMsScore"] < 0)
    PosDeltaPspLtScoreNum = np.sum(new_dat["deltaPspLtScore"] > 0)
    NegDeltaPspLtScoreNum = np.sum(new_dat["deltaPspLtScore"] < 0)
    PosDeltaPspScoreNum = np.sum(new_dat["deltaPspScore"] > 0)
    NegDeltaPspScoreNum = np.sum(new_dat["deltaPspScore"] < 0)
    PosDeltaDistScoreNum = np.sum(new_dat["deltaDistScore"] >= 0)
    NegDeltaDistScoreNum = np.sum(new_dat["deltaDistScore"] < 0)
    OldGsNumTot = np.sum(new_dat["OldGsNum"])
    NewGsNumTot = np.sum(new_dat["NewGsNum"])
    PosDeltaGsNum = np.sum(new_dat["deltaGsNum"] > 0)
    NegDeltaGsNum = np.sum(new_dat["deltaGsNum"] < 0)
    OldFunScoreSum = np.sum(new_dat["OldFunScore"])
    NewFunScoreSum = np.sum(new_dat["NewFunScore"])
    PosDeltaFunScoreNum = np.sum(new_dat["deltaFunScore"] > 0)
    NegDeltaFunScoreNum = np.sum(new_dat["deltaFunScore"] < 0)
    PosDeltaCombScoreNum = np.sum(new_dat["deltaCombScore"] > 0)
    NegDeltaCombScoreNum = np.sum(new_dat["deltaCombScore"] < 0)
    summary = [prefix, instance, MS_W, LT_W, DT_W, MeanLog2ProbFC, StdLog2ProbFC, MeanLog2OldProbFC, StdLog2OldProbFC, PosDeltaMaxAscoreNum, NegDeltaMaxAscoreNum, PosDeltaPspMsScoreNum, NegDeltaPspMsScoreNum, PosDeltaPspLtScoreNum, NegDeltaPspLtScoreNum, PosDeltaPspScoreNum, NegDeltaPspScoreNum, PosDeltaDistScoreNum, NegDeltaDistScoreNum, OldGsNumTot, NewGsNumTot, PosDeltaGsNum, NegDeltaGsNum, OldFunScoreSum, NewFunScoreSum, PosDeltaFunScoreNum, NegDeltaFunScoreNum, PosDeltaCombScoreNum, NegDeltaCombScoreNum]

    return(new_dat, summary)
    # return(summary)


# %% [markdown]
# ## The optimum parameters for reassignment

# %%
import time

# thr = 30
HTP_params = ("HTP_only", 0, 1, 0, 0)
Medians_params = ("Medians", 0, 0.40935512896496834, 0.0028834480249440245, 0.004439231598796614)
Fun_medians_params = ("FunMedians", 0, 0.3373114823599035, 0.004482122763962499, 0.003735925450381492)
Gs_medians_params = ("GsMedians", 0, 0.42056700185773666, 0.0031175186176344164, 0.004656634333672645)
# (prefix, instance, MS_W, LT_W, DT_W) = params

start_time = time.time()
HTP_reassigned, HTP_summary = reassignment_and_summary(HTP_params, df)
Medians_reassigned, Medians_summary = reassignment_and_summary(Medians_params, df)
Fun_medians_reassigned, Fun_medians_summary = reassignment_and_summary(Fun_medians_params, df)
Gs_medians_reassigned, Gs_medians_summary = reassignment_and_summary(Gs_medians_params, df)

end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# %%
new_reassignment = (HTP_reassigned, Medians_reassigned, Fun_medians_reassigned, Gs_medians_reassigned)
new_summary = (HTP_summary, Medians_summary, Fun_medians_summary, Gs_medians_summary)
with open("{}/new_reassignment.pkl".format(results_dir), "wb") as f:
    pickle.dump(new_reassignment, f)
with open("{}/new_summary.pkl".format(results_dir), "wb") as f:
    pickle.dump(new_summary, f)

# %%
with open("{}/new_reassignment.pkl".format(results_dir), "rb") as f:
    new_reassignment = pickle.load(f)
with open("{}/new_summary.pkl".format(results_dir), "rb") as f:
    new_summary = pickle.load(f)

HTP_reassigned, Medians_reassigned, Fun_medians_reassigned, Gs_medians_reassigned = new_reassignment
HTP_summary, Medians_summary, Fun_medians_summary, Gs_medians_summary = new_summary

# %% [markdown]
# ## Parallelize the function

# %%
# if __name__ == '__main__':
    
#     # Parallelize the function
#     from multiprocessing import Pool
#     import multiprocessing as mp

#     import sys
#     thread_num = int(sys.argv[1])
#     print("Number of threads: {}".format(thread_num))
#     prefix = str(sys.argv[2]).lower() # 'random' or 'params'
#     print("Using sampling method: {}".format(prefix))
#     # instance = int(sys.argv[3]) 
#     # print("Instance: {}".format(instance))
#     # thread_num = mp.cpu_count() // 2

#     if prefix == 'random':
#         parameters = [(prefix, i, 1, 1, 1) for i in range(10000)]
#     else:
#         parameters = [(prefix, i, 10 ** (-3 * random.uniform(0,1)), 10 ** (-3 * random.uniform(0,1)), 10 ** (-3 * random.uniform(0,1))) for i in range(10000)]
    
#     # import time
#     # start_time = time.time()
#     # print("Start time: ", start_time)

#     with Pool(thread_num) as p:
#         results = p.map(reassignment_and_summary, parameters)

#     # end_time = time.time()
#     # print("End time: ", end_time)
#     # print("Time elapsed: ", end_time - start_time)

#     # Save the results
#     # print(results)
#     summary = pd.DataFrame(results, columns=["prefix", "instance", "MS_W", "LT_W", "DT_W", "MeanLog2ProbFC", "StdLog2ProbFC", "MeanLog2OldProbFC", "StdLog2OldProbFC", "PosDeltaMaxAscoreNum", "NegDeltaMaxAscoreNum", "PosDeltaPspMsScoreNum", "NegDeltaPspMsScoreNum", "PosDeltaPspLtScoreNum", "NegDeltaPspLtScoreNum", "PosDeltaPspScoreNum", "NegDeltaPspScoreNum", "PosDeltaDistScoreNum", "NegDeltaDistScoreNum", "OldGsNumTot", "NewGsNumTot", "PosDeltaGsNum", "NegDeltaGsNum", "OldFunScoreSum", "NewFunScoreSum", "PosDeltaFunScoreNum", "NegDeltaFunScoreNum", "PosDeltaCombScoreNum", "NegDeltaCombScoreNum"])

#     summary.to_pickle("{}/sampling_summary_with_{}_method.pkl".format(results_dir, prefix))

# %% [markdown]
# ## Analyzing data from 20230822

# %%
local_results_dir = "./Optimizing_Reassignment/20230822_MCF10A/"
# local_results_dir = "./Optimizing_Reassignment/20230916_HS578T/"
if not os.path.exists(local_results_dir):
    os.makedirs(local_results_dir)

results_dir = "~/workspace/data/csbc/20230822_Optimizing_Reassignment/"
# results_dir = "~/workspace/data/csbc/20230916_HS578T/"

# %%
# Collect all the data

all_results = []
for i in range(61):
    file = results_dir + "summary_with_{}_AScoreThr_new.pkl".format(i)
    all_results.append(pd.read_pickle(file))

reassignment_summaries = pd.concat(all_results)

# reassignment_summaries.to_csv(local_results_dir + "summary_with_all_AScores.csv", index=False)
reassignment_summaries.to_pickle(local_results_dir + "summary_with_all_AScores_new.pkl")

# %% [markdown]
# ### Analyze the results
#
# First analyze the results with GS_W = 0 and FL_W = 0

# %%
# results_dir = "./Optimizing_Reassignment/20230712/"
reassignment_summaries = pd.read_pickle(local_results_dir + "summary_with_all_AScores_new.pkl")

df = reassignment_summaries[(reassignment_summaries["GS_W"] == 0) & (reassignment_summaries["FL_W"] == 0)].copy()
# df = reassignment_summaries.copy()

# df["DeltaCombScore"] = (df["NewCombScore"] - df["OldCombScore"]) / (df["OldCombScore"] + 1e-6)
# df["DeltaGsNum"] = (df["NewGsNum"] - df["OldGsNum"]) / df["OldGsNum"]
# df["DeltaRelGsNum"] = (df["NewGsNum"] - df["OldGsNum"]) / df["BadSiteNum"]
# df["DeltaGsScore"] = (df["NewGsScore"] - df["OldGsScore"]) / (df["OldGsScore"] + 1e-6)
# df["DeltaFunScore"] = (df["NewFunScore"] - df["OldFunScore"]) / (df["OldFunScore"] + 1e-6)
# df["DeltaPspScore"] = (df["NewPspScore"] - df["OldPspScore"]) / (df["OldPspScore"] + 1e-6)
# df["DeltaPspMsScore"] = (df["NewPspMsScore"] - df["OldPspMsScore"]) / (df["OldPspMsScore"] + 1e-6)
# df["DeltaPspLtScore"] = (df["NewPspLtScore"] - df["OldPspLtScore"]) / (df["OldPspLtScore"] + 1e-6)
# df["DeltaExpScore"] = (df["NewExpScore"] - df["OldExpScore"]) / (df["OldExpScore"] + 1e-6)
# df["DeltaDistScore"] = (df["NewDistScore"] - df["OldDistScore"]) / (df["OldDistScore"] + 1e-6)

df["DeltaCombScore"] = (df["NewCombScore"] - df["OldCombScore"])# / df["BadSiteNum"]
df["DeltaGsNum"] = (df["NewGsNum"] - df["OldGsNum"])
# df["DeltaRelGsNum"] = (df["NewGsNum"] - df["OldGsNum"]) / df["BadSiteNum"]
df["DeltaGsScore"] = (df["NewGsScore"] - df["OldGsScore"]) # / df["BadSiteNum"]
df["DeltaFunScore"] = (df["NewFunScore"] - df["OldFunScore"])
# df["DeltaFunScore"] = (df["NewFunScore"] - df["OldFunScore"]) / df["BadSiteNum"]
df["DeltaPspScore"] = (df["NewPspScore"] - df["OldPspScore"]) # / df["BadSiteNum"]
# df["DeltaPspMsScore"] = (df["NewPspMsScore"] - df["OldPspMsScore"]) # / df["BadSiteNum"]
# df["DeltaPspLtScore"] = (df["NewPspLtScore"] - df["OldPspLtScore"]) # / df["BadSiteNum"]
df["DeltaExpScore"] = (df["NewExpScore"] - df["OldExpScore"]) # / df["BadSiteNum"]
df["DeltaDistScore"] = (df["NewDistScore"] - df["OldDistScore"]) #/ df["BadSiteNum"]

df["DeltaFunScoreNorm"] = (df["DeltaFunScore"] - df["DeltaFunScore"].min()) / (df["DeltaFunScore"].max() - df["DeltaFunScore"].min())
df["DeltaGsNumNorm"] = (df["DeltaGsNum"] - df["DeltaGsNum"].min()) / (df["DeltaGsNum"].max() - df["DeltaGsNum"].min())

# df["MeanNewCombScore"] = df["NewCombScore"] / df["BadSiteNum"]
# df["NewGsNumRatio"] = df["NewGsNum"] / df["BadSiteNum"]
# df["MeanNewGsScore"] = df["NewGsScore"] / df["BadSiteNum"]
# df["MeanNewFunScore"] = df["NewFunScore"] / df["BadSiteNum"]
# df["MeanNewPspScore"] = df["NewPspScore"] / df["BadSiteNum"]
# df["MeanNewPspMsScore"] = df["NewPspMsScore"] / df["BadSiteNum"]
# df["MeanNewPspLtScore"] = df["NewPspLtScore"] / df["BadSiteNum"]
# df["MeanNewExpScore"] = df["NewExpScore"] / df["BadSiteNum"]
# df["MeanNewDistScore"] = df["NewDistScore"] / df["BadSiteNum"]

# df["MeanOldCombScore"] = df["OldCombScore"] / df["BadSiteNum"]
# df["OldGsNumRatio"] = df["OldGsNum"] / df["BadSiteNum"]
# df["MeanOldGsScore"] = df["OldGsScore"] / df["BadSiteNum"]
# df["MeanOldFunScore"] = df["OldFunScore"] / df["BadSiteNum"]
# df["MeanOldPspScore"] = df["OldPspScore"] / df["BadSiteNum"]
# df["MeanOldPspMsScore"] = df["OldPspMsScore"] / df["BadSiteNum"]
# df["MeanOldPspLtScore"] = df["OldPspLtScore"] / df["BadSiteNum"]
# df["MeanOldExpScore"] = df["OldExpScore"] / df["BadSiteNum"]
# df["MeanOldDistScore"] = df["OldDistScore"] / df["BadSiteNum"]

df["PositiveReassignRate"] = df["N2P"] / (df["N2N"] + df["N2P"])
df["NegativeReassignRate"] = df["P2N"] / (df["P2P"] + df["P2N"])
df["PositiveNegativeRatio"] = df["N2P"] / df["P2N"]
df["PositiveNegativeRateRatio"] = df["PositiveReassignRate"] / df["NegativeReassignRate"]

df["PositiveNegativeRatioNorm"] = (df["PositiveNegativeRatio"] - df["PositiveNegativeRatio"].min()) / (df["PositiveNegativeRatio"].max() - df["PositiveNegativeRatio"].min())
df["ReassignmentScore"] = df["DeltaFunScoreNorm"] * df["PositiveNegativeRatioNorm"]

df["GS_PositiveReassignRate"] = df["GS_N2P"] / (df["GS_N2N"] + df["GS_N2P"])
df["GS_NegativeReassignRate"] = df["GS_P2N"] / (df["GS_P2P"] + df["GS_P2N"])
df["GS_PositiveNegativeRatio"] = df["GS_N2P"] / df["GS_P2N"]
df["GS_PositiveNegativeRateRatio"] = df["GS_PositiveReassignRate"] / df["GS_NegativeReassignRate"]

df["GS_PositiveNegativeRatioNorm"] = (df["GS_PositiveNegativeRatio"] - df["GS_PositiveNegativeRatio"].min()) / (df["GS_PositiveNegativeRatio"].max() - df["GS_PositiveNegativeRatio"].min())
df["GS_ReassignmentScore"] = df["DeltaFunScoreNorm"] * df["GS_PositiveNegativeRatioNorm"]

df["Pars"] = df["MS_W"].astype(str) + "_" + df["LT_W"].astype(str) + "_" + df["DT_W"].astype(str) + "_" + df["MP_W"].astype(str)

df.to_pickle(local_results_dir + "summary_with_all_AScores_GS0_FL0.pkl")

# dfm = pd.pivot(df, index="Pars", columns="Thr", values="DeltaCombScore").iloc[1:,:].copy()

# %%
import seaborn as sns
# # %matplotlib inline
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['figure.figsize'] = [8, 6]

df = pd.read_pickle(local_results_dir + "summary_with_all_AScores_GS0_FL0.pkl")
# dfnz = df[~(df["Pars"].str.contains("0.0"))].copy()
dfnz = df.copy()

plt.figure()
sns.scatterplot(data=dfnz, x="DeltaFunScore", y="DeltaGsNum", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
# sns.scatterplot(data=dfnz, x="DeltaFunScore", y="DeltaGsNum", hue="Thr", alpha=0.2, s=1, linewidth=0)
plt.ylabel("Change of total gold standard site number")  # Set x and y axis labels
plt.xlabel("Change of total functional score")
plt.savefig(local_results_dir + "delta_fun_vs_delta_gs.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=dfnz, x="DeltaFunScore", y="PositiveNegativeRatio", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
# sns.scatterplot(data=dfnz, x="DeltaFunScore", y="PositiveNegativeRatio", hue="Thr", alpha=0.2, s=1, linewidth=0)
plt.ylabel("Ratio of postive and negative reassignments")  # Set x and y axis labels
plt.xlabel("Change of total functional score")
plt.savefig(local_results_dir + "delta_fun_vs_posneg_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=dfnz, x="DeltaFunScore", y="GS_PositiveNegativeRatio", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
# sns.scatterplot(data=dfnz, x="DeltaFunScore", y="PositiveNegativeRatio", hue="Thr", alpha=0.2, s=1, linewidth=0)
plt.ylabel("Ratio of postive and negative reassignments of GS")  # Set x and y axis labels
plt.xlabel("Change of total functional score")
plt.savefig(local_results_dir + "delta_fun_vs_GS_posneg_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=dfnz, x="PositiveNegativeRatio", y="GS_PositiveNegativeRatio", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
# sns.scatterplot(data=dfnz, x="PositiveNegativeRatio", y="PositiveNegativeRatio", hue="Thr", alpha=0.2, s=1, linewidth=0)
plt.ylabel("Ratio of postive and negative reassignments of GS")  # Set x and y axis labels
plt.xlabel("Ratio of postive and negative reassignments")  # Set x and y axis labels
plt.savefig(local_results_dir + "posneg_ratio_vs_GS_posneg_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaCombScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Relative changes of final score")
# plt.savefig("./Optimizing_Reassignment/relative_change_final_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="DeltaGsNum", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Changes of GS number")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "absolute_change_gold_std_site_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaRelGsNum", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Changes of relative gold_std_site number")
# plt.savefig("./Optimizing_Reassignment/relative_change_gold_std_site_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaGsScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Relative changes of gold_std_site score")
# plt.savefig("./Optimizing_Reassignment/relative_change_gold_std_site_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaFunScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Relative changes of functional score")
# plt.savefig("./Optimizing_Reassignment/relative_change_functional_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="DeltaFunScore", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Changes of total functional score")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "total_change_functional_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaPspScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Changes of total PSP score")
# plt.savefig(local_results_dir + "relative_change_PSP_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()


# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaExpScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Changes of total exp score")
# plt.savefig(local_results_dir + "relative_change_MP_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="DeltaDistScore", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Changes of total distance score")
# plt.savefig(local_results_dir + "relative_change_distance_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# plt.figure()
# sns.boxplot(data=dfnz, x="Thr", y="BadSiteNum", showfliers=True)
# plt.xlabel("AScore cutoff")  # Set x and y axis labels
# plt.ylabel("Number of potentially wrong sites")
# plt.savefig(local_results_dir + "bad_site_numbers.pdf", dpi = 300, bbox_inches='tight')
# plt.close()


plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="N2P", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Postitive reassign number")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "positive_reassign_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="P2N", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Negative reassign number")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "negative_reassign_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="PositiveReassignRate", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Postitive reassign rate")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "positive_reassign_rate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="NegativeReassignRate", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Negative reassign rate")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "negative_reassign_rate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="PositiveNegativeRatio", showfliers=True)
# plt.axhline(y=1, color='k', linestyle='--')
# plt.yscale('log')
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Positive-negative ratio")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "positive_negative_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="PositiveNegativeRateRatio", showfliers=True)
plt.axhline(y=1, color='k', linestyle='--')
# plt.yscale('log')
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Ratio of positive & negative reassignment rate")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "positive_negative_rate_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="ReassignmentScore", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Reassignment Score")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "reassginment_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=dfnz, x="PositiveReassignRate", y="NegativeReassignRate", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
plt.ylabel("Proportion of negatively reassgined sites")  # Set x and y axis labels
plt.xlabel("Proportion of positively reassgined sites")  # Set x and y axis labels
plt.savefig(local_results_dir + "neg_rate_vs_pos_rate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=dfnz, x="N2P", y="P2N", hue="Thr", alpha=0.3, s=1, linewidth=0, palette="cividis")
plt.ylabel("Number of negatively reassgined sites")  # Set x and y axis labels
plt.xlabel("Number of positively reassgined sites")  # Set x and y axis labels
plt.savefig(local_results_dir + "neg_number_vs_pos_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_N2P", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Postitive reassign number of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_positive_reassign_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_P2N", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Negative reassign number of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_negative_reassign_number.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_PositiveReassignRate", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Postitive reassign rate of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_positive_reassign_rate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_NegativeReassignRate", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Negative reassign rate of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_negative_reassign_rate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_PositiveNegativeRatio", showfliers=True)
# plt.axhline(y=1, color='', linestyle='--')
# plt.yscale('log')
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Positive-negative ratio of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_positive_negative_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_PositiveNegativeRateRatio", showfliers=True)
plt.axhline(y=1, color='k', linestyle='--')
# plt.yscale('log')
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Ratio of positive & negative reassignment rate of GS")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_positive_negative_rate_ratio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.boxplot(data=dfnz, x="Thr", y="GS_ReassignmentScore", showfliers=True)
plt.xlabel("AScore cutoff")  # Set x and y axis labels
plt.ylabel("Reassignment Score of GS sites")
plt.xticks(list(range(0,61,2)))
plt.savefig(local_results_dir + "GS_reassginment_score.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
# df35 = pd.read_pickle(local_results_dir + "summary_with_35_AScoreThr.pkl")

df_thr = pd.read_pickle(results_dir + "summary_with_39_AScoreThr_new.pkl")
# df_thr = pd.read_pickle(results_dir + "summary_with_45_AScoreThr_new.pkl")

df = df_thr[(df_thr["GS_W"] == 0) & (df_thr["FL_W"] == 0)].copy()

df["DeltaCombScore"] = (df["NewCombScore"] - df["OldCombScore"])# / df["BadSiteNum"]
df["DeltaGsNum"] = (df["NewGsNum"] - df["OldGsNum"])
# df["DeltaRelGsNum"] = (df["NewGsNum"] - df["OldGsNum"]) / df["BadSiteNum"]
df["DeltaGsScore"] = (df["NewGsScore"] - df["OldGsScore"]) # / df["BadSiteNum"]
df["DeltaFunScore"] = (df["NewFunScore"] - df["OldFunScore"])
# df["DeltaFunScore"] = (df["NewFunScore"] - df["OldFunScore"]) / df["BadSiteNum"]
df["DeltaPspScore"] = (df["NewPspScore"] - df["OldPspScore"]) # / df["BadSiteNum"]
# df["DeltaPspMsScore"] = (df["NewPspMsScore"] - df["OldPspMsScore"]) # / df["BadSiteNum"]
# df["DeltaPspLtScore"] = (df["NewPspLtScore"] - df["OldPspLtScore"]) # / df["BadSiteNum"]
df["DeltaExpScore"] = (df["NewExpScore"] - df["OldExpScore"]) # / df["BadSiteNum"]
df["DeltaDistScore"] = (df["NewDistScore"] - df["OldDistScore"]) #/ df["BadSiteNum"]

df["DeltaFunScoreNorm"] = (df["DeltaFunScore"] - df["DeltaFunScore"].min()) / (df["DeltaFunScore"].max() - df["DeltaFunScore"].min())
df["DeltaGsNumNorm"] = (df["DeltaGsNum"] - df["DeltaGsNum"].min()) / (df["DeltaGsNum"].max() - df["DeltaGsNum"].min())

df["PositiveReassignRate"] = df["N2P"] / (df["N2N"] + df["N2P"])
df["NegativeReassignRate"] = df["P2N"] / (df["P2P"] + df["P2N"])
df["PositiveNegativeRatio"] = df["N2P"] / df["P2N"]
df["PositiveNegativeRateRatio"] = df["PositiveReassignRate"] / df["NegativeReassignRate"]

df["PositiveNegativeRatioNorm"] = (df["PositiveNegativeRatio"] - df["PositiveNegativeRatio"].min()) / (df["PositiveNegativeRatio"].max() - df["PositiveNegativeRatio"].min())
df["ReassignmentScore"] = df["DeltaFunScoreNorm"] * df["PositiveNegativeRatioNorm"]

df["GS_PositiveReassignRate"] = df["GS_N2P"] / (df["GS_N2N"] + df["GS_N2P"])
df["GS_NegativeReassignRate"] = df["GS_P2N"] / (df["GS_P2P"] + df["GS_P2N"])
df["GS_PositiveNegativeRatio"] = df["GS_N2P"] / df["GS_P2N"]
df["GS_PositiveNegativeRateRatio"] = df["GS_PositiveReassignRate"] / df["GS_NegativeReassignRate"]

df["GS_PositiveNegativeRatioNorm"] = (df["GS_PositiveNegativeRatio"] - df["GS_PositiveNegativeRatio"].min()) / (df["GS_PositiveNegativeRatio"].max() - df["GS_PositiveNegativeRatio"].min())
df["GS_ReassignmentScore"] = df["DeltaFunScoreNorm"] * df["GS_PositiveNegativeRatioNorm"]

df["Pars"] = df["MS_W"].astype(str) + "_" + df["LT_W"].astype(str) + "_" + df["DT_W"].astype(str) + "_" + df["MP_W"].astype(str)

plt.figure()
sns.scatterplot(data=df, x="DeltaFunScore", y="DeltaGsNum", alpha=0.5, s=5, linewidth=0, palette="cividis")
plt.savefig(local_results_dir + "thr_scatterplot_DeltaFunScore_DeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DeltaFunScore", y="PositiveNegativeRatio", alpha=0.5, s=5, linewidth=0, palette="cividis")
plt.savefig(local_results_dir + "thr_scatterplot_DeltaFunScore_PositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DeltaFunScore", y="GS_PositiveNegativeRatio", alpha=0.5, s=5, linewidth=0, palette="cividis")
plt.savefig(local_results_dir + "thr_scatterplot_DeltaFunScore_GS_PositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="DT_W", hue="DeltaFunScore", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DeltaFunScore_DTvsLT.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="DT_W", hue="DeltaFunScore", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DeltaFunScore_DTvsMS.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="DT_W", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DeltaGsNum_DTvsLT.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="DT_W", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DeltaGsNum_MSvsLT.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DT_W", y="DeltaGsNum", hue="DeltaFunScore", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DTvsDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="DeltaGsNum", hue="DeltaFunScore", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_MSvsDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="DeltaGsNum", hue="DeltaFunScore", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_LTvsDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DT_W", y="DeltaFunScore", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DTvsDeltaFunScore.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="DeltaFunScore", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_MSvsDeltaFunScore.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="DeltaFunScore", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_LTvsDeltaFunScore.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="N2P", y="P2N", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_N2PvsP2N.pdf", dpi = 300, bbox_inches='tight')
# plt.close()


plt.figure()
sns.scatterplot(data=df, x="GS_N2P", y="GS_P2N", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_GS_N2PvsP2N.pdf", dpi = 300, bbox_inches='tight')
# plt.close()


plt.figure()
sns.scatterplot(data=df, x="PositiveReassignRate", y="NegativeReassignRate", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_PosRatevsNegRate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DT_W", y="PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DTvsPositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_MSvsPositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_LTvsPositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="GS_PositiveReassignRate", y="GS_NegativeReassignRate", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_GS_PosRate_vs_GS_NegRate.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="DT_W", y="GS_PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_DTvsGS_PositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="MS_W", y="GS_PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_MSvsGS_PositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

plt.figure()
sns.scatterplot(data=df, x="LT_W", y="GS_PositiveNegativeRatio", hue="DeltaGsNum", alpha=0.5, s=5, linewidth=0)
plt.xscale('log')
# plt.yscale('log')
plt.savefig(local_results_dir + "thr_scatterplot_LTvsGS_PositiveNegativeRatio.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# df_top = df[(df["DeltaFunScore"] > df["DeltaFunScore"].quantile(0.67)) & (df["DeltaGsNum"] > df["DeltaGsNum"].quantile(0.67)) & (df["PositiveNegativeRatio"] > df["PositiveNegativeRatio"].quantile(0.67))].copy()
# df_top = df[(df["DeltaGsNum"] > df["DeltaGsNum"].quantile(0.67)) & (df["PositiveNegativeRatio"] > df["PositiveNegativeRatio"].quantile(0.67))].copy()
df_top = df[(df["DeltaFunScore"] > df["DeltaFunScore"].quantile(2/3)) & (df["PositiveNegativeRatio"] > df["PositiveNegativeRatio"].quantile(2/3))].copy()
df_top.to_csv(local_results_dir + "quantile067_reassignment_weights.csv", index=False)

df_top100 = df.nlargest(100, "ReassignmentScore").sort_values(by="ReassignmentScore", ascending=False).copy()
# df_top100 = df.nlargest(100, "PositiveNegativeRatio").sort_values(by="PositiveNegativeRatio", ascending=False).copy()
df_top100.to_csv(local_results_dir + "top100_reassignment_weights.csv", index=False)

df_top100_GS = df.nlargest(100, "GS_ReassignmentScore").sort_values(by="GS_ReassignmentScore", ascending=False).copy()

# %%
df_top100_melt = df_top100.melt(id_vars=["ReassignmentScore"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
# fig = px.box(df_top100_melt, x="Parameter", y="ParameterValues", log_y=True, points="all")

fig = px.strip(df_top100_melt,
         x='Parameter',
         y='ParameterValues',
        #  color='Parameter',
         stripmode='overlay', log_y=True)

fig.add_trace(go.Box(y=df_top100_melt.query('Parameter == "MS_W"')['ParameterValues'], name='MS_W'))
fig.add_trace(go.Box(y=df_top100_melt.query('Parameter == "LT_W"')['ParameterValues'], name='LT_W'))
fig.add_trace(go.Box(y=df_top100_melt.query('Parameter == "DT_W"')['ParameterValues'], name='DT_W'))

fig.update_layout(autosize=False,
                  width=600,
                  height=600,
                  legend={'traceorder':'normal'})

fig.show()
fig.write_image(local_results_dir + "top100_reassignment_weights_boxplot.pdf")


df_top_melt = df_top.melt(id_vars=["ReassignmentScore"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
# fig = px.box(df_top_melt, x="Parameter", y="ParameterValues", log_y=True, points="all")

fig = px.strip(df_top_melt,
         x='Parameter',
         y='ParameterValues',
        #  color='Parameter',
         stripmode='overlay', log_y=True)

fig.add_trace(go.Box(y=df_top_melt.query('Parameter == "MS_W"')['ParameterValues'], name='MS_W'))
fig.add_trace(go.Box(y=df_top_melt.query('Parameter == "LT_W"')['ParameterValues'], name='LT_W'))
fig.add_trace(go.Box(y=df_top_melt.query('Parameter == "DT_W"')['ParameterValues'], name='DT_W'))

fig.update_layout(autosize=False,
                  width=600,
                  height=600,
                  legend={'traceorder':'normal'})

fig.show()
fig.write_image(local_results_dir + "top067quantile_reassignment_weights_boxplot.pdf")


df_top100_GS_melt = df_top100_GS.melt(id_vars=["ReassignmentScore"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
# fig = px.box(df_top100_GS_melt, x="Parameter", y="ParameterValues", log_y=True, points="all")

fig = px.strip(df_top100_GS_melt,
         x='Parameter',
         y='ParameterValues',
        #  color='Parameter',
         stripmode='overlay', log_y=True)

fig.add_trace(go.Box(y=df_top100_GS_melt.query('Parameter == "MS_W"')['ParameterValues'], name='MS_W'))
fig.add_trace(go.Box(y=df_top100_GS_melt.query('Parameter == "LT_W"')['ParameterValues'], name='LT_W'))
fig.add_trace(go.Box(y=df_top100_GS_melt.query('Parameter == "DT_W"')['ParameterValues'], name='DT_W'))

fig.update_layout(autosize=False,
                  width=600,
                  height=600,
                  legend={'traceorder':'normal'})

fig.show()
fig.write_image(local_results_dir + "top100_GS_reassignment_weights_boxplot.pdf")

# %%
fig = px.scatter_3d(df_top, x='MS_W', y='LT_W', z='DT_W', color='PositiveNegativeRatio', log_x=True, log_y=True, log_z=True)
fig.show()

fig = px.scatter_3d(df_top, x='MS_W', y='LT_W', z='DT_W', color='DeltaGsNum', log_x=True, log_y=True, log_z=True)
fig.show()

# fig = px.scatter(df_top, x='MS_W', y='LT_W', color='PositiveNegativeRatio', log_x=True, log_y=True, size='DeltaFunScore')
fig = px.scatter(df_top, x='MS_W', y='LT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_MSvsLTvsPositiveNegativeRatio.pdf")

fig = px.scatter(df_top, x='MS_W', y='DT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_MSvsDTvsPositiveNegativeRatio.pdf")

fig = px.scatter(df_top, x='LT_W', y='DT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_LTvsDTvsPositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='MS_W', y='LT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_MSvsLTvsPositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='MS_W', y='DT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_MSvsDTvsPositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='LT_W', y='DT_W', color='PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_LTvsDTvsPositiveNegativeRatio.pdf")



fig = px.scatter_3d(df_top, x='MS_W', y='LT_W', z='DT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True, log_z=True)
fig.show()

fig = px.scatter_3d(df_top, x='MS_W', y='LT_W', z='DT_W', color='DeltaGsNum', log_x=True, log_y=True, log_z=True)
fig.show()

# fig = px.scatter(df_top, x='MS_W', y='LT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True, size='DeltaFunScore')
fig = px.scatter(df_top, x='MS_W', y='LT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_MSvsLTvsGS_PositiveNegativeRatio.pdf")

fig = px.scatter(df_top, x='MS_W', y='DT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_MSvsDTvsGS_PositiveNegativeRatio.pdf")

fig = px.scatter(df_top, x='LT_W', y='DT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_quantile067_LTvsDTvsGS_PositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='MS_W', y='LT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_MSvsLTvsGS_PositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='MS_W', y='DT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_MSvsDTvsGS_PositiveNegativeRatio.pdf")

fig = px.scatter(df_top100, x='LT_W', y='DT_W', color='GS_PositiveNegativeRatio', log_x=True, log_y=True)
fig.show()
fig.write_image(local_results_dir + "thr_top100_LTvsDTvsGS_PositiveNegativeRatio.pdf")

# %% [markdown]
# ### Now let's get some other plots to filling the story and make it more complete

# %%
# the distribution of the AScores
# the ???



# the distribution of the reassignment score
# the distribution of the reassignment score for each AScore cutoff

reassignment_thr = reassignment([39] + df_top100.iloc[0][["MS_W", "LT_W", "DT_W", "MP_W", "GS_W", "FL_W"]].tolist(), mcf10a_ts_pept)

reassignment_thr.to_csv(local_results_dir + "reassignment_thr.csv", index=False)
reassignment_thr.to_pickle(local_results_dir + "reassignment_thr.pkl")


import plotly.figure_factory as ff

fig = ff.create_distplot([
    reassignment_thr["OldCombScore"].to_numpy(),
    reassignment_thr["NewCombScore"].to_numpy(),
                          ], [
    "Site score, old",
    "Site score, new",
                              ], 
                          bin_size=.04, 
                          histnorm= 'probability', 
                          curve_type= "kde",
                          show_rug=False)
fig.update_yaxes(type="log")
fig.update_traces(opacity=0.4)
fig.update_layout(barmode='overlay')
fig.show()
fig.write_image(local_results_dir + "mcf10a_comb_dist_scores_comparison.pdf")

fig = ff.create_distplot([
    reassignment_thr["OldPspScore"].to_numpy(),
    reassignment_thr["NewPspScore"].to_numpy(),
                          ], [
    "PSP score, old",
    "PSP score, new",
                              ], 
                          bin_size=.04, 
                          histnorm= 'probability', 
                          curve_type= "kde",
                          show_rug=False)
fig.update_yaxes(type="log")
fig.update_traces(opacity=0.4)
fig.update_layout(barmode='overlay')
fig.show()
fig.write_image(local_results_dir + "mcf10a_psp_dist_scores_comparison.pdf")

fig = ff.create_distplot([
    reassignment_thr["OldGsScore"].to_numpy(),
    reassignment_thr["NewGsScore"].to_numpy(),
                          ], [
    "GS score, old",
    "GS score, new",
                              ], 
                          bin_size=.04, 
                          histnorm= 'probability', 
                          curve_type= "kde",
                          show_rug=False)
fig.update_yaxes(type="log")
fig.update_traces(opacity=0.4)
fig.update_layout(barmode='overlay')
fig.show()
fig.write_image(local_results_dir + "mcf10a_gs_dist_scores_comparison.pdf")

fig = ff.create_distplot([
    reassignment_thr["OldFunScore"].to_numpy(),
    reassignment_thr["NewFunScore"].to_numpy(),
                          ], [
    "FUN score, old",
    "FUN score, new",
                              ], 
                          bin_size=.04, 
                          histnorm= 'probability', 
                          curve_type= "kde",
                          show_rug=False)
fig.update_yaxes(type="log")
fig.update_traces(opacity=0.4)
fig.update_layout(barmode='overlay')
fig.show()
fig.write_image(local_results_dir + "mcf10a_fun_dist_scores_comparison.pdf")



fig = go.Figure(
    data=[
        go.Bar(name="Positively reassigned", x=["Pos"], y=[reassignment_thr["N2P"].sum()]),
        go.Bar(name="Negatively reassigned", x=["Neg"], y=[reassignment_thr["P2N"].sum()]),

        # go.Bar(name="Negatively reassigned", x=["MCF10A, old", "MCF10A, new", "HS578T, old", "HS578T, new"], y=[mcf10a_ts_reassign_summary[mcf10a_ts_reassign_summary["MultiPhospho"]&mcf10a_ts_reassign_summary["OldGoldstd"]].shape[0], mcf10a_ts_reassign_summary[mcf10a_ts_reassign_summary["MultiPhospho"]&mcf10a_ts_reassign_summary["NewGoldstd"]].shape[0], hs578t_ts_reassign_summary[hs578t_ts_reassign_summary["MultiPhospho"]&hs578t_ts_reassign_summary["OldGoldstd"]].shape[0], hs578t_ts_reassign_summary[hs578t_ts_reassign_summary["MultiPhospho"]&hs578t_ts_reassign_summary["NewGoldstd"]].shape[0]]),
    ]
)
fig.show()
fig.write_image(local_results_dir + "mcf10a_reassignment_comparison_pos_vs_neg.pdf")

# %% [markdown]
# ## Analyzing data from 20240104
#
# ### Here we use an alternative approach, where we generate a background reassignment dataset, then compare the parameteric reassignment with the background reassignment
#
# Here are some good analysis to do:
# 0. Compare the scores of the random reassignment with the original scores
# 0. Compare the scores of the parametric reassignment with the original scores
# 1. Compare the scores of the parametric reassignment with the background reassignment
# 2. Compare the scores only for the sites that are reassigned in the parametric reassignment

# %%
# Now load both the summary data
results_dir = "./20240104_Optimizing_Reassignment"
params_file = f"{results_dir}/sampling_summary_with_params_method.pkl"
random_file = f"{results_dir}/sampling_summary_with_random_method.pkl"
params_df = pd.read_pickle(params_file)
# params_df.to_csv(f"{results_dir}/sampling_summary_with_params_method.csv", index=False)
random_df = pd.read_pickle(random_file)
# random_df.to_csv(f"{results_dir}/sampling_summary_with_random_method.csv", index=False)

# %%
print(params_df.columns)

# %%
params_df

# %%
random_df

# %% [markdown]
# #### Test plots of random_df and parametric_df

# %%
print(random_df["PosDeltaMaxAscoreNum"].unique().size)
print(random_df["NegDeltaMaxAscoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaMaxAscoreNum", y="NegDeltaMaxAscoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaMaxAscoreNum_vs_NegDeltaMaxAscoreNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
print(random_df["PosDeltaFunScoreNum"].unique().size)
print(random_df["NegDeltaFunScoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaFunScoreNum", y="NegDeltaFunScoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaFunScoreNum_vs_NegDeltaFunScoreNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
print(random_df["PosDeltaGsNum"].unique().size)
print(random_df["NegDeltaGsNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaGsNum", y="NegDeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaGsNum_vs_NegDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
print(params_df["PosDeltaMaxAscoreNum"].unique().size)
print(params_df["NegDeltaMaxAscoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaMaxAscoreNum", y="NegDeltaMaxAscoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaMaxAscoreNum_vs_NegDeltaMaxAscoreNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
print(params_df["PosDeltaFunScoreNum"].unique().size)
print(params_df["NegDeltaFunScoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaFunScoreNum", y="NegDeltaFunScoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaFunScoreNum_vs_NegDeltaFunScoreNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %%
print(params_df["PosDeltaGsNum"].unique().size)
print(params_df["NegDeltaGsNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaGsNum", y="NegDeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaGsNum_vs_NegDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
# plt.close()

# %% [markdown]
# ## Analyzing data from 20240110
#
# ### Steve's comments
# Start with monophosphorylated peptides, how much higher it is than the random reassignment.
#
# 1. Compare the distribution of the GS numbers and functional scores. Statistics for significance. This is to prove parametric reassignment is better than random reassignment.
# 2. Funcitonal score is better, it includes several different observations, and is a machine learning approach, covers more sites than gold standard site list.
#
# #### TODO
#
# 1. Send the an example reassginment table to Steve,
#
# 2. Ask statisticians about how to incoporate the probablity of reassignments in peptides with differnt number of candidate sites.
#
# 3. Statistics first then summary? Or summary first then statistics?
#
#
# 4. For each peptide, we can calculate the probability of parameteric probability p' = p(MS_W, LT_W, DT_W) and random probability p = 1/n:
#     4.1 For one peptide, we calculate the p' and p. 
#     4.2 Then get the p' and p for all the peptides.
#         this is a table
#         peptide, number of candidate site,  p', p 
#         pep1,                         5,   [0.1, 0.0, 0.5, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]
#         pep2,                         3,   [0.1, 0.4, 0.5], [0.33, 0.33, 0.33]
#         pep3,                         2,   [0.1, 0.9], [0.5, 0.5]

# %%
def get_quant(x):
    " for each value in x, return which quantile it corresponds to "
    return np.array([np.sum(x<x_i)/(len(x)-1) for x_i in x])

def scale2unit(x):
    " scale x to unit "
    # if np.sum(np.isnan(x)) > 0:
    #     print(np.sum(np.isnan(x))) 
    # x[np.isnan(x)] = 0
    return (x - x.min() + 1e-4) / (x.max() - x.min() + 1e-4)


# %%
results_dir = "./20240110_Optimizing_Reassignment"
params_file = f"{results_dir}/sampling_summary_with_params_method.pkl"
random_file = f"{results_dir}/sampling_summary_with_random_method.pkl"
params_df = pd.read_pickle(params_file)
# params_df.to_csv(f"{results_dir}/sampling_summary_with_params_method.csv", index=False)
random_df = pd.read_pickle(random_file)
# random_df.to_csv(f"{results_dir}/sampling_summary_with_random_method.csv", index=False)

# %%
reassigned_both = pd.concat([params_df, random_df])
print(reassigned_both.columns, reassigned_both.shape)

# %% [markdown]
# #### Load the input data for reassignment

# %%
df4 = pd.read_pickle("Optimizing_Reassignment/df4reassignment.pkl")
total_site_num = df4.shape[0]

# %%
dfa = pd.read_csv("./Optimizing_Reassignment/crosstab_median_normalized_phosphopeptide_with_ascores.tsv", sep="\t")
dfa["SiteNum"] = dfa["Residue"].apply(get_num_from_residue)
dfa

# %% [markdown]
# #### Plot the Ascore distribution

# %%
plt.figure(figsize=(10, 4))
sns.histplot(x= "AScore",
             data=dfa[(dfa["AScore"] != 0) & (dfa["AScore"] != 1000)] ,
            #   bins=25,
              hue="SiteNum",
              hue_order=[3, 2, 1],
              palette="Dark2",
              # palette="colorblind",
              # palette="tab10",
              kde=True,
              # log_scale=(True, True),
              # log_scale=(True, False),
              )
plt.xlim(0, 200)
plt.savefig("./Optimizing_Reassignment/all_ts_AScore_dist.pdf")

# %%
print(dfa.shape)

# %%
ascores_mono = dfa[dfa["SiteNum"] == 1]["AScore"].to_numpy()
print(f"Total mono-phosphorylated peptide: {len(ascores_mono)}")
print(f"Mono-phosphorylated peptide (AScore = 1000): {len(ascores_mono[ascores_mono == 1000])}")
print(f"Mono-phosphorylated peptide (AScore = 0): {len(ascores_mono[ascores_mono == 0])}")
print(f"Mono-phosphorylated peptide (AScore < 20): {len(ascores_mono[ascores_mono < 20])}")
print(f"Low AScore ratio (Mono-pho peptides): {len(ascores_mono[ascores_mono < 20]) / len(ascores_mono)}")

# %%
ascores_double = dfa[dfa["SiteNum"] == 2]["AScore"].to_numpy()
print(f"Total double-phosphorylated peptide: {len(ascores_double)}")
print(f"Double phosphorylated peptide (AScore = 1000): {len(ascores_double[ascores_double == 1000])}")
print(f"Double phosphorylated peptide (AScore = 0): {len(ascores_double[ascores_double == 0])}")
print(f"Double phosphorylated peptide (AScore < 20): {len(ascores_double[ascores_double < 20])}")
print(f"Low AScore ratio (Double phosphorylated peptides): {len(ascores_double[ascores_double < 20]) / len(ascores_double)}")

# %%
ascores_triple = dfa[dfa["SiteNum"] == 3]["AScore"].to_numpy()
print(f"Total triple-phosphorylated peptide: {len(ascores_triple)}")
print(f"Triple phosphorylated peptide (AScore = 1000): {len(ascores_triple[ascores_triple == 1000])}")
print(f"Triple phosphorylated peptide (AScore = 0): {len(ascores_triple[ascores_triple == 0])}")
print(f"Triple phosphorylated peptide (AScore < 20): {len(ascores_triple[ascores_triple < 20])}")
print(f"Low AScore ratio (Triple phosphorylated peptides): {len(ascores_triple[ascores_triple < 20]) / len(ascores_triple)}")

# %% [markdown]
# #### Adding new information in the reassignment table

# %%
reassigned_both.loc[len(reassigned_both.index)] = HTP_summary
reassigned_both.loc[len(reassigned_both.index)] = Medians_summary
reassigned_both.loc[len(reassigned_both.index)] = Fun_medians_summary
reassigned_both.loc[len(reassigned_both.index)] = Gs_medians_summary

# %%
reassigned_both

# %%
df = reassigned_both
df["Increase/Decrease"] = df["PosDeltaMaxAscoreNum"] / df["NegDeltaMaxAscoreNum"]
df["FPR"] = df["NegDeltaMaxAscoreNum"] / (df["PosDeltaMaxAscoreNum"] + df["NegDeltaMaxAscoreNum"])
df["TPR"] = df["PosDeltaMaxAscoreNum"] / (df["PosDeltaMaxAscoreNum"] + df["NegDeltaMaxAscoreNum"])

df["DeltaGsNum"] = df["NewGsNumTot"] - df["OldGsNumTot"]
df["DeltaFunScore"] = df["NewFunScoreSum"] - df["OldFunScoreSum"]

df["MeanDeltaGsNum"] = df["DeltaGsNum"] / total_site_num
df["MeanDeltaFunScore"] = df["DeltaFunScore"] / total_site_num

# gs_scaled = preproc.MinMaxScaler().fit_transform(df["MeanDeltaGsNum"].to_numpy().reshape(-1, 1)).flatten()
# fun_scaled = preproc.MinMaxScaler().fit_transform(df["MeanDeltaFunScore"].to_numpy().reshape(-1, 1)).flatten()
# tpr_scaled = preproc.MinMaxScaler().fit_transform(df["TPR"].to_numpy().reshape(-1, 1)).flatten()
gs_scaled = scale2unit(df["MeanDeltaGsNum"].to_numpy())
fun_scaled = scale2unit(df["MeanDeltaFunScore"].to_numpy())
tpr_scaled = scale2unit(df["TPR"].to_numpy())

df["TPR*GS"] = np.sqrt(tpr_scaled * gs_scaled)
df["TPR*FUN"] = np.sqrt(tpr_scaled * fun_scaled)
df["GS*FUN"] = np.sqrt(gs_scaled * fun_scaled)
df["TPR*GS*FUN"] = np.sqrt(tpr_scaled * gs_scaled * fun_scaled)

# %%
df

# %%
both_df_all = df.copy()

# %%
# reassign_results_dir = "/Users/feng626/workspace/data/csbc/20240110_Optimizing_Reassignment" 
# for file in os.listdir(reassign_results_dir):
#     if file.endswith(".pkl"):
#         print(os.path.join(reassign_results_dir, file))
#         reassign_df = pd.read_pickle(os.path.join(reassign_results_dir, file))
#         reassign_df.to_csv(os.path.join(reassign_results_dir, file.replace(".pkl", ".csv")), index=False)

# %%
print(params_df.columns)

# %%
params_df

# %%
random_df

# %% [markdown]
# #### Exploratory analysis on distribution of the reassignment scores

# %%
print(random_df["PosDeltaMaxAscoreNum"].unique().size)
print(random_df["NegDeltaMaxAscoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaMaxAscoreNum", y="NegDeltaMaxAscoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaMaxAscoreNum_vs_NegDeltaMaxAscoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(random_df["PosDeltaFunScoreNum"].unique().size)
print(random_df["NegDeltaFunScoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaFunScoreNum", y="NegDeltaFunScoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaFunScoreNum_vs_NegDeltaFunScoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(random_df["PosDeltaGsNum"].unique().size)
print(random_df["NegDeltaGsNum"].unique().size)
plt.figure()
sns.scatterplot(data=random_df, x="PosDeltaGsNum", y="NegDeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/random_PosDeltaGsNum_vs_NegDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(params_df["PosDeltaMaxAscoreNum"].unique().size)
print(params_df["NegDeltaMaxAscoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaMaxAscoreNum", y="NegDeltaMaxAscoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaMaxAscoreNum_vs_NegDeltaMaxAscoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(params_df["PosDeltaFunScoreNum"].unique().size)
print(params_df["NegDeltaFunScoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaFunScoreNum", y="NegDeltaFunScoreNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaFunScoreNum_vs_NegDeltaFunScoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(params_df["PosDeltaGsNum"].unique().size)
print(params_df["NegDeltaGsNum"].unique().size)
plt.figure()
sns.scatterplot(data=params_df, x="PosDeltaGsNum", y="NegDeltaGsNum", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/params_PosDeltaGsNum_vs_NegDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(df["PosDeltaMaxAscoreNum"].unique().size)
print(df["NegDeltaMaxAscoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=df, x="PosDeltaMaxAscoreNum", y="NegDeltaMaxAscoreNum", hue="prefix", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/PosDeltaMaxAscoreNum_vs_NegDeltaMaxAscoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(df["PosDeltaFunScoreNum"].unique().size)
print(df["NegDeltaFunScoreNum"].unique().size)
plt.figure()
sns.scatterplot(data=df, x="PosDeltaFunScoreNum", y="NegDeltaFunScoreNum", hue="prefix", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/PosDeltaFunScoreNum_vs_NegDeltaFunScoreNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(df["PosDeltaGsNum"].unique().size)
print(df["NegDeltaGsNum"].unique().size)
plt.figure()
sns.scatterplot(data=df, x="PosDeltaGsNum", y="NegDeltaGsNum", hue="prefix", alpha=0.5, s=5, linewidth=0)
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/PosDeltaGsNum_vs_NegDeltaGsNum.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df.columns

# %%
print(df["NewGsNumTot"].unique().size)
print(df["NewFunScoreSum"].unique().size)
plt.figure()
# sns.scatterplot(data=df, x="NewGsNumTot", y="NewFunScoreSum", hue="prefix", linewidth=0)
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="prefix", alpha=0.5, s=5, linewidth=0)
plt.axhline(0, 0, 1, color="grey", linestyle='--')
plt.axvline(0, 0, 1, color="grey", linestyle='--')
# plt.xscale('log')
# plt.yscale('log')
plt.legend([],[], frameon=False)
plt.savefig(f"{results_dir}/MeanDeltaGsNum_vs_MeanDeltaFunScore_for_all.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df["OldGsNumTot"].unique()/total_site_num

# %%
df["OldFunScoreSum"].unique()/total_site_num

# %%
plt.figure()
sns.histplot(
# sns.kdeplot(
    df, x="NewGsNumTot", y="NewFunScoreSum", hue="prefix",
    bins=200, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
)
plt.axhline(df["OldFunScoreSum"].tolist()[0], 0, 6000, color="grey")
plt.axvline(df["OldGsNumTot"].tolist()[0], 0, 6000, color="grey")
# sns.move_legend(ax, "upper left")
# sns.move_legend(
#     ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
# )
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(df["NewGsNumTot"].unique().size)
print(df["NewFunScoreSum"].unique().size)
plt.figure()
sns.histplot(data=df, x="NewGsNumTot", hue="prefix", kde=False, bins=200, alpha=0.5, linewidth=0)
plt.axvline(df["OldGsNumTot"].tolist()[0], 0, 6000, color="black")
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/GsNumTot_random_vs_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(df["NewGsNumTot"].unique().size)
print(df["NewFunScoreSum"].unique().size)
plt.figure()
sns.histplot(data=df, x="NewFunScoreSum", hue="prefix", kde=False, bins=200, alpha=0.5, linewidth=0)
plt.axvline(df["OldFunScoreSum"].tolist()[0], 0, 6000, color="black")
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/FuncScoreSum_random_vs_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    df[df["prefix"] == "params"], x="NewGsNumTot", y="NewFunScoreSum",
    bins=100, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_hist_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    df[df["prefix"] == "random"], x="NewGsNumTot", y="NewFunScoreSum",
    bins=100, discrete=(False, False), log_scale=(False, False),
    color=sns.color_palette()[1],
    # pthresh=.05, pmax=.9,
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_hist_random.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.kdeplot(
    df[df["prefix"] == "params"], x="NewGsNumTot", y="NewFunScoreSum",
    # bins=100, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
    shade=True, bw_adjust=.99, 
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_kde_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.kdeplot(
    df[df["prefix"] == "random"], x="NewGsNumTot", y="NewFunScoreSum",
    # bins=50, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
    color=sns.color_palette()[1],
    shade=True, bw_adjust=.99, 
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_kde_random.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.kdeplot(
    df[df["prefix"] == "params"], x="NewGsNumTot", y="NewFunScoreSum",
    # bins=100, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
    # shade=True, bw_adjust=.99, 
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_contour_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.kdeplot(
    df[df["prefix"] == "random"], x="NewGsNumTot", y="NewFunScoreSum",
    # bins=50, discrete=(False, False), log_scale=(False, False),
    # pthresh=.05, pmax=.9,
    color=sns.color_palette()[1],
    # shade=True, bw_adjust=.99, 
)
# plt.savefig(f"{results_dir}/NewGsNum_vs_NewFunScoreSum_contour_random.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
sns.color_palette()

# %% [markdown]
# ### Generating all the figures for manuscript
#
# #### Checking the data tables

# %%
# PSP_sites

# %%
# GS_sites

# %%
# FUN_sites

# %%
# PSP_sites["GoldStandard"] = [True if site in GS_sites["Site"].tolist() else False for site in PSP_sites["Site"].tolist()]
# PSP_sites["FuncScore"] = [FUN_sites[FUN_sites["Site"] == site]["Functional_score"].tolist()[0] if site in FUN_sites["Site"].tolist() else 0 for site in PSP_sites["Site"].tolist()]
# PSP_sites

# %%
# PSP_sites["HTP"] = PSP_sites["MS_LIT"].fillna(0) + PSP_sites["MS_CST"].fillna(0)
# PSP_sites["LTP"] = PSP_sites["LT_LIT"].fillna(0)
# PSP_sites

# %%
# PSP_sites.to_pickle("./phospho_site_plus_human_clustered_indexed_by_name.pkl")
# PSP_sites.to_csv("./phospho_site_plus_human_clustered_indexed_by_name.csv")

# %%
print(PSP_sites.columns)

# %% [markdown]
# #### Create proper distribution plots for the reassignment scores

# %%
plt.figure()
ax = sns.scatterplot(
    PSP_sites.replace({0.0, np.nan}), x="HTP", y="LTP", hue="GoldStandard"
)
ax.set(xscale="log", yscale="log")
# plt.savefig(f"{results_dir}/GSinPSP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
ax = sns.boxenplot(
# ax = sns.boxplot(
# ax = sns.violinplot(
    PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="LTP", orient='h', hue="GoldStandard",# showfliers=False,
    # whis=[0, 100], 
    # width=0.5,
)
# Add in points to show each observation
# sns.stripplot(PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="LTP", size=4, color=".3")
# ax.set(xscale="log", yscale="log")
ax.set(xscale="log")
# sns.despine(trim=True, left=True)
# plt.savefig(f"{results_dir}/GSinLTP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
ax = sns.boxenplot(
# ax = sns.boxplot(
# ax = sns.violinplot(
    PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="HTP", orient='h', hue="GoldStandard",# showfliers=False,
    # whis=[0, 100], 
    # width=0.5,
)
# Add in points to show each observation
# sns.stripplot(PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="LTP", size=4, color=".3")
# ax.set(xscale="log", yscale="log")
ax.set(xscale="log")
# sns.despine(trim=True, left=True)
# plt.savefig(f"{results_dir}/GSinHTP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
ax = sns.histplot(
    PSP_sites.replace({0.0, np.nan}), x="HTP", hue="GoldStandard", bins=30
)
# ax.set(xscale="log", yscale="log")
ax.set(yscale="log")
# plt.savefig(f"{results_dir}/GSinHTP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
ax = sns.scatterplot(
    PSP_sites.replace({0.0, np.nan}), x="HTP", y="LTP", hue="FuncScore", 
    palette="flare",
)
ax.set(xscale="log", yscale="log")
# plt.savefig(f"{results_dir}/FUNinPSP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
PSP_sites["FunLevel"] = np.nan
# PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0) & (PSP_sites["FuncScore"] < 0.2)] = "Low"
# PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.4) & (PSP_sites["FuncScore"] < 0.6)] = "Mid"
# PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.8) & (PSP_sites["FuncScore"] < 1.0)] = "High"
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0) & (PSP_sites["FuncScore"] < 0.1)] = 0
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.1) & (PSP_sites["FuncScore"] < 0.3)] = 0.2
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.3) & (PSP_sites["FuncScore"] < 0.5)] = 0.4
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.5) & (PSP_sites["FuncScore"] < 0.7)] = 0.6
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.7) & (PSP_sites["FuncScore"] < 0.9)] = 0.8
PSP_sites["FunLevel"][(PSP_sites["FuncScore"] > 0.9) & (PSP_sites["FuncScore"] < 1.0)] = 1.0

# %%
plt.figure()
ax = sns.boxenplot(
# ax = sns.boxplot(
# ax = sns.violinplot(
    PSP_sites.replace({0.0, np.nan}), y="FunLevel", x="LTP", orient='h', hue="FunLevel",# showfliers=False,
    palette="flare",
    # whis=[0, 100], 
    # width=0.5,
)
# Add in points to show each observation
# sns.stripplot(PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="LTP", size=4, color=".3")
# ax.set(xscale="log", yscale="log")
ax.set(xscale="log")
# sns.despine(trim=True, left=True)
# plt.savefig(f"{results_dir}/FUNinLTP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
ax = sns.boxenplot(
# ax = sns.boxplot(
# ax = sns.violinplot(
    PSP_sites.replace({0.0, np.nan}), y="FunLevel", x="HTP", orient='h', hue="FunLevel", # showfliers=False,
    palette="flare",
    # whis=[0, 100], 
    # width=0.5,
)
# Add in points to show each observation
# sns.stripplot(PSP_sites.replace({0.0, np.nan}), y="GoldStandard", x="LTP", size=4, color=".3")
# ax.set(xscale="log", yscale="log")
ax.set(xscale="log")
# sns.despine(trim=True, left=True)
# plt.savefig(f"{results_dir}/FUNinHTP.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
PSP_sites

# %%
plt.figure()
sns.histplot(
    PSP_sites[PSP_sites["GoldStandard"] == True].replace({0.0: np.nan}), x="HTP", y="LTP",# hue="Type", multiple="stack",
    bins=18, 
    discrete=(False, False), 
    log_scale=(True, True),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/GSinPSP_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    PSP_sites[PSP_sites["FuncScore"] > 0.5].replace({0.0: np.nan}), x="HTP", y="LTP",# hue="Type", multiple="stack",
    bins=18, 
    # discrete=(True, True), 
    discrete=(False, False), 
    log_scale=(True, True),
    # log_scale=(False, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/HighFUNinPSP_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
psp_melted = PSP_sites.melt(id_vars=["Site", "GoldStandard", "FuncScore"], value_vars=["HTP", "LTP"], var_name="Type", value_name="Evidence")
psp_melted["GoldStandard"] = psp_melted["GoldStandard"].astype(str)
psp_melted["FuncScoreCat"] = ["High" if score > 0.5 else "Low" for score in psp_melted["FuncScore"].tolist()]
psp_melted

# %%
plt.figure()
sns.histplot(
    psp_melted[psp_melted["GoldStandard"] == "True"].replace({0.0: np.nan}), x = "Evidence", hue = "Type",# multiple="stack",
    bins=15, 
    # discrete=(False, False), 
    log_scale=(True, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/GSinPSP_True_hist.pdf", dpi = 300, bbox_inches = 'tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    psp_melted[psp_melted["FuncScoreCat"] == "High"].replace({0.0: np.nan}), x="Evidence", hue="Type",# multiple="stack",
    bins=15, 
    # discrete=(False, False), 
    log_scale=(True, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/FUNinPSP_High_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    psp_melted[psp_melted["GoldStandard"] == "True"].replace({0.0: np.nan}), x="Evidence", hue="GoldStandard", multiple="stack",
    bins=15, 
    # discrete=(False, False), 
    # hue_order=["True", "False"],
    log_scale=(True, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/GSinPSP_LTP_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    psp_melted[psp_melted["Type"] == "HTP"].replace({0.0: np.nan}), x="Evidence", y="FuncScore",# hue="Type", multiple="stack",
    bins=15, 
    discrete=(False, False), 
    log_scale=(True, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/FUNinPSP_HTP_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.histplot(
    psp_melted[psp_melted["Type"] == "LTP"].replace({0.0: np.nan}), x="Evidence", y="FuncScore",# hue="Type", multiple="stack",
    bins=15, 
    discrete=(False, False), 
    log_scale=(True, False),
    # color=sns.color_palette(),
    # pthresh=.05, pmax=.9,
    # thresh=None,
)
# plt.savefig(f"{results_dir}/FUNinPSP_LTP_hist.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# #### Adding the quantile information in the reassignment table

# %%
df["GS_quant"] = np.nan
df["FUN_quant"] = np.nan
df["FPR_quant"] = np.nan
df["TPR_quant"] = np.nan
df["TPR*GS_quant"] = np.nan
df["TPR*FUN_quant"] = np.nan
df["GS*FUN_quant"] = np.nan
df["TPR*GS*FUN_quant"] = np.nan
# for prefix in ["params", "random"]:
prefix = "params"

gs = df[(df["prefix"] == prefix)]["DeltaGsNum"].to_numpy()
fun = df[(df["prefix"] == prefix)]["DeltaFunScore"].to_numpy()
fpr = df[(df["prefix"] == prefix)]["FPR"].to_numpy()
tpr = df[(df["prefix"] == prefix)]["TPR"].to_numpy()
tprxgs = df[(df["prefix"] == prefix)]["TPR*GS"].to_numpy()
tprxfun = df[(df["prefix"] == prefix)]["TPR*FUN"].to_numpy()
gsxfun = df[(df["prefix"] == prefix)]["GS*FUN"].to_numpy()
tprxgsxfun = df[(df["prefix"] == prefix)]["TPR*GS*FUN"].to_numpy()
df.loc[(df["prefix"] == prefix), "GS_quant"] = get_quant(gs)
df.loc[(df["prefix"] == prefix), "FUN_quant"] = get_quant(fun)
df.loc[(df["prefix"] == prefix), "FPR_quant"] = get_quant(fpr)
df.loc[(df["prefix"] == prefix), "TPR_quant"] = get_quant(tpr)
df.loc[(df["prefix"] == prefix), "TPR*GS_quant"] = get_quant(tprxgs)
df.loc[(df["prefix"] == prefix), "TPR*FUN_quant"] = get_quant(tprxfun)
df.loc[(df["prefix"] == prefix), "GS*FUN_quant"] = get_quant(gsxfun)
df.loc[(df["prefix"] == prefix), "TPR*GS*FUN_quant"] = get_quant(tprxgsxfun)

gs = df[(df["prefix"] != prefix)]["DeltaGsNum"].to_numpy()
fun = df[(df["prefix"] != prefix)]["DeltaFunScore"].to_numpy()
fpr = df[(df["prefix"] != prefix)]["FPR"].to_numpy()
tpr = df[(df["prefix"] != prefix)]["TPR"].to_numpy()
tprxgs = df[(df["prefix"] != prefix)]["TPR*GS"].to_numpy()
tprxfun = df[(df["prefix"] != prefix)]["TPR*FUN"].to_numpy()
gsxfun = df[(df["prefix"] != prefix)]["GS*FUN"].to_numpy()
tprxgsxfun = df[(df["prefix"] != prefix)]["TPR*GS*FUN"].to_numpy()
df.loc[(df["prefix"] != prefix), "GS_quant"] = get_quant(gs)
df.loc[(df["prefix"] != prefix), "FUN_quant"] = get_quant(fun)
df.loc[(df["prefix"] != prefix), "FPR_quant"] = get_quant(fpr)
df.loc[(df["prefix"] != prefix), "TPR_quant"] = get_quant(tpr)
df.loc[(df["prefix"] != prefix), "TPR*GS_quant"] = get_quant(tprxgs)
df.loc[(df["prefix"] != prefix), "TPR*FUN_quant"] = get_quant(tprxfun)
df.loc[(df["prefix"] != prefix), "GS*FUN_quant"] = get_quant(gsxfun)
df.loc[(df["prefix"] != prefix), "TPR*GS*FUN_quant"] = get_quant(tprxgsxfun)

# %%
df

# %% [markdown]
# #### Venn Diagrams of different quantiles

# %%
dfp = df[df["prefix"] == "params"].copy()
dfp.columns

# %%
dfp_gs_top01 = dfp[(dfp["NewGsNumTot"] > dfp["NewGsNumTot"].quantile(0.99))].copy()
dfp_gs_bot01 = dfp[(dfp["NewGsNumTot"] < dfp["NewGsNumTot"].quantile(0.01))].copy()
dfp_func_top01 = dfp[(dfp["NewFunScoreSum"] > dfp["NewFunScoreSum"].quantile(0.99))].copy()
dfp_func_bot01 = dfp[(dfp["NewFunScoreSum"] < dfp["NewFunScoreSum"].quantile(0.01))].copy()
dfp_ascore_top01 = dfp[(dfp["Increase/Decrease"] > dfp["Increase/Decrease"].quantile(0.99))].copy()
dfp_ascore_bot01 = dfp[(dfp["Increase/Decrease"] < dfp["Increase/Decrease"].quantile(0.01))].copy()

# %%
dfp_gs_top05 = dfp[(dfp["NewGsNumTot"] > dfp["NewGsNumTot"].quantile(0.95))].copy()
dfp_gs_bot05 = dfp[(dfp["NewGsNumTot"] < dfp["NewGsNumTot"].quantile(0.05))].copy()
dfp_func_top05 = dfp[(dfp["NewFunScoreSum"] > dfp["NewFunScoreSum"].quantile(0.95))].copy()
dfp_func_bot05 = dfp[(dfp["NewFunScoreSum"] < dfp["NewFunScoreSum"].quantile(0.05))].copy()
dfp_ascore_top05 = dfp[(dfp["Increase/Decrease"] > dfp["Increase/Decrease"].quantile(0.95))].copy()
dfp_ascore_bot05 = dfp[(dfp["Increase/Decrease"] < dfp["Increase/Decrease"].quantile(0.05))].copy()

# %%
from matplotlib_venn import venn3

# %%
plt.figure()
venn3([set(dfp_gs_top05["instance"]), set(dfp_func_top05["instance"]), set(dfp_ascore_top05["instance"])], ('Best 5% GS', 'Best 5% FuncScore', 'Best 5% Accuracy'))
# plt.savefig(f"{results_dir}/venn3_best05.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
venn3([set(dfp_gs_bot05["instance"]), set(dfp_func_bot05["instance"]), set(dfp_ascore_bot05["instance"])], ('Worst 5% GS', 'Worst 5% FuncScore', 'Worst 5% Accuracy'))
# plt.savefig(f"{results_dir}/venn3_worst05.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# #### Test plots for different quantiles

# %%
gs_top01 = dfp_gs_top01.melt(id_vars=["NewGsNumTot"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
gs_top01["Type"] = "Best 1%"
gs_top01["Metric"] = "GS"
gs_bot01 = dfp_gs_bot01.melt(id_vars=["NewGsNumTot"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
gs_bot01["Type"] = "Worst 1%"
gs_bot01["Metric"] = "GS"

# %%
gs_top01

# %%
gs_top01 = dfp_gs_top01.melt(id_vars=["NewGsNumTot"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
gs_top01["Type"] = "Best 1%"
gs_top01["Metric"] = "GS"
gs_bot01 = dfp_gs_bot01.melt(id_vars=["NewGsNumTot"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
gs_bot01["Type"] = "Worst 1%"
gs_bot01["Metric"] = "GS"

func_top01 = dfp_func_top01.melt(id_vars=["NewFunScoreSum"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
func_top01["Type"] = "Best 1%"
func_top01["Metric"] = "FUNC"
func_bot01 = dfp_func_bot01.melt(id_vars=["NewFunScoreSum"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
func_bot01["Type"] = "Worst 1%"
func_bot01["Metric"] = "FUNC"

ascore_top01 = dfp_ascore_top01.melt(id_vars=["Increase/Decrease"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
ascore_top01["Type"] = "Best 1%"
ascore_top01["Metric"] = "ASCORE"
ascore_bot01 = dfp_ascore_bot01.melt(id_vars=["Increase/Decrease"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
ascore_bot01["Type"] = "Worst 1%"
ascore_bot01["Metric"] = "ASCORE"

dfp_melt = pd.concat([gs_top01, gs_bot01, func_top01, func_bot01, ascore_top01, ascore_bot01])
dfp_melt = dfp_melt[dfp_melt.columns.tolist()[-2:] + dfp_melt.columns.tolist()[:-2]]
dfp_melt["Log10(W)"] = dfp_melt["ParameterValues"].apply(lambda x: np.log10(x))

# %%
dfp_melt

# %%
plt.figure()
sns.boxplot(data=dfp_melt[dfp_melt["Metric"] == "FUNC"], x="Parameter", y="Log10(W)", hue="Type")
# plt.savefig(f"{results_dir}/FUNC_Ws_box.pdf", dpi = 300, bbox_inches='tight')
plt.show()

# %%
plt.figure()
sns.boxplot(data=dfp_melt[dfp_melt["Metric"] == "ASCORE"], x="Parameter", y="Log10(W)", hue="Type")
# plt.savefig(f"{results_dir}/ASCORE_Ws_box.pdf", dpi = 300, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Study the 0.05 quantile of the reassignment scores

# %%
percentage = 5

# %%
df["FunCategory"] = "None"

# df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

df.loc[df["FUN_quant"] > 1 - (percentage/100), "FunCategory"] = f"Best {percentage}%"
df.loc[df["FUN_quant"] < percentage/100, "FunCategory"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
dfle = df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))].copy()
dfle_melt = dfle.melt(id_vars=["FunCategory"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
dfle_melt["Log10(W)"] = dfle_melt["ParameterValues"].apply(lambda x: np.log10(x))
plt.figure()
sns.boxenplot(data=dfle_melt, x="Parameter", y="ParameterValues", hue="FunCategory", hue_order=[f"Best {percentage}%", f"Worst {percentage}%"], showfliers=True)
plt.xlabel("Parameter")  # Set x and y axis labels
plt.ylabel("ParameterValues")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
# plt.title(f"Parameter Distributions of Top {percentage}% ({dfle[dfle["FunCategory"] == f'Best {percentage}%'].shape[0]}) and Worst {percentage}% ({dfle[dfle["FunCategory"] == f'Worst {percentage}%'].shape[0]}) of Average Changes on Functional Score")
plt.savefig(f"{results_dir}/ParameterDistributions_{percentage}_percent_fun.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df["GsCategory"] = "None"

# df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

df.loc[df["GS_quant"] > 1 - (percentage/100), "GsCategory"] = f"Best {percentage}%"
df.loc[df["GS_quant"] < percentage/100, "GsCategory"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
dfle = df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))].copy()
dfle_melt = dfle.melt(id_vars=["GsCategory"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
dfle_melt["Log10(W)"] = dfle_melt["ParameterValues"].apply(lambda x: np.log10(x))
plt.figure()
sns.boxenplot(data=dfle_melt, x="Parameter", y="ParameterValues", hue="GsCategory", hue_order=[f"Best {percentage}%", f"Worst {percentage}%"], showfliers=True)
plt.xlabel("Parameter")  # Set x and y axis labels
plt.ylabel("ParameterValues")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
# plt.title(f"Parameter Distributions of Top {percentage}% ({dfle[dfle["GsCategory"] == f'Best {percentage}%'].shape[0]}) and Worst {percentage}% ({dfle[dfle["GsCategory"] == f'Worst {percentage}%'].shape[0]}) of Average Changes on Gsctional Score")
plt.savefig(f"{results_dir}/ParameterDistributions_{percentage}_percent_gs.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df["Category"] = "None"

# df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Best {percentage}%"
df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Worst {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
dfle = df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))].copy()
dfle_melt = dfle.melt(id_vars=["Category"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
dfle_melt["Log10(W)"] = dfle_melt["ParameterValues"].apply(lambda x: np.log10(x))
plt.figure()
sns.boxenplot(data=dfle_melt, x="Parameter", y="ParameterValues", hue="Category", hue_order=[f"Best {percentage}%", f"Worst {percentage}%"], showfliers=True)
# sns.violinplot(data=dfle_melt, x="Parameter", y="ParameterValues", hue="Category", hue_order=[f"Top {percentage}%", f"Bottom {percentage}%"])
plt.xlabel("Parameter")  # Set x and y axis labels
plt.ylabel("ParameterValues")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.title(f"Parameter Distributions of Top {percentage}% (" + str(dfle[dfle["Category"] == f'Best {percentage}%'].shape[0]) + f") and Worst {percentage}% (" + str(dfle[dfle["Category"] == f'Worst {percentage}%'].shape[0]) + ")")
plt.savefig(f"{results_dir}/ParameterDistributions_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df["Category"] = "None"

df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Best {percentage}%"
df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
dfle = df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))].copy()
dfle_melt = dfle.melt(id_vars=["Category"], value_vars=["MS_W", "LT_W", "DT_W"], var_name="Parameter", value_name="ParameterValues")
dfle_melt["Log10(W)"] = dfle_melt["ParameterValues"].apply(lambda x: np.log10(x))
plt.figure()
sns.boxenplot(data=dfle_melt, x="Parameter", y="ParameterValues", hue="Category", hue_order=[f"Best {percentage}%", f"Worst {percentage}%"], showfliers=True)
plt.xlabel("Parameter")  # Set x and y axis labels
plt.ylabel("ParameterValues")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
# plt.title(f"Parameter Distributions of Top {percentage}% ({dfle[dfle["Category"] == f'Best {percentage}%'].shape[0]}) and Worst {percentage}% ({dfle[dfle["Category"] == f'Worst {percentage}%'].shape[0]})")
# plt.title(f"Parameter Distributions of Top {percentage}% (" + str(dfle[dfle["Category"] == f'Best {percentage}%'].shape[0]) + f") and Worst {percentage}% (" + str(dfle[dfle["Category"] == f'Worst {percentage}%'].shape[0]) + ")")
plt.savefig(f"{results_dir}/ParameterDistributions_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
df_top5 = df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%"))].copy()
best_parms_medians = [df_top5["MS_W"].median(), df_top5["LT_W"].median(), df_top5["DT_W"].median()]
print(best_parms_medians)

# %%
df_fun_top5 = df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%"))].copy()
best_fun_parms_medians = [df_fun_top5["MS_W"].median(), df_fun_top5["LT_W"].median(), df_fun_top5["DT_W"].median()]
print(best_fun_parms_medians)

# %%
df_gs_top5 = df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%"))].copy()
best_gs_parms_medians = [df_gs_top5["MS_W"].median(), df_gs_top5["LT_W"].median(), df_gs_top5["DT_W"].median()]
print(best_gs_parms_medians)

# %%
# df.to_csv(f"{results_dir}/all_summary_rand-param-new.csv", index=False)
df.to_pickle(f"{results_dir}/all_summary_rand-param-new.pkl")

# %%
df = pd.read_pickle(f"{results_dir}/all_summary_rand-param-new.pkl")

# %% [markdown]
# #### Explore the different Ascore ranges

# %%
print(list(range(0, 55, 5)))

# %%
ascores = list(range(0, 55, 5))
ascore_ranges = [f"({ascores[i-1]}-{ascores[i]}]" if i > 0 else "0" for i, ascore in enumerate(ascores)]
print(ascores)
print(ascore_ranges)

# %%
print(os.path.abspath(results_dir))

# %%
params_results = {}
for i, ascore in enumerate(ascores):
    key = ascore_ranges[i]
    sampling = pd.read_pickle(f"{results_dir}/sampling_summary_with_params_method_Ascore/sampling_summary_with_params_method_Ascore_{ascore}.pkl")
    sampling["Increase/Decrease"] = sampling["PosDeltaMaxAscoreNum"] / sampling["NegDeltaMaxAscoreNum"]
    sampling["FPR"] = sampling["NegDeltaMaxAscoreNum"] / (sampling["PosDeltaMaxAscoreNum"] + sampling["NegDeltaMaxAscoreNum"])
    sampling["TPR"] = sampling["PosDeltaMaxAscoreNum"] / (sampling["PosDeltaMaxAscoreNum"] + sampling["NegDeltaMaxAscoreNum"])
    sampling["AscoreRange"] = key
    params_results[key] = sampling

# %%
random_results = {}
for i, ascore in enumerate(ascores):
    key = ascore_ranges[i]
    sampling = pd.read_pickle(f"{results_dir}/sampling_summary_with_random_method_Ascore/sampling_summary_with_random_method_Ascore_{ascore}.pkl")
    sampling["Increase/Decrease"] = sampling["PosDeltaMaxAscoreNum"] / sampling["NegDeltaMaxAscoreNum"]
    sampling["FPR"] = sampling["NegDeltaMaxAscoreNum"] / (sampling["PosDeltaMaxAscoreNum"] + sampling["NegDeltaMaxAscoreNum"])
    sampling["TPR"] = sampling["PosDeltaMaxAscoreNum"] / (sampling["PosDeltaMaxAscoreNum"] + sampling["NegDeltaMaxAscoreNum"])
    sampling["AscoreRange"] = key
    random_results[key] = sampling

# %%
HTP_reassigned

# %%
Fun_medians_reassigned

# %%
Gs_medians_reassigned

# %%
summary_new_results = {}
for i, ascore in enumerate(ascores):
    summary_list = []
    for reassigned_all, params in zip([HTP_reassigned, Medians_reassigned, Fun_medians_reassigned, Gs_medians_reassigned], [HTP_params, Medians_params, Fun_medians_params, Gs_medians_params]):
        key = ascore_ranges[i]
        if i > 0:
            new_dat = reassigned_all[(reassigned_all["AScore"] > ascores[i-1]) & (reassigned_all["AScore"] <= ascore)].copy()
        else:
            new_dat = reassigned_all[(reassigned_all["AScore"] <= ascore)].copy()

        number_of_sites = new_dat.shape[0]
        
        MeanLog2ProbFC = np.mean(new_dat["log2ProbFC"])
        StdLog2ProbFC = np.std(new_dat["log2ProbFC"])
        MeanLog2OldProbFC = np.mean(new_dat["log2OldProbFC"])
        StdLog2OldProbFC = np.std(new_dat["log2OldProbFC"])
        PosDeltaMaxAscoreNum = np.sum(new_dat["deltaMaxAscore"] > 0)
        NegDeltaMaxAscoreNum = np.sum(new_dat["deltaMaxAscore"] < 0)
        PosDeltaPspMsScoreNum = np.sum(new_dat["deltaPspMsScore"] > 0)
        NegDeltaPspMsScoreNum = np.sum(new_dat["deltaPspMsScore"] < 0)
        PosDeltaPspLtScoreNum = np.sum(new_dat["deltaPspLtScore"] > 0)
        NegDeltaPspLtScoreNum = np.sum(new_dat["deltaPspLtScore"] < 0)
        PosDeltaPspScoreNum = np.sum(new_dat["deltaPspScore"] > 0)
        NegDeltaPspScoreNum = np.sum(new_dat["deltaPspScore"] < 0)
        PosDeltaDistScoreNum = np.sum(new_dat["deltaDistScore"] >= 0)
        NegDeltaDistScoreNum = np.sum(new_dat["deltaDistScore"] < 0)
        OldGsNumTot = np.sum(new_dat["OldGsNum"])
        NewGsNumTot = np.sum(new_dat["NewGsNum"])
        PosDeltaGsNum = np.sum(new_dat["deltaGsNum"] > 0)
        NegDeltaGsNum = np.sum(new_dat["deltaGsNum"] < 0)
        OldFunScoreSum = np.sum(new_dat["OldFunScore"])
        NewFunScoreSum = np.sum(new_dat["NewFunScore"])
        PosDeltaFunScoreNum = np.sum(new_dat["deltaFunScore"] > 0)
        NegDeltaFunScoreNum = np.sum(new_dat["deltaFunScore"] < 0)
        PosDeltaCombScoreNum = np.sum(new_dat["deltaCombScore"] > 0)
        NegDeltaCombScoreNum = np.sum(new_dat["deltaCombScore"] < 0)
        summary = [*params, number_of_sites, MeanLog2ProbFC, StdLog2ProbFC, MeanLog2OldProbFC, StdLog2OldProbFC, PosDeltaMaxAscoreNum, NegDeltaMaxAscoreNum, PosDeltaPspMsScoreNum, NegDeltaPspMsScoreNum, PosDeltaPspLtScoreNum, NegDeltaPspLtScoreNum, PosDeltaPspScoreNum, NegDeltaPspScoreNum, PosDeltaDistScoreNum, NegDeltaDistScoreNum, OldGsNumTot, NewGsNumTot, PosDeltaGsNum, NegDeltaGsNum, OldFunScoreSum, NewFunScoreSum, PosDeltaFunScoreNum, NegDeltaFunScoreNum, PosDeltaCombScoreNum, NegDeltaCombScoreNum]

        summary_list.append(summary)

    summary_df = pd.DataFrame(summary_list, columns=["prefix", "instance", "MS_W", "LT_W", "DT_W", "SiteNum", "MeanLog2ProbFC", "StdLog2ProbFC", "MeanLog2OldProbFC", "StdLog2OldProbFC", "PosDeltaMaxAscoreNum", "NegDeltaMaxAscoreNum", "PosDeltaPspMsScoreNum", "NegDeltaPspMsScoreNum", "PosDeltaPspLtScoreNum", "NegDeltaPspLtScoreNum", "PosDeltaPspScoreNum", "NegDeltaPspScoreNum", "PosDeltaDistScoreNum", "NegDeltaDistScoreNum", "OldGsNumTot", "NewGsNumTot", "PosDeltaGsNum", "NegDeltaGsNum", "OldFunScoreSum", "NewFunScoreSum", "PosDeltaFunScoreNum", "NegDeltaFunScoreNum", "PosDeltaCombScoreNum", "NegDeltaCombScoreNum"])

    summary_df["Increase/Decrease"] = summary_df["PosDeltaMaxAscoreNum"] / summary_df["NegDeltaMaxAscoreNum"]
    summary_df["FPR"] = summary_df["NegDeltaMaxAscoreNum"] / (summary_df["PosDeltaMaxAscoreNum"] + summary_df["NegDeltaMaxAscoreNum"])
    summary_df["TPR"] = summary_df["PosDeltaMaxAscoreNum"] / (summary_df["PosDeltaMaxAscoreNum"] + summary_df["NegDeltaMaxAscoreNum"])
    summary_df["AscoreRange"] = key
    summary_new_results[key] = summary_df

# %%
with open(f"{results_dir}/summary_new_results.pkl", "wb") as f:
    pickle.dump(summary_new_results, f)

# %%
with open(f"{results_dir}/summary_new_results.pkl", "rb") as f:
    summary_new_results = pickle.load(f)

# %%
summary_new_results[ascore_ranges[0]]

# %%
params_df = params_results["0"]
params_df

# %%
random_df = random_results["0"]
random_df

# %%
df_par = pd.concat(params_results.values(), ignore_index=True)
df_ran = pd.concat(random_results.values(), ignore_index=True)
df_new = pd.concat(summary_new_results.values(), ignore_index=True)

# %%
df_par

# %%
df_new

# %%
df_par[df_par["SiteNum"].isna()]

# %%
print(df_par["AscoreRange"].unique())
print(df_ran["AscoreRange"].unique())

# %%
print(df_par.columns)

# %%
print(df_ran.columns)

# %%
df_par["DeltaGsNum"] = df_par["NewGsNumTot"] - df_par["OldGsNumTot"]
df_par["DeltaFunScore"] = df_par["NewFunScoreSum"] - df_par["OldFunScoreSum"]

df_new["DeltaGsNum"] = df_new["NewGsNumTot"] - df_new["OldGsNumTot"]
df_new["DeltaFunScore"] = df_new["NewFunScoreSum"] - df_new["OldFunScoreSum"]

df_ran["DeltaGsNum"] = df_ran["NewGsNumTot"] - df_ran["OldGsNumTot"]
df_ran["DeltaFunScore"] = df_ran["NewFunScoreSum"] - df_ran["OldFunScoreSum"]

# %% [markdown]
# ##### Test the plot of different Ascore ranges

# %%
plt.figure()
ax = sns.boxenplot(data=df_par, x="AscoreRange", y="DeltaFunScore", showfliers=True)
# sns.violinplot(data=df_par, x="AscoreRange", y="DeltaFunScore")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Change of total functional score after reassignment")
# plt.xticks(list(range(0,61,2)))
ax.tick_params(axis='x', rotation=30)
# plt.savefig(f"{results_dir}/delta_functional_score_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.boxenplot(data=df_par, x="AscoreRange", y="DeltaGsNum", showfliers=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Change of total GS sites after reassignment")
# plt.xticks(list(range(0,61,2)))
# plt.yscale('log')
# plt.savefig(f"{results_dir}/delta_GS_sites_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
# sns.boxplot(data=dfpa, x="AscoreRange", y="Increase/Decrease", showfliers=False)
sns.boxenplot(data=df_par, x="AscoreRange", y="Increase/Decrease", showfliers=True)
# sns.violinplot(data=df_par, x="AscoreRange", y="Increase/Decrease")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
plt.axhline(1, 0, 6000, color="grey")
# plt.xticks(list(range(0,61,2)))
plt.yscale('log')
# plt.savefig(f"{results_dir}/reassignment_accuracy_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
# plt.savefig(f"{results_dir}/reassignment_accuracy_from_ascore_ranges_log.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
# sns.boxplot(data=dfpa, x="AscoreRange", y="Increase/Decrease", showfliers=False)
sns.boxenplot(data=df_par, x="AscoreRange", y="FPR", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("False Positive Rate of Reassignment")
plt.axhline(0.5, 0, 60, color="grey")
# plt.xticks(list(range(0,61,2)))
# plt.yscale('log')
# plt.savefig(f"{results_dir}/reassignment_accuracy_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
# plt.savefig(f"{results_dir}/reassignment_FPR_from_ascore_ranges_log.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
# sns.boxplot(data=dfpa, x="AscoreRange", y="Increase/Decrease", showfliers=False)
sns.boxenplot(data=df_par, x="AscoreRange", y="TPR", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("True Positive Rate of Reassignment")
plt.axhline(0.5, 0, 60, color="grey")
# plt.xticks(list(range(0,61,2)))
# plt.yscale('log')
# plt.savefig(f"{results_dir}/reassignment_accuracy_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
# plt.savefig(f"{results_dir}/reassignment_TPR_from_ascore_ranges_log.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# #### Study the distributions of the reassignment scores with different Ascore ranges

# %%
df = pd.concat([df_new, df_par, df_ran]).reset_index(drop=True)

# %%
df

# %%
df4[(df4["AScore"] > 5) & (df4["AScore"] <= 10)]

# %%
df.loc[(df["prefix"] == "params") & (df["AscoreRange"] == "(5-10]"), "SiteNum"] = df4[(df4["AScore"] > 5) & (df4["AScore"] <= 10)].shape[0]
df[(df["prefix"] == "params") & (df["AscoreRange"] == "(5-10]")]

# %%
df.loc[(df["prefix"] == "params") & (df["AscoreRange"] == "(10-15]"), "SiteNum"] = df4[(df4["AScore"] > 10) & (df4["AScore"] <= 15)].shape[0]
df[(df["prefix"] == "params") & (df["AscoreRange"] == "(10-15]")]

# %%
df[df["SiteNum"].isna()]

# %%
df[df["SiteNum"] <= 0]

# %%
df["MeanDeltaGsNum"] = df["DeltaGsNum"] / df["SiteNum"]
df["MeanDeltaFunScore"] = df["DeltaFunScore"] / df["SiteNum"]

# %%
df_bak = df.copy()

# %%
dfs = []
for i in ascore_ranges:
    # for prefix in ["params", "random"]:
    dfi = df[(df["prefix"] != "random") & (df["AscoreRange"] == i)].copy()
    gs_scaled = scale2unit(dfi["MeanDeltaGsNum"].to_numpy())
    fun_scaled = scale2unit(dfi["MeanDeltaFunScore"].to_numpy())
    tpr_scaled = scale2unit(dfi["TPR"].to_numpy())
    # dfi["TPR*GS"] = scale2unit(tpr_scaled * gs_scaled)
    # dfi["TPR*FUN"] = scale2unit(tpr_scaled * fun_scaled)
    # dfi["GS*FUN"] = scale2unit(gs_scaled * fun_scaled)
    # dfi["TPR*GS*FUN"] = scale2unit(tpr_scaled * gs_scaled * fun_scaled)
    dfi["TPR*GS"] = np.sqrt(tpr_scaled * gs_scaled)
    dfi["TPR*FUN"] = np.sqrt(tpr_scaled * fun_scaled)
    dfi["GS*FUN"] = np.sqrt(gs_scaled * fun_scaled)
    dfi["TPR*GS*FUN"] = np.sqrt(tpr_scaled * gs_scaled * fun_scaled)
    dfs.append(dfi)
    dfi = df[(df["prefix"] == "random") & (df["AscoreRange"] == i)].copy()
    gs_scaled = scale2unit(dfi["MeanDeltaGsNum"].to_numpy())
    fun_scaled = scale2unit(dfi["MeanDeltaFunScore"].to_numpy())
    tpr_scaled = scale2unit(dfi["TPR"].to_numpy())
    # dfi["TPR*GS"] = scale2unit(tpr_scaled * gs_scaled)
    # dfi["TPR*FUN"] = scale2unit(tpr_scaled * fun_scaled)
    # dfi["GS*FUN"] = scale2unit(gs_scaled * fun_scaled)
    # dfi["TPR*GS*FUN"] = scale2unit(tpr_scaled * gs_scaled * fun_scaled)
    dfi["TPR*GS"] = np.sqrt(tpr_scaled * gs_scaled)
    dfi["TPR*FUN"] = np.sqrt(tpr_scaled * fun_scaled)
    dfi["GS*FUN"] = np.sqrt(gs_scaled * fun_scaled)
    dfi["TPR*GS*FUN"] = np.sqrt(tpr_scaled * gs_scaled * fun_scaled)
    dfs.append(dfi)
df = pd.concat(dfs)

# %%
df["GS_quant"] = np.nan
df["FUN_quant"] = np.nan
df["FPR_quant"] = np.nan
df["TPR_quant"] = np.nan
df["TPR*GS_quant"] = np.nan
df["TPR*FUN_quant"] = np.nan
df["GS*FUN_quant"] = np.nan
df["TPR*GS*FUN_quant"] = np.nan
for i in ascore_ranges:
    # for prefix in ["params", "random"]:
    prefix = "random"

    gs = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["DeltaGsNum"].to_numpy()
    fun = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["DeltaFunScore"].to_numpy()
    fpr = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["FPR"].to_numpy()
    tpr = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["TPR"].to_numpy()
    tprxgs = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["TPR*GS"].to_numpy()
    tprxfun = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["TPR*FUN"].to_numpy()
    gsxfun = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["GS*FUN"].to_numpy()
    tprxgsxfun = df[(df["prefix"] == prefix) & (df["AscoreRange"] == i)]["TPR*GS*FUN"].to_numpy()
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "GS_quant"] = get_quant(gs)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "FUN_quant"] = get_quant(fun)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "FPR_quant"] = get_quant(fpr)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "TPR_quant"] = get_quant(tpr)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "TPR*GS_quant"] = get_quant(tprxgs)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "TPR*FUN_quant"] = get_quant(tprxfun)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "GS*FUN_quant"] = get_quant(gsxfun)
    df.loc[(df["prefix"] == prefix) & (df["AscoreRange"] == i), "TPR*GS*FUN_quant"] = get_quant(tprxgsxfun)    

    gs = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["DeltaGsNum"].to_numpy()
    fun = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["DeltaFunScore"].to_numpy()
    fpr = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["FPR"].to_numpy()
    tpr = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["TPR"].to_numpy()
    tprxgs = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["TPR*GS"].to_numpy()
    tprxfun = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["TPR*FUN"].to_numpy()
    gsxfun = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["GS*FUN"].to_numpy()
    tprxgsxfun = df[(df["prefix"] != prefix) & (df["AscoreRange"] == i)]["TPR*GS*FUN"].to_numpy()
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "GS_quant"] = get_quant(gs)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "FUN_quant"] = get_quant(fun)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "FPR_quant"] = get_quant(fpr)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "TPR_quant"] = get_quant(tpr)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "TPR*GS_quant"] = get_quant(tprxgs)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "TPR*FUN_quant"] = get_quant(tprxfun)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "GS*FUN_quant"] = get_quant(gsxfun)
    df.loc[(df["prefix"] != prefix) & (df["AscoreRange"] == i), "TPR*GS*FUN_quant"] = get_quant(tprxgsxfun)

# %%
df_bak2 = df.copy()

# %%
df.to_pickle(f"{results_dir}/all_summary_rand-param-new_annotated.pkl")

# %%
df = pd.read_pickle(f"{results_dir}/all_summary_rand-param-new_annotated.pkl")

# %% [markdown]
# ##### Plot the distributions of the reassignment scores

# %%
plt.figure()
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
# for h in list(set((df["OldFunScoreSum"]/df["SiteNum"]).tolist())):
#     plt.axhline(h, 0, 6000, color="black")
plt.axhline(0, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs_params_new.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(0, 0, 1, color="grey")
plt.axvline(0, 0, 1, color="grey")
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="viridis", alpha=0.5, s=5, linewidth=0)
# for h in list(set((df["OldFunScoreSum"]/df["SiteNum"]).tolist())):
#     plt.axhline(h, 0, 6000, color="black")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs_params_alt_new.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(0, 0, 1, color="grey")
plt.axvline(0, 0, 1, color="grey")

sns.scatterplot(data=df[df["prefix"] == "random"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
# for h in list(set((df["OldFunScoreSum"]/df["SiteNum"]).tolist())):
#     plt.axhline(h, 0, 6000, color="black")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs_random_new.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.scatterplot(data=df, x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
# for h in list(set((df["OldFunScoreSum"]/df["SiteNum"]).tolist())):
#     plt.axhline(h, 0, 6000, color="black")
plt.axhline(0, 0, 6000, color="black")
plt.axvline(0, 0, 6000, color="black")
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(0, 0, 6000, color="grey")
plt.axvline(0, 0, 6000, color="grey")

# sns.scatterplot(data=df, x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="viridis", alpha=0.5, s=5, linewidth=0)
sns.scatterplot(data=df[df["prefix"] == "random"], x="MeanDeltaGsNum", y="MeanDeltaFunScore", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# for h in list(set((df["OldFunScoreSum"]/df["SiteNum"]).tolist())):
#     plt.axhline(h, 0, 6000, color="black")
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs_alt_new.pdf", dpi = 300, bbox_inches='tight')
# plt.legend([],[], frameon=False)
plt.savefig(f"{results_dir}/MeanDeltaFun_vs_MeanDeltaGs_alt_new_with_legend.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# ##### Parameter distributions
#
# 1. Generate the parameter boxplots with different Ascore ranges. And do the Ascore depenpendt weighting reassignments

# %%
plt.figure()
sns.boxenplot(data=df[df["prefix"] == "params"], x="AscoreRange", y="MeanDeltaFunScore", showfliers=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Average changes of functional score")
plt.tick_params(axis='x', rotation=30)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/mean_delta_functional_score_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.boxenplot(data=df[df["prefix"] == "params"], x="AscoreRange", y="MeanDeltaGsNum", showfliers=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
plt.tick_params(axis='x', rotation=30)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"{results_dir}/mean_delta_gs_number_from_ascore_ranges.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
print(ascore_ranges)

# %%
percentage = 5

# %%
df["FunCategory"] = "None"

# df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

df.loc[df["FUN_quant"] > 1 - (percentage/100), "FunCategory"] = f"Best {percentage}%"
df.loc[df["FUN_quant"] < percentage/100, "FunCategory"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
df["GsCategory"] = "None"

# df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

df.loc[df["GS_quant"] > 1 - (percentage/100), "GsCategory"] = f"Best {percentage}%"
df.loc[df["GS_quant"] < percentage/100, "GsCategory"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
df["Category"] = "None"

df.loc[df["GS*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Best {percentage}%"
df.loc[df["GS*FUN_quant"] < percentage/100, "Category"] = f"Worst {percentage}%"

# df.loc[df["FUN_quant"] > 1 - (percentage/100), "Category"] = f"Best {percentage}%"
# df.loc[df["FUN_quant"] < percentage/100, "Category"] = f"Worst {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Top {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}% (FUN only)"
# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}% (GS only)"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Bottom {percentage}% (FUN only)"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
from turtle import color


plt.figure()
# sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="FunCategory", showfliers=False)
plt.axhline(y=0.0, color='k', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="FunCategory", showfliers=False)
sns.lineplot(data=df[(df["prefix"] == "HTP_only")], x="AscoreRange", y="MeanDeltaFunScore", color="b", label="HTP only")
sns.lineplot(data=df[(df["prefix"] == "FunMedians")], x="AscoreRange", y="MeanDeltaFunScore", color="r", label=f"Best {percentage}% Medians")
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="FunCategory", split=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Average changes of functional score")
plt.tick_params(axis='x', rotation=30)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/mean_delta_functional_score_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(y=0.0, color='k', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y="MeanDeltaGsNum", hue="GsCategory", showfliers=False)
sns.lineplot(data=df[(df["prefix"] == "HTP_only")], x="AscoreRange", y="MeanDeltaGsNum", color="b", label="HTP only")
sns.lineplot(data=df[(df["prefix"] == "GsMedians")], x="AscoreRange", y="MeanDeltaGsNum", color="r", label=f"Best {percentage}% Medians")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
plt.tick_params(axis='x', rotation=30)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/mean_delta_gs_number_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))], x="AscoreRange", y="FPR", hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
plt.tick_params(axis='x', rotation=30)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f"{results_dir}/FPR_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "MS_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="Category")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "LT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "DT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Best {percentage}%") | (df["Category"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "MS_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="FunCategory", showfliers=False)
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="FunCategory")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_FunCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "LT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="FunCategory", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_FunCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "DT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["FunCategory"] == f"Best {percentage}%") | (df["FunCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="FunCategory", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_FunCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "MS_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="GsCategory", showfliers=False)
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="GsCategory")
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_GsCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "LT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="GsCategory", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_GsCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "DT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["GsCategory"] == f"Best {percentage}%") | (df["GsCategory"] == f"Worst {percentage}%"))], x="AscoreRange", y=parameter, hue="GsCategory", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
plt.tick_params(axis='x', rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_GsCategory.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# ##### Explore the reassignment accuracy distribution

# %%
plt.figure()
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
plt.xlabel("Average changes of functional score")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
# plt.xscale('log')
# plt.yscale('log')
plt.axhline(0.5, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/DeltaFunScore_vs_TPR_params.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="viridis", alpha=0.5, s=5, linewidth=0)
plt.xlabel("Average changes of functional score")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
# plt.xscale('log')
# plt.yscale('log')
plt.axhline(0.5, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/DeltaFunScore_vs_TPR_params_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.scatterplot(data=df[df["prefix"] == "random"], x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
plt.xlabel("Average changes of functional score")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
# plt.xscale('log')
# plt.yscale('log')
plt.axhline(0.5, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/DeltaFunScore_vs_TPR_random.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
sns.scatterplot(data=df, x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
plt.xlabel("Average changes of functional score")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
# plt.xscale('log')
# plt.yscale('log')
plt.axhline(0.5, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/DeltaFunScore_vs_TPR.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
# sns.scatterplot(data=df, x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
sns.scatterplot(data=df[df["prefix"] == "random"], x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="cool", alpha=0.5, s=5, linewidth=0)
sns.scatterplot(data=df[df["prefix"] == "params"], x="MeanDeltaFunScore", y="TPR", hue="AscoreRange", palette="viridis", alpha=0.5, s=5, linewidth=0)
plt.xlabel("Average changes of functional score")  # Set x and y axis labels
plt.ylabel("Reassignment accuracy")
# plt.xscale('log')
# plt.yscale('log')
plt.axhline(0.5, 0, 1, color="black")
plt.axvline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/DeltaFunScore_vs_TPR_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# ##### Explore parameter distributions for other quantiles 

# %%
percentage = 25

# %%
df["Category"] = "None"

# df.loc[df["TPR*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["TPR*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)) & (df["TPR_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100) & (df["TPR_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

df.loc[(df["FUN_quant"] > 1 - (percentage/100)) & (df["TPR_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
df.loc[(df["FUN_quant"] < percentage/100) & (df["TPR_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="Category", showfliers=False)
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="Category", split=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Average changes of functional score")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/mean_delta_functional_score_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaGsNum", hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/mean_delta_gs_number_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="TPR", hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0.5, 0, 1, color="black")
plt.savefig(f"{results_dir}/FPR_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "MS_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "LT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "DT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
percentage = 15

# %%
df["Category"] = "None"

# df.loc[df["TPR*FUN_quant"] > 1 - (percentage/100), "Category"] = f"Top {percentage}%"
# df.loc[df["TPR*FUN_quant"] < percentage/100, "Category"] = f"Bottom {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) & (df["FUN_quant"] > 1 - (percentage/100)) & (df["TPR_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) & (df["FUN_quant"] < percentage/100) & (df["TPR_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

df.loc[(df["FUN_quant"] > 1 - (percentage/100)) & (df["TPR_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
df.loc[(df["FUN_quant"] < percentage/100) & (df["TPR_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# df.loc[(df["GS_quant"] > 1 - (percentage/100)) | (df["FUN_quant"] > 1 - (percentage/100)), "Category"] = f"Top {percentage}%"
# df.loc[(df["GS_quant"] < percentage/100) | (df["FUN_quant"] < percentage/100), "Category"] = f"Bottom {percentage}%"

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="Category", showfliers=False)
# sns.violinplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaFunScore", hue="Category", split=True)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Average changes of functional score")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/mean_delta_functional_score_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="MeanDeltaGsNum", hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0, 0, 1, color="black")
plt.savefig(f"{results_dir}/mean_delta_gs_number_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
plt.figure()
plt.axhline(y=0.0, color='r', linestyle='--')
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y="TPR", hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel("Changes in ratios of GS sites")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(0.5, 0, 1, color="black")
plt.savefig(f"{results_dir}/FPR_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "MS_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "LT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()

# %%
parameter = "DT_W"
plt.figure()
sns.boxenplot(data=df[(df["prefix"] == "params") & ((df["Category"] == f"Top {percentage}%") | (df["Category"] == f"Bottom {percentage}%"))], x="AscoreRange", y=parameter, hue="Category", showfliers=False)
plt.xlabel("AScore Range")  # Set x and y axis labels
plt.ylabel(f"Parameter value of {parameter}")
# plt.xticks(list(range(0,61,2)))
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f"{results_dir}/dist_of_{parameter}_from_ascore_ranges_{percentage}_percent_alt.pdf", dpi = 300, bbox_inches='tight')
plt.show()
plt.close()
