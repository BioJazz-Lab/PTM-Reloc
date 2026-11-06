import pandas as pd
import numpy as np
import sys

timestamp = "20240110"
results_dir = "../{}_Optimizing_Reassignment".format(timestamp)

if __name__ == '__main__':

    
    ascore = int(sys.argv[1])
    interval = int(sys.argv[2])
    prefixes = str(sys.argv[3])
    
    # for prefix in ["random", "params"]:
    # for prefix in ["params"]:
    for prefix in [prefixes]:
        if prefix == "params":
            original_summary = pd.read_pickle("{}/sampling_summary_with_{}_method.pkl".format(results_dir, prefix))
        # for ascore in range(0, 55, 5):
        summary_list = []
        for i in range(10000):
            instance = i
            if prefix == "params":
                ori_instance = original_summary[original_summary["instance"] == instance]
                MS_W = ori_instance["MS_W"].values[0]
                LT_W = ori_instance["LT_W"].values[0]
                DT_W = ori_instance["DT_W"].values[0]
            else:
                MS_W = 1
                LT_W = 1
                DT_W = 1
            file = "{}/{}_reassignment_{}.pkl".format(results_dir, prefix, i)
            new_dat_all = pd.read_pickle(file)    # Now return a summarized list of sums of individual column
            new_dat = new_dat_all[(new_dat_all["AScore"] > ascore - interval) & (new_dat_all["AScore"] <= ascore)].copy()
            number_of_sites = new_dat.shape[0]
            
            # new_dat["log2ProbFC"] = np.log2(new_dat["ParamProb"] / new_dat["RandProb"])
            # new_dat["log2OldProbFC"] = np.log2(new_dat["OldParamProb"] / new_dat["RandProb"])
            # new_dat["deltaMaxAscore"] = new_dat["NewMaxAscore"] - new_dat["OldMaxAscore"]
            # new_dat["deltaPspMsScore"] = new_dat["NewPspMsScore"] - new_dat["OldPspMsScore"]
            # new_dat["deltaPspLtScore"] = new_dat["NewPspLtScore"] - new_dat["OldPspLtScore"]
            # new_dat["deltaPspScore"] = new_dat["NewPspScore"] - new_dat["OldPspScore"]
            # new_dat["deltaDistScore"] = new_dat["NewDistScore"] - new_dat["OldDistScore"]
            # new_dat["deltaGsNum"] = new_dat["NewGsNum"] - new_dat["OldGsNum"]
            # # new_dat["deltaGsScore"] = new_dat["NewGsScore"] - new_dat["OldGsScore"]
            # new_dat["deltaFunScore"] = new_dat["NewFunScore"] - new_dat["OldFunScore"]
            # new_dat["deltaCombScore"] = new_dat["NewCombScore"] - new_dat["OldCombScore"]
            # new_dat["deltaProb"] = np.log2(new_dat["ParamProb"] / new_dat["RandProb"])
            # Option: save the reassigned data to a file
            # new_dat.to_pickle("{}/{}_reassignment_{}.pkl".format(results_dir, prefix, instance))

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
            summary = [prefix, instance, MS_W, LT_W, DT_W, number_of_sites, MeanLog2ProbFC, StdLog2ProbFC, MeanLog2OldProbFC, StdLog2OldProbFC, PosDeltaMaxAscoreNum, NegDeltaMaxAscoreNum, PosDeltaPspMsScoreNum, NegDeltaPspMsScoreNum, PosDeltaPspLtScoreNum, NegDeltaPspLtScoreNum, PosDeltaPspScoreNum, NegDeltaPspScoreNum, PosDeltaDistScoreNum, NegDeltaDistScoreNum, OldGsNumTot, NewGsNumTot, PosDeltaGsNum, NegDeltaGsNum, OldFunScoreSum, NewFunScoreSum, PosDeltaFunScoreNum, NegDeltaFunScoreNum, PosDeltaCombScoreNum, NegDeltaCombScoreNum]

            summary_list.append(summary)

        summary_df = pd.DataFrame(summary_list, columns=["prefix", "instance", "MS_W", "LT_W", "DT_W", "SiteNum", "MeanLog2ProbFC", "StdLog2ProbFC", "MeanLog2OldProbFC", "StdLog2OldProbFC", "PosDeltaMaxAscoreNum", "NegDeltaMaxAscoreNum", "PosDeltaPspMsScoreNum", "NegDeltaPspMsScoreNum", "PosDeltaPspLtScoreNum", "NegDeltaPspLtScoreNum", "PosDeltaPspScoreNum", "NegDeltaPspScoreNum", "PosDeltaDistScoreNum", "NegDeltaDistScoreNum", "OldGsNumTot", "NewGsNumTot", "PosDeltaGsNum", "NegDeltaGsNum", "OldFunScoreSum", "NewFunScoreSum", "PosDeltaFunScoreNum", "NegDeltaFunScoreNum", "PosDeltaCombScoreNum", "NegDeltaCombScoreNum"])

        summary_df.to_pickle("{}/sampling_summary_with_{}_method_Ascore_{}.pkl".format(results_dir, prefix, ascore))
    
