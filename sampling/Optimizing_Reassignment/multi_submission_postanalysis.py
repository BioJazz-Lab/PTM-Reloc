import os, shutil, sys

name = "Reassignment"
interval = 5
threshold_vals_begin = list(range(0, 55, interval))
# threshold_vals = threshold_vals_begin + [thr for thr in range(61) if thr not in threshold_vals_begin]
# threshold_vals = [35, 50]

# loop_num = 50

# account = "ldrd_cheung2022"
account = "carbstor"
# time = "1:00:00" 
time = "4-0" 
queue = "shared"
node = '1'
core = "1" 
out = "out_thr"
err = "err_thr"
mail = "song.feng@pnnl.gov"
modules = ["gcc/11.2.0", "python/miniconda23.3.1"]
extras = [
        "source /share/apps/python/miniconda23.3.1/etc/profile.d/conda.sh", 
        "conda activate csbc",
        "export PATH=/people/feng626/.conda/envs/csbc/bin:$PATH",
        ]
workdir = "cd /people/feng626/csbc/egfr/BIML/Optimizing_Reassignment"

slurm_template = "#!/bin/sh\n#SBATCH -A " + account + "\n#SBATCH -t " + time + "\n#SBATCH -p " + queue + "\n#SBATCH -N " + str(node) + "\n#SBATCH -n " + str(core)

for prefix in ["random", "params"]:
    for threshold in threshold_vals_begin:
        slurm = slurm_template
        slurm += "\n#SBATCH -o " + out + '_' + str(prefix) +  '_' + str(threshold) + ".txt\n#SBATCH -e " + err + '_' + str(prefix) + '_' + str(threshold) + ".txt\n#SBATCH --mail-user=" + mail + "\n#SBATCH --mail-type " + "END" + "\n \nmodule purge\n"
        for module in modules:
            slurm += "module load " + module + '\n'
        slurm += '\n'
        for extra in extras:
            slurm += extra + '\n'
        slurm += '\n'
        slurm += workdir + "\n \n"

        # slurm += "max=" + str((threshold + 1) * loop_num - 1) + "\nfor i in `seq " + str(threshold * loop_num) + " $max`\ndo\n    python scripts/main.py -config=cyano_rna/cyano_rna_config.json -i=$i\ndone\n"
        slurm += "python postanalysis_summary.py " + str(threshold) + ' ' + str(interval) + ' ' + str(prefix) + "\n"

        filename = name + "_" + prefix + "_thr_" + str(threshold) + ".sbatch"
        with open(filename, 'w') as sbatch:
            sbatch.write(slurm)
        submission = "sbatch " + filename 
        os.system(submission)
