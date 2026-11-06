import os, shutil, sys

name = "bootstrapping"

# loop_num = 50

# account = "ldrd_cheung2022"
# account = "carbstor"
account = "proteometer"
# time = "1:00:00" 
time = "4-0" 
node = '1'
core = "64" 
out = "out_"
err = "err_"
mail = "song.feng@pnnl.gov"
modules = ["gcc/11.2.0", "python/miniconda23.3.1"]
extras = [
        "source /share/apps/python/miniconda23.3.1/etc/profile.d/conda.sh", 
        "conda activate csbc",
        "export PATH=/people/feng626/.conda/envs/csbc/bin:$PATH",
        ]
# workdir = "cd /people/feng626/cancer/egfr/BIML"
workdir = "cd /rcfs/projects/proteometer/csbc/egfr/BIML"

# slurm_template = "#!/bin/sh\n#SBATCH -A " + account + "\n#SBATCH -t " + time + "\n#SBATCH -N " + str(node) + "\n#SBATCH -n " + str(core)
slurm_template = "#!/bin/sh\n#SBATCH -A {}\n#SBATCH -t {}\n#SBATCH -N {}\n#SBATCH -n {}\n".format(account, time, node, core)


for method in ["random", "params"]:
    slurm = [slurm_template]
    slurm.append("\n#SBATCH -o {}{}.txt\n#SBATCH -e {}{}.txt\n#SBATCH --mail-user={}\n#SBATCH --mail-type END\n \nmodule purge\n".format(out, method, err, method, mail))
    for module in modules:
        slurm.append("module load {}\n\n".format(module))
    for extra in extras:
        slurm.append("{}\n".format(extra))
    slurm.append("{}\n\n".format(workdir))

    slurm.append("python Optimizing_Reassignment_Parallel_bootstrapping.py {} {}\n".format(core, method))

    filename = "{}_{}.sbatch".format(name, method)
    with open(filename, 'w') as sbatch:
        sbatch.write(''.join(slurm))
    submission = "sbatch {}".format(filename) 
    os.system(submission)
