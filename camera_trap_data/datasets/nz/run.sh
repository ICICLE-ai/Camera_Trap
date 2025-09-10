#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=data_seq
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-node=1
##SBATCH --exclusive
#SBATCH --cpus-per-task=12
#SBATCH --array=0-99
#SBATCH --output=output/array_%a.out
# #SBATCH --output=output/%j_%x.slurm.out
# #SBATCH --dependency=afterany:27580776


# pitzer: 42 dual V100 w/32GB + 32 dual V100 w/16GB + 4 quad V100 w/32GB
# owens: 160 P100 w/16GB
# sinfo -o "%P %G %D %N" | grep gpuparallel to check gpu gres
# ex: gpu:v100-32g, gpu:v100, gpu:v100-quad are on pitzer
# #SBATCH --gres=gpu:v100:1 # specify particular gpus

# #SBATCH --mail-type=ALL
#SBATCH --mail-user=jeon.193@osu.edu

# set -x # each command in the batch file to be printed to the log file as it is executed
# module load cuda/11.6.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camera_trap

# srun python process.py $SLURM_ARRAY_TASK_ID
# srun python count.py
cd /fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/nz

srun python add_seq.py --node_id $SLURM_ARRAY_TASK_ID