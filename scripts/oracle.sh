#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2099
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=camera-trap-oracle
#SBATCH --time=06:00:00
#SBATCH --output=myjob.out.%j

module load cuda/11.8.0
source /users/PAS2119/jeonso193/miniconda3/etc/profile.d/conda.sh
conda activate /fs/ess/PAS2099/sooyoung/envs/camera_trap

cd /fs/ess/PAS2099/sooyoung/Camera_Trap
python main.py --camera ENO_C05 --config configs/oracle.yaml --train_val --train_test --wandb