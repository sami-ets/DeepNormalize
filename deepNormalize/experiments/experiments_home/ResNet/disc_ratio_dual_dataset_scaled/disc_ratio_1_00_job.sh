#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:v100l:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=64G                 # memory (per node)
#SBATCH --time=05-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=ResNet_home_disc_ratio_dual_dataset_scaled_1_00.out
#SBATCH --job-name=ResNet_home_disc_ratio_dual_dataset_scaled_1_00
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /mnt/md0/Data/code/deepNormalizev5/main_cc.py --config=/mnt/md0/Research/code/deepNormalizev5/deepNormalize/experiments/experiments_home/ResNet/disc_ratio_dual_dataset_scaled/config_disc_ratio_1.00.yaml