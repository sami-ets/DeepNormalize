#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:v100l:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=92G                 # memory (per node)
#SBATCH --time=05-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=LSGAN_DCGAN_home_scaled_scaled_disc_ratio_0_001.out
#SBATCH --job-name=LSGAN_DCGAN_home_scaled_scaled_disc_ratio_0_001
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /mnt/md0/Data/code/deepNormalizev5/main_cc.py --config=/mnt/md0/Research/code/deepNormalizev5/deepNormalize/experiments/experiments_home/LSGAN_DCGAN/disc_ratio/config_disc_ratio_0.001.yaml