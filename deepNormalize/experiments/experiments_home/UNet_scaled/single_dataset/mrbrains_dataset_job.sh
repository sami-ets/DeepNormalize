#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=32G                 # memory (per node)
#SBATCH --time=05-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=UNet_home_scaled_mrbrains_dataset.out
#SBATCH --job-name=UNet_home_scaled_mrbrains_dataset
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /mnt/md0/Data/code/deepNormalizev5/main_cc_unet.py --config=/mnt/md0/Research/code/deepNormalizev5/deepNormalize/experiments/experiments_home/UNet_scaledsingle_dataset/config_mrbrains_dataset.yaml