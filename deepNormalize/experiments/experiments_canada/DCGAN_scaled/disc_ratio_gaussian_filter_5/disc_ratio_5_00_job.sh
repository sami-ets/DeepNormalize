#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:v100l:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=46G                 # memory (per node)
#SBATCH --time=05-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=DCGAN_canada_scaled_gaussian_filter_5_disc_ratio_5_00.out
#SBATCH --job-name=DCGAN_canada_scaled_gaussian_filter_5_disc_ratio_5_00
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /project/def-lombaert/pld2602/code/deepNormalizev5/main_cc.py --config=/project/def-lombaert/pld2602/code/deepNormalizev5/deepNormalize/experiments/experiments_canada/DCGAN_scaled/disc_ratio_gaussian_filter_5/config_disc_ratio_5.00.yaml