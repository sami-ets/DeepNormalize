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
#SBATCH --output=DCGAN_canada_scaled_dual_dataset_disc_ratio_1_50_seed_8809.out
#SBATCH --job-name=DCGAN_canada_scaled_dual_dataset_disc_ratio_1_50_seed_8809
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /project/def-lombaert/pld2602/code/deepNormalizev5/main_cc.py --config=/project/def-lombaert/pld2602/code/deepNormalizev5/deepNormalize/experiments/experiments_canada/DCGAN_scaled/disc_ratio_dual_dataset/config_disc_ratio_1.50_seed_8809.yaml