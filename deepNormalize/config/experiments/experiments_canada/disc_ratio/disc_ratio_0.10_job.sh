#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=150G                 # memory (per node)
#SBATCH --time=3-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=config=config_disc_ratio_0_10.out
#SBATCH --job-name=config=config_disc_ratio_0_10
nvidia-smi
source /home/pld2602/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /project/def-lombaert/pld2602/code/deepNormalizev5/main_cc.py --config=/project/def-lombaert/pld2602/code/deepNormalizev5/config/experiments/experiments_canada/disc_ratio/config_disc_ratio_0_10.yaml