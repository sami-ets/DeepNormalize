#!/bin/bash
#SBATCH --account=def-lombaert
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=20G                 # memory (per node)
#SBATCH --time=3-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --output=config=config_disc_ratio_1_00.out
#SBATCH --job-name=config=config_disc_ratio_1_00    # Job name
nvidia-smi
./python3 ../../../../../main.py --config=config_disc_ratio_1.00.yaml