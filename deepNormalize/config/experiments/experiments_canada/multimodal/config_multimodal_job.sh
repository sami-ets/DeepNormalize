#!/bin/bash
#SBATCH --account=def-pld2602
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=32G                 # memory (per node)
#SBATCH --time=7-00:00            # time (DD-HH:MM)
#SBATCH --mail-user=pierre-luc.delisle@live.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=10
nvidia-smi
./python3 ../../../../../main.py --config=config_multimodal.yaml