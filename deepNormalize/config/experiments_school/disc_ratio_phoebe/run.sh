#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_1.00.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_0.75.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_0.50.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_0.25.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_0.15.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_disc_ratio_0.00.yaml