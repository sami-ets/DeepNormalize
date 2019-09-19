#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/datasets/config_iSEG_only.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/datasets/config_MRBrainS_only.yaml

