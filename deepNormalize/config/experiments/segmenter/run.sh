#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_iSEG.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_MRBrainS.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_min_max_scaler.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_min_max_scaler_iSEG.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_min_max_scaler_MRBrainS.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_quantile_scaler.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_quantile_scaler_MRBrainS.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_quantile_scaler_iSEG.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_standardized_scaler.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_standardized_scaler_MRBrainS.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../main.py --config=config_segmenter_standardized_scaler_iSEG.yaml