#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/datasets/config_iSEG_only.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/datasets/config_MRBrainS_only.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/generator_segmenter/config_generator_segmenter.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/generator_segmenter/config_generator_segmenter_iSEG.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/generator_segmenter/config_generator_segmenter_MRBrainS.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_0.00.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_0.15.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_0.25.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_0.50.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_0.75.yaml
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=deepNormalize/config/experiments/lambda/config_lambda_1.00.yaml

