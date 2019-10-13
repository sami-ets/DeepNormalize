#!/bin/bash
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_segmenter_train_iSEG_test_on_MRBrainS.yaml --amp-opt-level="O1"

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_segmenter_train_MRBrainS_test_on_iSEG.yaml --amp-opt-level="O1"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_segmenter_train_iSEG_test_on_MRBrainS.yaml --amp-opt-level="O1"

#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 ../../../../main.py --config=config_segmenter_train_MRBrainS_test_on_MRBrainS.yaml --amp-opt-level="O1"
