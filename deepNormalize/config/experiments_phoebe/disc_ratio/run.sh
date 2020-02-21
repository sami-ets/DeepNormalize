#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 ../../../../main.py --config=disc_ratio_0_25.yaml --use-amp --amp-opt-level=O0 > log_0_25.txt
CUDA_VISIBLE_DEVICES=0 python3 ../../../../main.py --config=disc_ratio_0_50.yaml --use-amp --amp-opt-level=O0 > log_0_50.txt
CUDA_VISIBLE_DEVICES=0 python3 ../../../../main.py --config=disc_ratio_0_75.yaml --use-amp --amp-opt-level=O0 > log_0_75.txt
CUDA_VISIBLE_DEVICES=0 python3 ../../../../main.py --config=disc_ratio_1_00.yaml --use-amp --amp-opt-level=O0 > log_1_00.txt
