#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 ../../../../../main.py --config=config_disc_ratio_0.75.yaml
CUDA_VISIBLE_DEVICES=1 python3 ../../../../../main.py --config=config_disc_ratio_0.25.yaml
CUDA_VISIBLE_DEVICES=1 python3 ../../../../../main.py --config=config_disc_ratio_0.05.yaml
