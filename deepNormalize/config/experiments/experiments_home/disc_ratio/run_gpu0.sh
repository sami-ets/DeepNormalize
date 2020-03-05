#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 ../../../../../main.py --config=config_disc_ratio_1.00.yaml
CUDA_VISIBLE_DEVICES=0 python3 ../../../../../main.py --config=config_disc_ratio_0.50.yaml
CUDA_VISIBLE_DEVICES=0 python3 ../../../../../main.py --config=config_disc_ratio_0.10.yaml

