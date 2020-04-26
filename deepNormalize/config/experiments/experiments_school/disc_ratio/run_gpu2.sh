#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 ../../../../../main.py --config=config_disc_ratio_0.25.yaml
CUDA_VISIBLE_DEVICES=2 python3 ../../../../../main.py --config=config_disc_ratio_0.10.yaml
