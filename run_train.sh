#!/bin/bash

# mobile
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network mobile0.25

# resnet50
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50

# vit
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network vit
