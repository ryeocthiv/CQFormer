#!/usr/bin/env bash

# for ((i=0;i <=10;i++));do
#     CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 3 
# done
cuda_id=1
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 2
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 3
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 4
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 5
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 6
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 7
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 8
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 16
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 32
CUDA_VISIBLE_DEVICES=${cuda_id} python train.py -d EuroSAT --num_colors 64