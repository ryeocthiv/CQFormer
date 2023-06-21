#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_classifier.py -d EuroSAT
CUDA_VISIBLE_DEVICES=1 python train_classifier.py -d AID