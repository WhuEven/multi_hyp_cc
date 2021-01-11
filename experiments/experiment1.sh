#!/bin/bash

#在Gehler-Shi数据集上的结果
TRAINED_MODEL_PATH="./output_exp/experiment1/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py conf_exp/experiment1.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH
