#!/bin/bash

#在Gehler-Shi数据集上的结果，全域均匀候选光源，两注意力
TRAINED_MODEL_PATH="./my_output_exp/experiment3/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py my_conf_exp/experiment3.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH
