#!/bin/bash

#在Gehler-Shi数据集上的结果,Table1-our
TRAINED_MODEL_PATH="./new_ex_output/ex1"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py new_ex/ex1.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH
