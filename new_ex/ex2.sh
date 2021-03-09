#!/bin/bash

#2021.3.5,新模型的table1-pre训练测试
TRAINED_MODEL_PATH="./new_ex_output/ex2"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex2_step1.json Multidataset nus_cube+ data/multidataset/nus_cube+/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py new_ex/ex2_step2.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_cube+/table1_pre_cube_nus_pretrain/0/model_best.pth.tar
