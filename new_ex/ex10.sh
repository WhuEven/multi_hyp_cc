#!/bin/bash

#Cube challenge上的结果，用Gehler-Shi和NUS预训练

TRAINED_MODEL_PATH="./new_ex/ex10"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex10_step1.json Multidataset nus_shigehler data/multidataset/nus_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex10_step2.json Cube plus data/cube/plus.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH  --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_shigehler/table-4-pretrain/0/model_best.pth.tar

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 inference.py new_ex/ex10_step2.json Cube challenge data/cube/challenge.txt $TRAINED_MODEL_PATH/Cube/plus/table-4-finetune/0/model_best.pth.tar -gpu 0 --outputfolder $TRAINED_MODEL_PATH/cube_test/
