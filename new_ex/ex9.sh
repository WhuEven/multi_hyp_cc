#!/bin/bash

#Cube challenge上的结果

TRAINED_MODEL_PATH="./new_ex/ex9"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex9.json Cube plus data/cube/plus.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 inference.py new_ex/ex9.json Cube challenge data/cube/challenge.txt $TRAINED_MODEL_PATH/Cube/plus/table4/0/model_best.pth.tar -gpu 0 --outputfolder $TRAINED_MODEL_PATH/cube_test/
