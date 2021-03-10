#!/bin/bash

#Table3，单设备训练，本文，NUS数据集

# Declare an array of string with type
declare -a CAMERAS=("canon_eos_1D_mark3" "canon_eos_600D" "fuji" "nikonD5200" "panasonic" "olympus" "sony" "samsung" )

TRAINED_MODEL_PATH="./new_ex_output/ex7"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex7_step1.json Multidataset cube+_shigehler data/multidataset/cube+_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

# Iterate the string array using for loop
for camera in ${CAMERAS[@]}; do
   CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py new_ex/ex7_step2.json data/nus/splits_multicam/$camera.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH/$camera/ --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/cube+_shigehler/table3-pretrain/0/model_best.pth.tar
done
